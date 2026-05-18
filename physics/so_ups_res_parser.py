"""Parse optional SO UPS RES monthly context into a model-ready CSV cache.

The parser is conservative by design. It extracts only labelled monthly facts
from official PDF/HTML reports and leaves ambiguous fields empty for manual QA.
The prediction pipeline treats these features as optional and masks post-train
months unless explicitly enabled for diagnostics.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "external_energy" / "so_ups_res_monthly.csv"
MONTHS_RU = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "ма": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
}
SO_UPS_MONTH_SLUGS = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "aprl",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sept",
    10: "oct",
    11: "nov",
    12: "dec",
}
SO_UPS_MONTH_SLUG_ALIASES = {
    "yan": 1,
    "january": 1,
    "apr": 4,
    "mai": 5,
    "sep": 9,
}
SO_UPS_URL_TEMPLATE = "https://www.so-ups.ru/fileadmin/files/company/markets/{year}/res/res_{mon}_{yy}.pdf"
SO_UPS_YEAR_PAGE_TEMPLATE = "https://www.so-ups.ru/functioning/markets/surveys/renewable/{year}/"
FIELD_COLUMNS = [
    "installed_mw",
    "generation_month_mwh",
    "generation_ytd_mwh",
    "curtailment_hours_month",
    "max_curtailment_mw_month",
    "max_deviation_mw_month",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse SO UPS RES monthly reports into CSV.")
    parser.add_argument("inputs", nargs="*", help="PDF/HTML files or URLs.")
    parser.add_argument("--input-list", type=Path, help="Text file with one URL/path per line.")
    parser.add_argument(
        "--so-ups-months",
        help="Month list/range to fetch from official SO UPS URLs, e.g. 2024-09,2025-01:2025-12.",
    )
    parser.add_argument(
        "--so-ups-url-template",
        default=SO_UPS_URL_TEMPLATE,
        help="Official report URL template with {year}, {yy}, and {mon}.",
    )
    parser.add_argument(
        "--so-ups-years",
        help="Comma-separated official SO UPS renewable report archive years to crawl for PDF links.",
    )
    parser.add_argument(
        "--so-ups-year-page-template",
        default=SO_UPS_YEAR_PAGE_TEMPLATE,
        help="Official yearly archive URL template with {year}.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--region-pattern", default="Ростов", help="Region substring to keep.")
    parser.add_argument("--default-region", default="Ростовская область")
    parser.add_argument("--table-fallback", action="store_true", help="Use nearby region-line numbers when labels fail.")
    parser.add_argument(
        "--keep-unknown-months",
        action="store_true",
        help="Keep parsed rows whose report month could not be resolved. Default drops them from the CSV.",
    )
    parser.add_argument("--dump-json", type=Path, help="Optional parser diagnostics JSON path.")
    return parser.parse_args()


def _read_inputs(args: argparse.Namespace) -> list[str]:
    inputs = list(args.inputs)
    if args.input_list:
        inputs.extend(
            line.strip()
            for line in args.input_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if args.so_ups_months:
        inputs.extend(_expand_so_ups_months(args.so_ups_months, args.so_ups_url_template))
    if args.so_ups_years:
        inputs.extend(_expand_so_ups_year_pages(args.so_ups_years, args.so_ups_year_page_template))
    if not inputs:
        raise ValueError("Pass at least one report path/URL or --input-list.")
    return inputs


def _parse_month_token(token: str) -> pd.Timestamp:
    return pd.Timestamp(token.strip()).to_period("M").to_timestamp()


def _expand_so_ups_months(spec: str, template: str) -> list[str]:
    months = []
    for part in [item.strip() for item in spec.split(",") if item.strip()]:
        if ":" in part:
            start_raw, end_raw = part.split(":", 1)
            month = _parse_month_token(start_raw)
            end = _parse_month_token(end_raw)
            while month <= end:
                months.append(month)
                month = month + pd.offsets.MonthBegin(1)
        else:
            months.append(_parse_month_token(part))

    urls = []
    for month in months:
        urls.append(
            template.format(
                year=month.year,
                yy=str(month.year)[-2:],
                mon=SO_UPS_MONTH_SLUGS[int(month.month)],
            )
        )
    return urls


def _expand_so_ups_year_pages(spec: str, template: str) -> list[str]:
    urls = []
    for raw_year in [item.strip() for item in spec.split(",") if item.strip()]:
        year = int(raw_year)
        page_url = template.format(year=year)
        payload, _ = _fetch_bytes(page_url)
        html = payload.decode("utf-8", errors="ignore")
        for href in re.findall(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', html, flags=re.I):
            if href.startswith("fileadmin/"):
                href = "/" + href
            full_url = urljoin(page_url, href)
            if "/res/" in full_url or "renewable" in full_url.lower():
                urls.append(full_url)
    return sorted(set(urls))


def _fetch_bytes(source: str) -> tuple[bytes, str]:
    if source.startswith(("http://", "https://")):
        try:
            import requests
            response = requests.get(source, timeout=60)
            response.raise_for_status()
            payload = response.content
            content_type = response.headers.get("content-type", "").lower()
        except ImportError:
            request = Request(source, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=60) as response:
                payload = response.read()
                content_type = response.headers.get("content-type", "").lower()
        suffix = Path(urlparse(source).path).suffix.lower()
        if not suffix:
            suffix = ".pdf" if "pdf" in content_type else ".html"
        return payload, suffix
    path = Path(source)
    return path.read_bytes(), path.suffix.lower()


def _extract_pdf_text(payload: bytes) -> str:
    for package in ("pypdf", "PyPDF2"):
        try:
            module = __import__(package)
            reader_cls = module.PdfReader
            reader = reader_cls(io.BytesIO(payload))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            continue
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(payload)
        tmp.flush()
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", tmp.name, "-"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.stdout
        except Exception as exc:
            raise RuntimeError("Install pypdf/PyPDF2 or pdftotext to parse PDF reports.") from exc


def _extract_html_text(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="ignore")
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text("\n")
    except ImportError:
        text = re.sub(r"<(script|style).*?</\1>", " ", text, flags=re.I | re.S)
        return re.sub(r"<[^>]+>", "\n", text)


def extract_text(source: str) -> tuple[str, str]:
    payload, suffix = _fetch_bytes(source)
    if suffix.lower() == ".pdf":
        return _extract_pdf_text(payload), suffix
    return _extract_html_text(payload), suffix


def _normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _number_tokens(text: str) -> list[float]:
    out = []
    number_re = r"(?<![\w\d])-?(?:\d{1,3}(?: \d{3})+|\d+)(?:[,.]\d+)?(?![\w\d])"
    for raw in re.findall(number_re, text):
        value = raw.replace(" ", "").replace(",", ".")
        try:
            out.append(float(value))
        except ValueError:
            continue
    return out


def _month_from_text(text: str, source: str) -> pd.Timestamp | None:
    haystack = f"{Path(urlparse(source).path).name} {text[:4000]}".lower()
    slug_to_month = {slug: month for month, slug in SO_UPS_MONTH_SLUGS.items()}
    slug_to_month.update(SO_UPS_MONTH_SLUG_ALIASES)
    m = re.search(r"res_([a-z]+)_?(\d{2})(?:\D|$)", haystack)
    if m and m.group(1) in slug_to_month:
        return pd.Timestamp(year=2000 + int(m.group(2)), month=slug_to_month[m.group(1)], day=1)
    m = re.search(r"(20\d{2})[-_. ](0?[1-9]|1[0-2])", haystack)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    for stem, month_no in MONTHS_RU.items():
        pattern = rf"{stem}\w*\s+(20\d{{2}})"
        m = re.search(pattern, haystack, flags=re.I)
        if m:
            return pd.Timestamp(year=int(m.group(1)), month=month_no, day=1)
    return None


def _expected_months_from_inputs(inputs: list[str]) -> set[str]:
    years: set[int] = set()
    months: set[str] = set()
    for source in inputs:
        source_name = str(source)
        month = _month_from_text("", source_name)
        if month is not None:
            months.add(month.strftime("%Y-%m-01"))
            continue
        for match in re.finditer(r"/(20\d{2})/", source_name):
            years.add(int(match.group(1)))

    for year in years:
        for month_no in range(1, 13):
            months.add(pd.Timestamp(year=year, month=month_no, day=1).strftime("%Y-%m-01"))
    return months


def _window_around(lines: list[str], index: int, radius: int = 2) -> str:
    start = max(index - radius, 0)
    end = min(index + radius + 1, len(lines))
    return "\n".join(lines[start:end])


def _metric_from_lines(lines: list[str], region_pattern: str, keywords: tuple[str, ...]) -> float | None:
    region_re = re.compile(region_pattern, flags=re.I)
    keyword_res = [re.compile(keyword, flags=re.I) for keyword in keywords]
    candidates = []
    for idx, line in enumerate(lines):
        window = _window_around(lines, idx)
        if not region_re.search(window):
            continue
        if not any(pattern.search(window) for pattern in keyword_res):
            continue
        nums = _number_tokens(window)
        if nums:
            candidates.extend(nums)
    if not candidates:
        return None
    candidates = [value for value in candidates if abs(value) < 1_000_000_000]
    if not candidates:
        return None
    return float(candidates[-1])


def _fallback_region_numbers(lines: list[str], region_pattern: str) -> list[float]:
    region_re = re.compile(region_pattern, flags=re.I)
    values: list[float] = []
    for idx, line in enumerate(lines):
        if region_re.search(line):
            values.extend(_number_tokens(_window_around(lines, idx, radius=1)))
    return values


def _numeric_token_value(token: str) -> float:
    return float(token.replace(" ", "").replace(",", "."))


def _combine_integer_groups(groups: list[str]) -> float:
    return float("".join(groups))


def _table_numbers_from_region_line(line: str, region_re: re.Pattern) -> list[float]:
    tail = region_re.sub("", line, count=1).strip()
    tokens = re.findall(r"-?\d+(?:[,.]\d+)?", tail)
    if len(tokens) < 3:
        return []

    installed = _numeric_token_value(tokens[0])
    first_decimal_idx = None
    for idx in range(1, len(tokens)):
        if "," in tokens[idx] or "." in tokens[idx]:
            first_decimal_idx = idx
            break
    if first_decimal_idx is None or first_decimal_idx <= 2:
        return [_numeric_token_value(token) for token in tokens]

    generation_groups = tokens[1:first_decimal_idx]
    ytd = None
    month = None
    max_month_generation = max(installed * 24.0 * 31.0, 1.0)
    max_ytd_generation = max(installed * 24.0 * 366.0, max_month_generation)
    for split in range(1, len(generation_groups)):
        ytd_candidate = _combine_integer_groups(generation_groups[:split])
        month_candidate = _combine_integer_groups(generation_groups[split:])
        if month_candidate <= max_month_generation and month_candidate <= ytd_candidate <= max_ytd_generation:
            ytd = ytd_candidate
            month = month_candidate
            break
    if ytd is None or month is None:
        return [_numeric_token_value(token) for token in tokens]

    rest = [_numeric_token_value(token) for token in tokens[first_decimal_idx:]]
    return [installed, ytd, month] + rest


def _parse_region_table_line(lines: list[str], region_pattern: str) -> dict:
    region_re = re.compile(region_pattern, flags=re.I)
    best_numbers: list[float] = []
    for line in lines:
        if not region_re.search(line):
            continue
        nums = _table_numbers_from_region_line(line, region_re)
        if len(nums) > len(best_numbers):
            best_numbers = nums

    if len(best_numbers) < 3:
        return {}

    parsed = {
        "installed_mw": best_numbers[0],
        "generation_ytd_mwh": best_numbers[1],
        "generation_month_mwh": best_numbers[2],
    }
    # The official SO UPS table tail is usually:
    # SO command hours YTD/month, SO max curtailment YTD/month,
    # ARChM hours YTD/month, ARChM max curtailment YTD/month.
    if len(best_numbers) >= 11:
        parsed["curtailment_hours_month"] = best_numbers[-7]
        parsed["max_curtailment_mw_month"] = best_numbers[-5]
    return parsed


def parse_report(source: str, region_pattern: str, default_region: str, table_fallback: bool) -> tuple[dict, dict]:
    text, suffix = extract_text(source)
    text = _normalize_space(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    month = _month_from_text(text, source)
    row = {
        "month": month.strftime("%Y-%m-01") if month is not None else None,
        "region": default_region,
        "source": source,
        "source_format": suffix.lstrip("."),
    }
    row.update(_parse_region_table_line(lines, region_pattern))

    patterns = {
        "installed_mw": (r"установлен\w*\s+мощн", r"мощн\w*\s+объект"),
        "generation_month_mwh": (r"выработк", r"производств\w*\s+электроэнерг"),
        "generation_ytd_mwh": (r"с\s+начала\s+года", r"накоплен"),
        "curtailment_hours_month": (r"огранич\w*.*час", r"час\w*.*огранич"),
        "max_curtailment_mw_month": (r"максим\w*.*огранич", r"огранич\w*.*мвт"),
        "max_deviation_mw_month": (r"максим\w*.*отклон", r"отклон\w*.*мвт"),
    }
    for field, keywords in patterns.items():
        if row.get(field) is None:
            row[field] = _metric_from_lines(lines, region_pattern, keywords)

    if table_fallback and all(row.get(field) is None for field in FIELD_COLUMNS):
        values = _fallback_region_numbers(lines, region_pattern)
        if values:
            row["installed_mw"] = values[0] if len(values) > 0 else None
            row["generation_month_mwh"] = values[1] if len(values) > 1 else None
            row["generation_ytd_mwh"] = values[2] if len(values) > 2 else None

    diagnostics = {
        "source": source,
        "month": row["month"],
        "found_fields": [field for field in FIELD_COLUMNS if row.get(field) is not None],
        "region_line_samples": [line for line in lines if re.search(region_pattern, line, flags=re.I)][:8],
    }
    return row, diagnostics


def main() -> None:
    args = parse_args()
    inputs = _read_inputs(args)
    rows = []
    diagnostics = []
    for source in inputs:
        row, diag = parse_report(source, args.region_pattern, args.default_region, args.table_fallback)
        rows.append(row)
        diagnostics.append(diag)

    raw_frame = pd.DataFrame(rows)
    if "month" in raw_frame:
        unknown_mask = raw_frame["month"].isna()
    else:
        unknown_mask = pd.Series([True] * len(raw_frame), index=raw_frame.index)
    unknown_rows = raw_frame.loc[unknown_mask].copy()
    frame_input = raw_frame if args.keep_unknown_months else raw_frame.loc[~unknown_mask].copy()

    duplicate_rows = pd.DataFrame()
    if {"month", "region"}.issubset(frame_input.columns):
        duplicate_mask = frame_input.duplicated(["month", "region"], keep=False)
        duplicate_rows = frame_input.loc[duplicate_mask].sort_values(["month", "region", "source"])
        frame = (
            frame_input.sort_values(["month", "source"], na_position="last")
            .drop_duplicates(["month", "region"], keep="last")
            .reset_index(drop=True)
        )
    else:
        frame = frame_input.reset_index(drop=True)

    expected_months = _expected_months_from_inputs(inputs)
    parsed_months = set(frame["month"].dropna().astype(str)) if "month" in frame else set()
    missing_expected_months = sorted(expected_months - parsed_months)
    extra_months = sorted(parsed_months - expected_months) if expected_months else []
    duplicate_replacements = 0
    if not duplicate_rows.empty:
        unique_duplicate_keys = duplicate_rows[["month", "region"]].drop_duplicates().shape[0]
        duplicate_replacements = int(max(len(duplicate_rows) - unique_duplicate_keys, 0))
    summary = {
        "input_reports": len(inputs),
        "parsed_rows_raw": len(raw_frame),
        "written_rows": len(frame),
        "unknown_month_rows": int(len(unknown_rows)),
        "unknown_month_sources": unknown_rows["source"].dropna().astype(str).tolist() if "source" in unknown_rows else [],
        "duplicate_month_region_rows": int(len(duplicate_rows)),
        "duplicate_replacements": duplicate_replacements,
        "expected_month_count": len(expected_months),
        "parsed_month_count": len(parsed_months),
        "missing_expected_months": missing_expected_months,
        "extra_months": extra_months,
        "kept_unknown_months": bool(args.keep_unknown_months),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_path, index=False)
    if args.dump_json:
        args.dump_json.write_text(
            json.dumps({"summary": summary, "reports": diagnostics}, indent=2, ensure_ascii=False)
        )
    print(f"Wrote {args.output_path} rows={len(frame)}")
    print(
        "SO UPS parser summary: "
        f"inputs={summary['input_reports']} written={summary['written_rows']} "
        f"unknown_month_rows={summary['unknown_month_rows']} "
        f"duplicate_replacements={summary['duplicate_replacements']} "
        f"parsed_months={summary['parsed_month_count']}/{summary['expected_month_count']}"
    )
    if summary["unknown_month_rows"]:
        action = "kept" if args.keep_unknown_months else "dropped"
        print(
            f"WARNING: {action} {summary['unknown_month_rows']} rows with unknown month:",
            file=sys.stderr,
        )
        for source in summary["unknown_month_sources"]:
            print(f"  - {source}", file=sys.stderr)
    if not duplicate_rows.empty:
        print(
            f"WARNING: replaced {summary['duplicate_replacements']} duplicate month/region rows; "
            "kept the lexicographically latest source per month.",
            file=sys.stderr,
        )
    if missing_expected_months:
        print(
            "WARNING: missing expected SO UPS months: "
            + ", ".join(missing_expected_months),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
