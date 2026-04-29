import re
import warnings
import numpy as np
import pandas as pd
from dateutil import parser as du_parser
from dateutil.parser import ParserError

warnings.filterwarnings("ignore", category=UserWarning)

INPUT_XLSX = "Raw Datasets/DOE_Electric_Disturbance_Events.xlsx"
OUTPUT_CSV = "src/doe_events_db_ready.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MONTH_NAMES = {
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
}

# Maps cleaned slug names → canonical column names used in the output table
COL_MAP = {
    # ── Format A (2002–2010) ──────────────────────────────────────────────
    "date":                             "date_event_began",
    "time":                             "time_event_began",
    "area":                             "area_affected",
    "area_affected":                    "area_affected",
    "type_of_disturbance":              "event_type",
    "loss_megawatts":                   "demand_loss_mw",
    "restoration_time":                 "restoration_raw",
    "restoration":                      "restoration_raw",
    "restoration_date_time":            "restoration_raw",
    # ── Format B (2011–2014) / C (2015–2022) / D (2023) ──────────────────
    "date_event_began":                 "date_event_began",
    "time_event_began":                 "time_event_began",
    "date_of_restoration":              "date_of_restoration",
    "time_of_restoration":              "time_of_restoration",
    "demand_loss_mw":                   "demand_loss_mw",
    "demand_loss_megawatts":            "demand_loss_mw",
    "number_of_customers_affected":     "customers_affected",
    "number_of_customers_affected_1":   "customers_affected",
    "number_of_customers_affected_1_1": "customers_affected",  
    "nerc_region":                      "nerc_region",
    "event_type":                       "event_type",
    "alert_criteria":                   "alert_criteria",
    # ── Format C/D month/year cols — ignored (dates from date_event_began) ─
    "month":                            "_ignore",
    "event_year":                       "_ignore",
    "event_month":                      "_ignore",
}

# ---------------------------------------------------------------------------
# String / row helpers
# ---------------------------------------------------------------------------

def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _nonnull_vals(row_series) -> list:
    return [v for v in row_series
            if pd.notna(v) and str(v).strip() not in ("", "nan")]


def _is_month_label_row(row_series) -> bool:
    vals = _nonnull_vals(row_series)
    if not vals:
        return False
    first = str(vals[0]).strip().lower().rstrip("1234567890 ")
    return first in MONTH_NAMES and len(vals) <= 2


def _is_header_row(row_vals) -> bool:
    """
    True when the row looks like a column-header row.
    """
    import datetime as _dt
    vals = [v for v in row_vals if pd.notna(v)]
    if not vals:
        return False
    # If the first value is an actual date/datetime, this is a data row
    first = vals[0]
    if isinstance(first, (_dt.datetime, _dt.date, pd.Timestamp)):
        return False

    if any(len(str(v)) > 60 for v in vals):
        return False
    text = " ".join(str(v).lower() for v in vals)
    kws = ["date","time","area","nerc","region","restoration",
           "demand","customers","event","disturbance","loss","alert"]
    hits  = sum(k in text for k in kws)
    non_null = sum(1 for v in vals if str(v).strip() not in ("", "nan"))
    return hits >= 2 and non_null >= 4


def _is_filler_row(row_series) -> bool:
    """True when a row is blank, month-label, footnote, or page-break artifact."""
    vals = _nonnull_vals(row_series)
    if not vals:
        return True
    first_str = str(vals[0]).strip().lower()

    if _is_month_label_row(row_series):
        return True

    # Footnote / annotation rows
    if first_str.startswith(("[","note","source","1 source","report","data for","(continued")):
        return True

    # Bare small integer footnote marker (e.g. "1", "2")
    # Must NOT match 4-digit years (2023) or large numbers
    if re.fullmatch(r"\d+", first_str) and int(first_str) <= 20:
        return True

    if _is_header_row(vals):
        return True


    _DATE_PAT = re.compile(
        r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{2}[/\-]\d{2})\b'
    )
    if len(vals) <= 2:
        has_date = any(
            isinstance(v, pd.Timestamp)
            or (isinstance(v, str) and _DATE_PAT.search(v))
            for v in vals
        )
        if not has_date:
            return True
    return False

# ---------------------------------------------------------------------------
# Header / data block extraction
# ---------------------------------------------------------------------------

def extract_header_and_data(df_raw):
    """
    Locate the header row, handle split sub-headers (e.g. "Restoration / Date/Time"),
    and return (header_list, data_df) without merging month-label rows.
    """
    header_idx = None
    for i in range(min(10, len(df_raw))):
        if _is_header_row(df_raw.iloc[i].tolist()):
            header_idx = i
            break
    if header_idx is None:
        header_idx = int(df_raw.iloc[:10].notna().sum(axis=1).idxmax())

    header = list(df_raw.iloc[header_idx])
    next_idx   = header_idx + 1
    data_start = next_idx

    if next_idx < len(df_raw):
        next_row  = df_raw.iloc[next_idx]
        next_vals = _nonnull_vals(next_row)
        is_month  = _is_month_label_row(next_row)


        is_sub_header = (
            len(next_vals) <= 3
            and not is_month
            and any(any(kw in str(v).lower() for kw in ["date","time","month"])
                    for v in next_vals)
        )
        if is_sub_header:
            for col_pos, val in enumerate(next_row):
                if pd.notna(val) and str(val).strip():
                    base = str(header[col_pos]) if pd.notna(header[col_pos]) else ""
                    header[col_pos] = (base + " " + str(val).strip()).strip()
            data_start = next_idx + 1

    data_df = df_raw.iloc[data_start:].copy()
    data_df.columns = range(len(data_df.columns))
    return header, data_df


def normalise_cols(df, header):
    clean = [_slug(str(h)) if pd.notna(h) else f"_col{i}"
             for i, h in enumerate(header)]
    df = df.copy()
    df.columns = clean
    df.rename(columns={c: COL_MAP.get(c, c) for c in df.columns}, inplace=True)
    # Deduplicate column names (e.g. two '_ignore' cols in the 2023 Format D sheet)
    seen: dict = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df

# ---------------------------------------------------------------------------
# Numeric field parsing
# ---------------------------------------------------------------------------

def _parse_numeric_field(series: pd.Series) -> pd.Series:

    def _conv(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s.lower() in ("unknown","n/a","na","-","--","","nan","none","tbd","unk"):
            return np.nan
        m = re.fullmatch(r"(\d+\.?\d*)\s*[-]\s*(\d+\.?\d*)", s)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2.0
        m2 = re.search(r"(\d+\.?\d*)", s.replace(",", ""))
        return float(m2.group(1)) if m2 else np.nan
    return series.apply(_conv)

# ---------------------------------------------------------------------------
# Time / datetime helpers
# ---------------------------------------------------------------------------

def _parse_time_val(val) -> pd.Timestamp:

    if pd.isna(val):
        return pd.NaT
    if isinstance(val, pd.Timestamp):
        return val
    if hasattr(val, "hour"):      # datetime.time object
        return pd.Timestamp(f"1900-01-01 {val.hour:02d}:{val.minute:02d}:{getattr(val,'second',0):02d}")
    s = str(val).strip()
    if s.lower() in ("","nan","unknown","tbd","none"):
        return pd.NaT
    try:
        return pd.to_datetime(s, errors="raise")
    except Exception:
        pass
    try:
        t = du_parser.parse(s, default=pd.Timestamp("1900-01-01").to_pydatetime())
        return pd.Timestamp(t)
    except (ParserError, OverflowError, ValueError):
        return pd.NaT


def _build_ts(date_val, time_val) -> pd.Timestamp:
    """Combine date + time, single Timestamp (defaults to midnight if time is missing)."""
    d = pd.to_datetime(date_val, errors="coerce")
    if pd.isna(d):
        return pd.NaT
    d_norm = d.normalize()
    t = _parse_time_val(time_val)
    if pd.isna(t):
        return d_norm
    return d_norm + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)


def _parse_restoration_freetext(val, year_hint: int) -> pd.Timestamp:

    if pd.isna(val):
        return pd.NaT
    if isinstance(val, pd.Timestamp):
        return val
    if hasattr(val, "year"):
        return pd.Timestamp(val)
    s = str(val).strip()
    if s.lower() in ("","nan","unknown","tbd","date/time","none"):
        return pd.NaT
    default_dt = pd.Timestamp(f"{year_hint}-01-01").to_pydatetime()
    try:
        t = du_parser.parse(s, default=default_dt)
        return pd.Timestamp(t)
    except (ParserError, OverflowError, ValueError):
        return pd.NaT


def _fix_end_ts(start_ts, end_ts, year_hint: int) -> pd.Timestamp:
    """
    Repair end timestamps:
    - If end_ts has a year that is implausibly far from the start_ts year (e.g. 2018 in a 2019 sheet), try rebuilding end_ts using start_ts's date + end_ts's time

    """
    if pd.isna(end_ts) or pd.isna(start_ts):
        return end_ts
    # Use <= so that off-by-one year typos (e.g. 2018 in a 2019 sheet) are caught
    if end_ts.year <= (year_hint - 1) or end_ts.year > (year_hint + 2):
        try:
            rebuilt = start_ts.normalize() + pd.Timedelta(
                hours=end_ts.hour, minutes=end_ts.minute, seconds=end_ts.second)
            if rebuilt.year < (year_hint - 1) or rebuilt.year > (year_hint + 2):
                return pd.NaT
            end_ts = rebuilt
        except Exception:
            return pd.NaT
    if end_ts < start_ts:
        delta = start_ts - end_ts
        if delta.days <= 1:
            # Overnight crossover (e.g. 23:30 start, 00:30 end next day)
            end_ts = end_ts + pd.Timedelta(days=1)
        else:
            end_ts = end_ts + pd.DateOffset(years=1)
    return end_ts

# ---------------------------------------------------------------------------
# Per-sheet processor
# ---------------------------------------------------------------------------

def process_sheet(sheet_name: str, workbook_path: str) -> pd.DataFrame:
    year   = int(sheet_name)
    df_raw = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)

    header, df = extract_header_and_data(df_raw)
    df = normalise_cols(df, header)

    # Drop blank rows, month labels, footnotes, AND mid-sheet repeated header rows
    def _should_drop(row):
        return _is_filler_row(row) or _is_header_row(row.tolist())

    keep = ~df.apply(_should_drop, axis=1)
    df   = df[keep].reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame()

    if "date_event_began" in df.columns:
        df["event_start_ts"] = df.apply(
            lambda r: _build_ts(r["date_event_began"],
                                r.get("time_event_began", np.nan)), axis=1)
    else:
        df["event_start_ts"] = pd.NaT


    if "date_of_restoration" in df.columns and "time_of_restoration" in df.columns:
        df["event_end_ts"] = df.apply(
            lambda r: _build_ts(r["date_of_restoration"],
                                r["time_of_restoration"]), axis=1)
    elif "restoration_raw" in df.columns:
        df["event_end_ts"] = df["restoration_raw"].apply(
            lambda v: _parse_restoration_freetext(v, year))
    else:
        df["event_end_ts"] = pd.NaT

    df["event_end_ts"] = df.apply(
        lambda r: _fix_end_ts(r["event_start_ts"], r["event_end_ts"], year), axis=1)

    # Numeric fields
    for col in ("customers_affected", "demand_loss_mw"):
        src = df[col] if col in df.columns else pd.Series([np.nan] * len(df))
        df[col] = _parse_numeric_field(src)


    if "demand_loss_mw" in df.columns:
        implausible = df["demand_loss_mw"] > 100_000
        if implausible.any():
            print(f"    WARNING {year}: {implausible.sum()} implausible demand_loss_mw "
                  f"value(s) set to NaN: "
                  f"{df.loc[implausible, 'demand_loss_mw'].tolist()}")
            df.loc[implausible, "demand_loss_mw"] = np.nan

    #Text fields
    def _clean_text(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index)
        s = df[col].astype(str).str.strip()
        return s.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})

    # Normalise NERC region — fix known typos from source data
    _NERC_FIXES = {
        "NPPC":              "NPCC",          # typo: C vs P
        "MR0":               "MRO",           # typo: digit 0 vs letter O
        "WEENERGIESMAIN":    "MAIN",          # utility name crept into region field
        "MIDWEST ISO (RFC":  "RFC",           # truncated label
    }
    df["_nerc"] = (
        _clean_text("nerc_region")
        .str.upper()
        .str.strip()
        .replace(_NERC_FIXES)
    )
    df["_event_type"] = _clean_text("event_type")
    df["_alert"]      = _clean_text("alert_criteria")
    df["_area"]       = _clean_text("area_affected")
    df["year_sheet"]  = year
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    xls    = pd.ExcelFile(INPUT_XLSX)
    frames = []

    for sheet in xls.sheet_names:
        try:
            df = process_sheet(sheet, INPUT_XLSX)
            if len(df):
                frames.append(df)
                print(f"  {sheet}: {len(df):>4} rows")
            else:
                print(f"  {sheet}:    0 rows (no data found)")
        except Exception as exc:
            print(f"  {sheet}: ERROR — {exc}")
            raise

    df_all = pd.concat(frames, ignore_index=True)

    # Duration
    start = pd.to_datetime(df_all["event_start_ts"], errors="coerce")
    end   = pd.to_datetime(df_all["event_end_ts"],   errors="coerce")
    dur   = (end - start).dt.total_seconds() / 3600.0
    dur   = dur.where(dur.isna() | (dur >= 0),    np.nan)   # negative = data error
    dur   = dur.where(dur.isna() | (dur <= 8760), np.nan)   # > 1 year = implausible

    # Assemble output table
    out = pd.DataFrame({
        "event_id":              range(1, len(df_all) + 1),
        "year_sheet":            df_all["year_sheet"],
        "event_start_ts":        start,
        "event_end_ts":          end,
        "outage_duration_hours": dur,
        "customers_affected":    df_all["customers_affected"],
        "demand_loss_mw":        df_all["demand_loss_mw"],
        "nerc_region":           df_all["_nerc"],
        "event_type":            df_all["_event_type"],
        "alert_criteria":        df_all["_alert"],
        "area_affected_raw":     df_all["_area"],
    })

    out["event_year"]             = out["event_start_ts"].dt.year
    out["event_month"]            = out["event_start_ts"].dt.month
    out["has_end_ts"]             = out["event_end_ts"].notna()
    out["has_customers_affected"] = out["customers_affected"].notna()

    out.to_csv(OUTPUT_CSV, index=False)

    # Quality report
    print("\n" + "=" * 55)
    print(f"Output  : {OUTPUT_CSV}")
    print(f"Rows    : {len(out):,}   Cols: {len(out.columns)}")
    print("-" * 55)
    print("Missing values per column:")
    for col in out.columns:
        n   = int(out[col].isna().sum())
        pct = 100 * n / len(out)
        print(f"  {col:<30} {n:>5}  ({pct:.1f}%)")
    print("-" * 55)
    print("Rows per year sheet:")
    print(out["year_sheet"].value_counts().sort_index().to_string())
    print("=" * 55)


if __name__ == "__main__":
    main()
