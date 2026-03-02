import re
import numpy as np
import pandas as pd

INPUT_XLSX = "DOE_Electric_Disturbance_Events.xlsx"
OUTPUT_CSV = "doe_events_db_ready.csv"

# drops whitespace-only / mostly blank rows
MIN_NON_NULL_PER_ROW = 4


# -----------------------------
# CLEANING HELPERS
# -----------------------------
def clean_colname(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def standardize_missing(df):
    df = df.copy()
    missing_like = ["unknown", "n/a", "na", "", " ", "  ", "-", "--", "null", "none", "tbd"]
    df.replace(missing_like, np.nan, inplace=True)
    return df


def strip_whitespace_strings(df):
    """
    Converts cells like "   " into NaN so blank rows can be dropped.
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        s = df[c].astype(str).str.strip()
        df[c] = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})
    return df


def drop_blankish_rows(df):
    df = df.copy()
    df = df.dropna(how="all")
    df = df.dropna(thresh=MIN_NON_NULL_PER_ROW)
    return df


def find_header_row(df_raw):
    """
    DOE sheets often have a title row above the header.
    We pick the row that looks like a header based on keywords + non-null count.
    """
    keywords = ["date", "time", "area", "region", "nerc", "restoration", "demand", "customers", "event"]
    best_idx = 0
    best_score = -1

    for i in range(min(40, len(df_raw))):
        row = df_raw.iloc[i].astype(str).str.lower()
        non_null = df_raw.iloc[i].notna().sum()
        hit = sum(row.str.contains(k, na=False).any() for k in keywords)
        score = hit * 10 + non_null
        if score > best_score and non_null >= 5:
            best_score = score
            best_idx = i

    return best_idx


def normalize_columns(df):
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]

    rename_map = {
        "area": "area_affected",
        "type_of_disturbance": "event_type",
        "demand_loss_megawatts": "demand_loss_mw",
        "loss_megawatts": "demand_loss_mw",
        "number_of_customers_affected": "customers_affected",
        "restoration_time": "restoration_time",  # keep name; we'll handle it in end_ts builder
    }

    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)
    return df


def parse_date_series(date_series):
    """
    Parse dates robustly. Works for strings and Excel datetimes.
    """
    return pd.to_datetime(date_series, errors="coerce")


def parse_time_series(time_series):
    """
    Parse times robustly.
    - If it's already datetime-like, keep time-of-day.
    - If it's strings like 06:00:00, parse.
    - If it's words like 'Evening', it becomes NaT (we'll default later).
    """
    # Try parsing as full datetime first (in case the column contains datetimes)
    dt = pd.to_datetime(time_series, errors="coerce")
    # If parsing worked but date component is present, we'll still use its time component.
    # If parsing fails, dt is NaT.
    return dt


def build_start_ts(df):
    """
    Builds event_start_ts for both old and new sheet formats.
    Key fix: if time is unparseable (e.g., 'Evening'), keep the date and default time to 00:00.
    """
    # Newer format
    if "date_event_began" in df.columns and "time_event_began" in df.columns:
        d = parse_date_series(df["date_event_began"])
        t = parse_time_series(df["time_event_began"])
    # Older format
    elif "date" in df.columns:
        d = parse_date_series(df["date"])
        t = parse_time_series(df["time"]) if "time" in df.columns else pd.Series([pd.NaT] * len(df), index=df.index)
    else:
        return pd.Series([pd.NaT] * len(df), index=df.index)

    # Default time to midnight when missing/unparseable
    t = t.fillna(pd.Timestamp("1900-01-01 00:00:00"))
    start_ts = d.dt.normalize() + pd.to_timedelta(t.dt.hour, unit="h") + pd.to_timedelta(t.dt.minute, unit="m") + pd.to_timedelta(t.dt.second, unit="s")

    return start_ts


def build_end_ts(df):
    """
    Builds event_end_ts across multiple DOE formats:
    - Newer: date_of_restoration + time_of_restoration
    - Older: restoration_time (already a datetime)
    - If only time exists or time is weird -> combine with start date later via fix_end_ts
    """
    # If older sheets have a full datetime
    if "restoration_time" in df.columns:
        end_ts = pd.to_datetime(df["restoration_time"], errors="coerce")
        if end_ts.notna().any():
            return end_ts

    # Newer format date + time
    if "date_of_restoration" in df.columns and "time_of_restoration" in df.columns:
        d = parse_date_series(df["date_of_restoration"])
        t = parse_time_series(df["time_of_restoration"])
        t = t.fillna(pd.Timestamp("1900-01-01 00:00:00"))
        end_ts = d.dt.normalize() + pd.to_timedelta(t.dt.hour, unit="h") + pd.to_timedelta(t.dt.minute, unit="m") + pd.to_timedelta(t.dt.second, unit="s")
        return end_ts

    # If only time_of_restoration exists (some weird years)
    if "time_of_restoration" in df.columns:
        return pd.to_datetime(df["time_of_restoration"], errors="coerce")

    return pd.Series([pd.NaT] * len(df), index=df.index)


def fix_end_ts(start_ts, end_ts, year_sheet):
    """
    Repair bad end timestamps:
      - Weird year (1899/1900 or far future) => treat as time-only, rebuild using start date + end time-of-day
      - End earlier than start => add 1 day (midnight crossover)
      - Still implausible => set NaT
    """
    end_fixed = end_ts.copy()

    # weird year detection
    weird_year = (
        end_fixed.notna() &
        (
            (end_fixed.dt.year < 1990) |
            (end_fixed.dt.year > (year_sheet + 2))
        )
    )

    # salvage: use start date + end time-of-day
    salvage = weird_year & start_ts.notna()
    start_date = start_ts.dt.normalize()
    end_time_str = end_fixed.dt.strftime("%H:%M:%S")

    rebuilt = pd.to_datetime(
        start_date.dt.strftime("%Y-%m-%d") + " " + end_time_str,
        errors="coerce"
    )
    end_fixed.loc[salvage] = rebuilt.loc[salvage]

    crossed = start_ts.notna() & end_fixed.notna() & (end_fixed < start_ts)
    end_fixed.loc[crossed] = end_fixed.loc[crossed] + pd.Timedelta(days=1)

    weird_after = (
        end_fixed.notna() &
        (
            (end_fixed.dt.year < 1990) |
            (end_fixed.dt.year > (year_sheet + 2))
        )
    )
    end_fixed.loc[weird_after] = pd.NaT

    return end_fixed


def main():
    xls = pd.ExcelFile(INPUT_XLSX)
    all_rows = []

    for sheet in xls.sheet_names:
        df_raw = pd.read_excel(INPUT_XLSX, sheet_name=sheet, header=None)

        header_idx = find_header_row(df_raw)
        header = df_raw.iloc[header_idx].tolist()

        df = df_raw.iloc[header_idx + 1:].copy()
        df.columns = header

        df = standardize_missing(df)
        df = strip_whitespace_strings(df)
        df = drop_blankish_rows(df)
        df = normalize_columns(df)

        # add year_sheet
        try:
            df["year_sheet"] = int(sheet)
        except:
            df["year_sheet"] = np.nan

        # build timestamps robustly
        df["event_start_ts"] = build_start_ts(df)
        df["event_end_ts"] = build_end_ts(df)

        # numeric cleanup
        if "customers_affected" in df.columns:
            df["customers_affected"] = pd.to_numeric(df["customers_affected"], errors="coerce")

        if "demand_loss_mw" in df.columns:
            mw = df["demand_loss_mw"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
            df["demand_loss_mw"] = pd.to_numeric(mw, errors="coerce")

        # keep only the columns we need in the final DB-ready table
        all_rows.append(df)

    df_all = pd.concat(all_rows, ignore_index=True)

    # Final end date repair (critical)
    start_ts = pd.to_datetime(df_all["event_start_ts"], errors="coerce")
    end_ts = pd.to_datetime(df_all["event_end_ts"], errors="coerce")
    year_sheet = df_all["year_sheet"]

    end_ts_fixed = fix_end_ts(start_ts, end_ts, year_sheet)

    duration = (end_ts_fixed - start_ts).dt.total_seconds() / 3600.0
    duration = duration.where(duration.isna() | (duration >= 0), np.nan)

    out = pd.DataFrame({
        "event_id": range(1, len(df_all) + 1),
        "year_sheet": year_sheet,
        "event_start_ts": start_ts,
        "event_end_ts": end_ts_fixed,
        "outage_duration_hours": duration,
        "customers_affected": df_all.get("customers_affected"),
        "demand_loss_mw": df_all.get("demand_loss_mw"),
        "nerc_region": df_all.get("nerc_region"),
        "event_type": df_all.get("event_type"),
        "alert_criteria": df_all.get("alert_criteria"),
        "area_affected_raw": df_all.get("area_affected"),
    })

    out["event_year"] = out["event_start_ts"].dt.year
    out["event_month"] = out["event_start_ts"].dt.month

    # flags 
    out["has_end_ts"] = out["event_end_ts"].notna()
    out["has_customers_affected"] = out["customers_affected"].notna()

    out.to_csv(OUTPUT_CSV, index=False)

    #quality printout
    print("Saved ->", OUTPUT_CSV)
    print("Rows:", out.shape[0], "Cols:", out.shape[1])
    print("Missing start_ts:", int(out["event_start_ts"].isna().sum()))
    print("Missing end_ts  :", int(out["event_end_ts"].isna().sum()))
    print("Missing customers_affected:", int(out["customers_affected"].isna().sum()))


if __name__ == "__main__":
    main()
