
import re
import math
import pdfplumber
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, date
import streamlit as st 
import numpy as np 
from io import BytesIO


# -------------------------
# Regex & utilities
# -------------------------
# --- regex helpers (kept local to this file) ---
DATE_RX = re.compile(r"\b\d{2}-[A-Z]{3}-\d{2}\b")
NUM_RX  = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
VOUCHER_DATE_RX = re.compile(r"voucher\s*date\s*:?\s*(\d{2}-[A-Z]{3}-\d{2})", re.IGNORECASE)
OPENING_BAL_RX = re.compile(
    r"opening\s+balance.*?\b(?:as\s+on\s+(\d{2}-[A-Z]{3}-\d{2}))?.*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(DR|CR)\b",
    re.IGNORECASE
)

def _to_float(txt):
    if txt is None: return None
    m = re.findall(NUM_RX, str(txt))
    if not m: return None
    try: return float(m[0].replace(",", ""))
    except: return None

def _parse_date(tok):
    if not tok: return None
    try:
        d = datetime.strptime(tok, "%d-%b-%y")
        if d.year < 1970: d = d.replace(year=d.year + 100)
        return d.date()
    except: return None

def _group_lines(words, y_tol=2):
    byy = defaultdict(list)
    for w in words:
        y = round(w["top"]/y_tol)*y_tol
        byy[y].append(w)
    out = []
    for y in sorted(byy):
        out.append((y, sorted(byy[y], key=lambda x: x["x0"])))
    return out

def _find_column_bands_strict(page):
    """Find non-overlapping x-bands for Debit and Credit from the header row."""
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    header_top = None
    for w in words:
        t = w["text"].strip().lower()
        if t in {"debit", "credit", "running", "balance", "running balance"}:
            if header_top is None or w["top"] < header_top:
                header_top = w["top"]
    if header_top is None: return None

    debit_x = credit_x = rb_x = None
    for w in words:
        if abs(w["top"] - header_top) > 6: continue
        t = w["text"].strip().lower()
        if t == "debit":   debit_x  = w["x0"]
        elif t == "credit": credit_x = w["x0"]
        elif ("running" in t and "balance" in t) or t in {"running", "balance"}:
            rb_x = w["x0"]

    if debit_x is None or credit_x is None: return None

    page_right = page.width - 5
    mid_dc = (debit_x + credit_x) / 2
    right_anchor = rb_x if rb_x is not None else page_right
    mid_cr = (credit_x + right_anchor) / 2

    debit_band  = (max(0, debit_x - 95), max(mid_dc - 8, debit_x + 12))
    credit_band = (min(mid_dc + 8, credit_x - 12), min(mid_cr - 8, page_right))

    if debit_band[0] >= debit_band[1] or credit_band[0] >= credit_band[1]:
        return None
    return {"debit": debit_band, "credit": credit_band}

def _collect_numbers_in_band(words, band):
    x0, x1 = band
    vals = []
    for w in words:
        if x0 <= w["x0"] and w["x1"] <= x1:
            try:
                val = _to_float(w["text"])
                # skip tiny values that are likely noise
                if val > 100:  
                    vals.append(val)
            except:
                pass
    return vals

def extract_account_name(page):
    """Extract only the customer name from the Account Name line."""
    text = page.extract_text()
    if not text:
        return None

    match = re.search(r"Account\s*Name\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        line = match.group(1).strip()

        # stop at keywords that are not part of the customer name
        stop_keywords = ["Opening Balance", "Period", "Voucher", "Date"]
        for kw in stop_keywords:
            if kw in line:
                line = line.split(kw, 1)[0].strip()

        # remove trailing (Dr)/(Cr)
        line = re.sub(r"\(D[RC]\)$", "", line).strip()
        return line
    return None


# Extracting data from the pdf. I've used blocking by voucher date 
def extract_ledger_by_blocks_strict(pdf_path: str) -> pd.DataFrame:
    """
    Top→bottom, across pages:
      - 'Voucher Date : DD-MMM-YY' starts a BLOCK; store that voucher date.
      - DEBIT amount(s) on a row → each gets date = current voucher date (block date).
        (If no voucher date seen yet, skip the debit to avoid a wrong date.)
      - CREDIT amount(s) on a row → date = row date if present, else voucher date.
      - Totals/closing lines ("Total/Closing Balance", "Grand Total") are *never* included.
    Output columns: Invoice Date | Debit | Credit
    """
    records = []
    current_voucher_date: date | None = None
    opening_row_added = False
    last_bands = None
    customer_name = None 

    with pdfplumber.open(pdf_path) as pdf:
        for  page in pdf.pages:
            words = page.extract_words(
                x_tolerance=2, y_tolerance=2,
                keep_blank_chars=False, use_text_flow=True
            )
            if not words:
                continue

            if customer_name is None: 
                customer_name = extract_account_name(page)

            bands = _find_column_bands_strict(page) or last_bands
            if bands: 
                last_bands = bands
            if not bands:
                continue

            lines = _group_lines(words, y_tol=2)
            texts = {i: " ".join(w["text"] for w in lw).strip() for i, (_, lw) in enumerate(lines)}

            for i, (y, lw) in enumerate(lines):
                text = texts[i]
                if not text:
                    continue


                # Opening Balance
                if not opening_row_added:
                    m_ob = OPENING_BAL_RX.search(text)
                    if m_ob:
                        as_on = _parse_date(m_ob.group(1)) or (
                            _parse_date(DATE_RX.search(text).group(0)) 
                            if DATE_RX.search(text) else None
                        )
                        amt   = _to_float(m_ob.group(2))
                        side  = (m_ob.group(3) or "").upper()
                        if as_on and amt:
                            records.append({
                                "Invoice Date": as_on,
                                "Debit": amt if side == "DR" else 0.0,
                                "Credit": amt if side == "CR" else 0.0
                            })
                            opening_row_added = True
                        continue

                # Block boundary
                vm = VOUCHER_DATE_RX.search(text)
                if vm:
                    vd = _parse_date(vm.group(1))
                    if vd:
                        current_voucher_date = vd
                    continue

                # Row date (for credits)
                row_date = None
                dm = DATE_RX.search(text)
                if dm:
                    row_date = _parse_date(dm.group(0))

                # Collect debit/credit numbers
                debit_vals  = _collect_numbers_in_band(lw, bands["debit"])
                credit_vals = _collect_numbers_in_band(lw, bands["credit"])

                # Look ahead for wrapped values
                if not debit_vals and i + 1 < len(lines) and abs(lines[i+1][0] - y) <= 6:
                    debit_vals = _collect_numbers_in_band(lines[i+1][1], bands["debit"])
                if not credit_vals and i + 1 < len(lines) and abs(lines[i+1][0] - y) <= 6:
                    credit_vals = _collect_numbers_in_band(lines[i+1][1], bands["credit"])

                # Emit debits
                if debit_vals and current_voucher_date is not None:
                    for dv in debit_vals:
                        if dv and dv != 0.0:
                            records.append({
                                "Invoice Date": current_voucher_date,
                                "Debit": float(dv),
                                "Credit": 0.0
                            })

                # Emit credits
                credit_date = row_date if row_date is not None else current_voucher_date
                if credit_vals and credit_date is not None:
                    for cv in credit_vals:
                        if cv and cv != 0.0:
                            records.append({
                                "Invoice Date": credit_date,
                                "Debit": 0.0,
                                "Credit": float(cv)
                            })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.drop_duplicates().reset_index(drop=True)
    df["Debit"]  = pd.to_numeric(df["Debit"], errors="coerce").fillna(0.0)
    df["Credit"] = pd.to_numeric(df["Credit"], errors="coerce").fillna(0.0)

    df.drop(df.tail(2).index, inplace=True)  # dropping last 2 rows (usually totals)
    return df, customer_name


# ------- AGING ------------ #
#---------------------#

def aging_analysis(
    df: pd.DataFrame,
    customer_name: str = "Customer",
    as_of: date | None = None,
    *,
    day30_in_first_bucket: bool = True,
    return_breakdown: bool = False,
    debug: bool = False
):
    """
    DataFrame has ['Invoice Date','Debit','Credit'].
    Row 1 is Opening Balance (Debit if DR, Credit if CR).
    Outstanding = ΣDebit − ΣCredit.

    - Cleans malformed tokens like '21-AUG-251' -> '21-AUG-2025'
    - Forces day-first parsing so 05-09-2025 = 5-Sep-2025 
    - FIFO allocation, splitting partial debits
    - Excess credits carried forward as 'Advance Credit'
    - By default ages on today's date 
    """

    as_of = as_of or date.today()

    # --- Safe date parser ---
    def _fix_date_token(val):
        if pd.isna(val):
            return pd.NaT
        # if it's already a datetime, leave it alone
        if isinstance(val, (pd.Timestamp, date)):
            return pd.to_datetime(val)
        s = str(val).strip().upper()

        # Fix malformed like "21-AUG-251"
        m = re.match(r"(\d{2}-[A-Z]{3}-)(\d{2})(\d*)", s)
        if m:
            yy = m.group(2)  # '25'
            fixed = f"{m.group(1)}20{yy}"  # -> 21-AUG-2025
            return pd.to_datetime(fixed, format="%d-%b-%Y", errors="coerce")

        # Fallback – always force day-first
        return pd.to_datetime(s, dayfirst=True, errors="coerce")


    # --- Prep dataframe
    d = df.copy()
    d["Invoice Date"] = d["Invoice Date"].apply(_fix_date_token)
    d["Debit"]  = pd.to_numeric(d["Debit"],  errors="coerce").fillna(0.0)
    d["Credit"] = pd.to_numeric(d["Credit"], errors="coerce").fillna(0.0)

    # --- Outstanding
    outstanding = round(float(d["Debit"].sum() - d["Credit"].sum()), 2)

    # --- FIFO allocation
    q = deque() 
    for idx, r in d.iterrows():
        inv_dt = r["Invoice Date"] if pd.notna(r["Invoice Date"]) else pd.to_datetime(as_of)
        db, cr = float(r["Debit"]), float(r["Credit"])

        if db > 0:
            if debug:
                print(f"[{idx}] ADD debit {db:,.2f} dated {inv_dt.date() if pd.notna(inv_dt) else inv_dt}")
            q.append([inv_dt, db])

        if cr > 0:
            rem = cr
            if debug:
                print(f"[{idx}] APPLY credit {cr:,.2f} dated {inv_dt.date() if pd.notna(inv_dt) else inv_dt}")
            while rem > 1e-9 and q:
                dt0, amt0 = q.popleft()
                if amt0 <= rem + 1e-9:
                    if debug:
                        print(f"    -> FULLY CLEARS debit {amt0:,.2f} from {dt0.date()}")
                    rem -= amt0
                else:
                    leftover = amt0 - rem
                    if debug:
                        print(f"    -> PARTIALLY CLEARS debit {amt0:,.2f} from {dt0.date()}, leftover {leftover:,.2f}")
                    q.appendleft([dt0, leftover])  # preserve debit's original date
                    rem = 0.0
            if rem > 1e-9:
                if debug:
                    print(f"    -> CREDIT {rem:,.2f} UNMATCHED, carried as advance dated {as_of}")
                q.append([pd.to_datetime(as_of), -rem])

    open_items = [(dt, amt) for dt, amt in q if abs(amt) > 1e-9]

    if debug:
        print("\nRemaining open items after FIFO:")
        for dt, amt in open_items:
            age = (as_of - dt.date()).days if pd.notna(dt) else None
            print(f"  {amt:,.2f} dated {dt.date()} (age {age} days)")

    fifo_total = round(sum(amt for _, amt in open_items), 2)
    recon_diff = round(outstanding - fifo_total, 2)

    # --- Bucket labels
    def _bucket_label(age):
        if day30_in_first_bucket:
            if age <= 30: return "Less than 30 days"
            if age <= 60: return "31-60 days"
            if age <= 90: return "61-90 days"
            if age <= 120: return "91-120 days"
            if age <= 150: return "121-150 days"
            if age <= 180: return "151-180 days"
            return "181 days and above"
        else:
            if age < 30: return "Less than 30 days"
            if age <= 60: return "31-60 days"
            if age <= 90: return "61-90 days"
            if age <= 120: return "91-120 days"
            if age <= 150: return "121-150 days"
            if age <= 180: return "151-180 days"
            return "181 days and above"

    # --- Bucketing
    buckets = {
        "Less than 30 days": 0.0,
        "31-60 days": 0.0,
        "61-90 days": 0.0,
        "91-120 days": 0.0,
        "121-150 days": 0.0,
        "151-180 days": 0.0,
        "181 days and above": 0.0,
        "Advance Credit": 0.0
    }

    breakdown_rows = []
    for inv_dt, amt in open_items:
        if amt >= 0:
            age = (as_of - inv_dt.date()).days
            b = _bucket_label(age)
            buckets[b] += amt
            if return_breakdown:
                breakdown_rows.append({
                    "Invoice Date": inv_dt.date(),
                    "Open Amount": round(amt, 2),
                    "Age (days)": age,
                    "Bucket": b
                })
        else:
            buckets["Advance Credit"] += amt
            if return_breakdown:
                breakdown_rows.append({
                    "Invoice Date": inv_dt.date(),
                    "Open Amount": round(amt, 2),
                    "Age (days)": 0,
                    "Bucket": "Advance Credit"
                })

    # --- Summary
    def rounding(x):
        return round(x, 2)
    summary = {
        "Date": pd.Timestamp(as_of).strftime("%d-%b-%Y"),
        "Customer Name": customer_name,
        "Current Outstanding in UGX": rounding(outstanding),
        **{k: rounding(v) for k, v in buckets.items()},
        #"Buckets Total (FIFO)": fifo_total,
        #"Recon Diff": recon_diff
    }

    # --- Percentages row
    pct_row = {
        "Date": "",
        "Customer Name": "% of Total",
        "Current Outstanding in UGX": "",
        **{k: (f"{round(v/outstanding*100,2)}%" if outstanding != 0 else "0.0%") for k, v in buckets.items()},
        #"Buckets Total (FIFO)": "100.0%" if outstanding != 0 else "0.0%",
        #"Recon Diff": ""
    }

    summary_df = pd.DataFrame([summary, pct_row])

    return summary_df

# -------- converting to excel ------------------ # 
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Ageing Analysis")
    processed_data = output.getvalue()
    return processed_data

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="📊 ***Aging Analysis Tool***", layout="wide")

# Page title
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>📊 Aging Analysis Tool</h1>",
    unsafe_allow_html=True
)
st.write("Upload a customer statement (PDF) and get an AR aging report in Excel format")

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded successfully.")

    df, customer_name = extract_ledger_by_blocks_strict(uploaded_file)
    if df.empty:
        st.error("⚠️ No valid invoices/amounts found in the PDF.")
    else:
        result_df = aging_analysis(df, customer_name=customer_name, debug=False)

        # --- Styling DataFrame ---
        styled = result_df.style.set_table_styles(
            [
                {"selector": "th", "props": [("font-weight", "bold"),
                                             ("background-color", "#2E86C1"),
                                             ("color", "white"),
                                             ("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]}
            ]
        ).set_properties(**{"font-size": "14px", "padding": "6px"})

        st.markdown(f"### 📍 Customer: **{customer_name}**")
        st.dataframe(styled, use_container_width=True)

        # Excel download
        excel = convert_df_to_excel(result_df)
        st.download_button(
            "📥 Download Excel",
            data=excel,
            file_name="aging_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



