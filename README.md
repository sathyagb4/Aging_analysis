# Aging Analysis 

This project automates the extraction, processing, and ageing analysis of customer account ledgers — specifically built for Sugar Corporation of Uganda Ltd. 
The system parses messy PDF ledger statements,
cleans and structures transaction data, and computes customer balances, ageing summaries within brackets, and visual analytics for financial reporting.

### Key Features 
- PDF Extraction : uses pdfplumber to parse an unoragnised pdf for voucher numbers, cheque numbers, dates, debit/credit and balances automatically
- Aging Analysis : Calculates outstanding balances and groups them into ageing buckets based on voucher or invoice dates.
- Cleansing : Removes duplicates, handles OCR errors, and validates for running balance consistency.
- Output : results into CSV format for integration into a Streamlit app for displaying customer aging table.

## Streamlit app 
Link: https://financial-aging-analysis-scoul.streamlit.app/
