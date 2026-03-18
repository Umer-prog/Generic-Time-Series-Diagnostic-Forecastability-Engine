import sys
sys.path.insert(0, r"D:/OneDrive - Global Data 365/Analytics Engine templates/tests/data tests/ts_dfe")
from ts_dfe.engine import run_ts_dfe
import pandas as pd

df = pd.read_excel("Sales.xlsx")
#df = pd.read_csv("Walmart_Sales.csv")

report = run_ts_dfe(df, date_col="Date", target_col="Weekly_Sales", structural_cols=[""])

print(report)  # human-readable formatted report
raw_dict = dict(report)  # plain dict if needed for JSON/serialization
