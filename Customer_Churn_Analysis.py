# ============================================
# PROJECT: CUSTOMER CHURN ANALYSIS
# DATA ANALYST PROJECT
# ============================================

# Install libraries:
# pip install pandas numpy matplotlib seaborn

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 2. LOAD DATASET

df = pd.read_csv("Customer_Churn.csv")

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA INFORMATION")
df.info()

# 3. DATA CLEANING

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nDUPLICATE CUSTOMER IDs")
print(df["Customer_ID"].duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove duplicate customer IDs
df = df.drop_duplicates(subset="Customer_ID")

# Convert charges to numeric values
df["Monthly_Charges"] = pd.to_numeric(
    df["Monthly_Charges"], errors="coerce"
)

df["Total_Charges"] = pd.to_numeric(
    df["Total_Charges"], errors="coerce"
)

# 4. CREATE CHURN FLAG

# Yes = 1
# No = 0

df["Churn_Flag"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# 5. KEY PERFORMANCE INDICATORS (KPIs)

total_customers = df["Customer_ID"].nunique()

churned_customers = (
    df[df["Churn"] == "Yes"]["Customer_ID"].nunique()
)

retained_customers = (
    df[df["Churn"] == "No"]["Customer_ID"].nunique()
)

churn_rate = (
    churned_customers / total_customers
) * 100

avg_monthly_charges = df["Monthly_Charges"].mean()

avg_tenure = df["Tenure_Months"].mean()

total_monthly_revenue = df["Monthly_Charges"].sum()

print("\n========== CUSTOMER CHURN KPIs ==========")

print("Total Customers:", total_customers)

print("Churned Customers:", churned_customers)

print("Retained Customers:", retained_customers)

print(f"Churn Rate: {churn_rate:.2f}%")

print(
    f"Average Monthly Charges: "
    f"{avg_monthly_charges:.2f}"
)

print(
    f"Average Tenure: "
    f"{avg_tenure:.1f} months"
)

print(
    f"Total Monthly Charges: "
    f"{total_monthly_revenue:.2f}"
)

# 6. CHURN ANALYSIS BY CATEGORY

def show_churn_analysis(column):

    print(f"\nCHURN COUNT BY {column}")

    print(
        pd.crosstab(
            df[column],
            df["Churn"],
            margins=True
        )
    )

    print(f"\nCHURN RATE BY {column} (%)")

    rate = (
        pd.crosstab(
            df[column],
            df["Churn"],
            normalize="index"
        )["Yes"] * 100
    )

    print(rate.round(2))


# Analyze different categories

categories = [
    "Gender",
    "Senior_Citizen",
    "Partner",
    "Dependents",
    "Internet_Service",
    "Contract",
    "Payment_Method",
    "Tech_Support",
    "Online_Security"
]

for col in categories:
    show_churn_analysis(col)

# 7. NUMERICAL ANALYSIS

print("\nAVERAGE VALUES BY CHURN")

print(
    df.groupby("Churn")[
        [
            "Tenure_Months",
            "Monthly_Charges",
            "Total_Charges"
        ]
    ].mean().round(2)
)

# 8. CREATE OUTPUT FOLDER

Path("churn_outputs").mkdir(
    exist_ok=True
)

# 9. CHART 1: CHURN DISTRIBUTION

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Churn",
    order=["No", "Yes"]
)

plt.title("Customer Churn Distribution")

plt.xlabel("Churn")

plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "churn_outputs/01_churn_distribution.png",
    dpi=150
)

plt.show()

# 10. CHART 2: CHURN BY CONTRACT

contract_rate = (
    df.groupby("Contract", as_index=False)
    ["Churn_Flag"].mean()
)

contract_rate["Churn_Rate_Percent"] = (
    contract_rate["Churn_Flag"] * 100
)

plt.figure(figsize=(8, 4))

sns.barplot(
    data=contract_rate,
    x="Contract",
    y="Churn_Rate_Percent"
)

plt.title("Churn Rate by Contract")

plt.xlabel("Contract")

plt.ylabel("Churn Rate (%)")

plt.tight_layout()

plt.savefig(
    "churn_outputs/02_churn_by_contract.png",
    dpi=150
)

plt.show()

# 11. CHART 3: CHURN BY INTERNET SERVICE

internet_rate = (
    df.groupby("Internet_Service", as_index=False)
    ["Churn_Flag"].mean()
)

internet_rate["Churn_Rate_Percent"] = (
    internet_rate["Churn_Flag"] * 100
)

plt.figure(figsize=(8, 4))

sns.barplot(
    data=internet_rate,
    x="Internet_Service",
    y="Churn_Rate_Percent"
)

plt.title("Churn Rate by Internet Service")

plt.xlabel("Internet Service")

plt.ylabel("Churn Rate (%)")

plt.tight_layout()

plt.savefig(
    "churn_outputs/03_churn_by_internet.png",
    dpi=150
)

plt.show()

# 12. CHART 4: CHURN BY PAYMENT METHOD

payment_rate = (
    df.groupby("Payment_Method", as_index=False)
    ["Churn_Flag"].mean()
)

payment_rate["Churn_Rate_Percent"] = (
    payment_rate["Churn_Flag"] * 100
)

plt.figure(figsize=(10, 4))

sns.barplot(
    data=payment_rate,
    x="Payment_Method",
    y="Churn_Rate_Percent"
)

plt.title("Churn Rate by Payment Method")

plt.xlabel("Payment Method")

plt.ylabel("Churn Rate (%)")

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "churn_outputs/04_churn_by_payment.png",
    dpi=150
)

plt.show()

# 13. CHART 5: TENURE BY CHURN

plt.figure(figsize=(8, 4))

sns.histplot(
    data=df,
    x="Tenure_Months",
    hue="Churn",
    bins=18,
    multiple="stack"
)

plt.title("Customer Tenure by Churn Status")

plt.xlabel("Tenure (Months)")

plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "churn_outputs/05_tenure_by_churn.png",
    dpi=150
)

plt.show()

# 14. CHART 6: MONTHLY CHARGES BY CHURN

plt.figure(figsize=(7, 4))

sns.boxplot(
    data=df,
    x="Churn",
    y="Monthly_Charges",
    order=["No", "Yes"]
)

plt.title("Monthly Charges by Churn Status")

plt.xlabel("Churn")

plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    "churn_outputs/06_charges_by_churn.png",
    dpi=150
)

plt.show()

# 15. EXPORT ANALYZED DATA
# 15. EXPORT CLEANED DATA

# Create output folder
Path("churn_outputs").mkdir(exist_ok=True)

# Export cleaned data to CSV
df.to_csv(
    "churn_outputs/Customer_Churn_Cleaned.csv",
    index=False

)

# 15. EXPORT CLEANED DATA

# Create output folder
Path("churn_outputs").mkdir(exist_ok=True)

# Export cleaned data to CSV
df.to_csv(
    "churn_outputs/Customer_Churn_Cleaned.csv",
    index=False
)

# Export cleaned data to Excel
df.to_excel(
    "churn_outputs/Customer_Churn_Cleaned.xlsx",
    index=False,
    sheet_name="Cleaned_Data"
)

print("\n===================================")
print("CUSTOMER CHURN ANALYSIS COMPLETED")
print("===================================")

print("Cleaned CSV file created successfully!")
print("Cleaned Excel file created successfully!")
print("Charts saved in churn_outputs folder")