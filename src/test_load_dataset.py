import pandas as pd

# Load the CSV from the data folder (go one level up to reach data/)
df = pd.read_csv("../data/interview_questions.csv")

print("✅ Rows in dataset:", len(df))
print("\nFirst few questions:\n", df.head(5))
