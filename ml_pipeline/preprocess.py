import pandas as pd

df = pd.read_csv("../data/credit_card_fraud_dataset.csv")

df.drop_duplicates(inplace=True)

df.fillna(0, inplace=True)

df.to_csv("../data/processed.csv", index=False)