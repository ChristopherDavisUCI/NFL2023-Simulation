import pandas as pd
from odds_helper import kelly, odds_to_prob

def win_div(raw_data):
    df = raw_data.set_index("Team", drop=True)
    df = df[df["Seed"] == 4].copy()
    return df["Equal_better"] 

def compare_market(raw_data):
    market = pd.read_csv("data/markets.csv")
    ser_div = win_div(raw_data)
    market["prob"] = market["team"].map(ser_div)
    market["kelly"] = market.apply(lambda row: kelly(row["prob"], row["odds"]), axis=1)
    rec = market[market["kelly"] > 0].sort_values("kelly", ascending=False)
    return rec[["team", "odds", "prob", "site", "kelly"]].reset_index(drop=True)