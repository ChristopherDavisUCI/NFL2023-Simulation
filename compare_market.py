import pandas as pd
import numpy as np
from odds_helper import kelly, odds_to_prob

def win_div(raw_data):
    '''Returns a Series indicating the probability of winning the division for each team.'''
    df = raw_data.set_index("Team", drop=True)
    df = df[df["Seed"] == 4].copy()
    return df["Equal_better"] 

def make_playoffs(raw_data):
    '''Returns a Series indicating the probability of making the playoffs for each team.'''
    df = raw_data.set_index("Team", drop=True)
    # Just want each team one time.  Could just as well say 7.
    df = df[df["Seed"] == 1].copy()
    return df["Make_playoffs"]

def get_prob(row, ser_div, ser_mp):
    if row["raw_market"] == "division":
        return ser_div[row["team"]]
    elif (row["raw_market"] == "make playoffs") & (row["result"] == "Yes"):
        return ser_mp[row["team"]]
    elif (row["raw_market"] == "make playoffs") & (row["result"] == "No"):
        return 1-ser_mp[row["team"]]
    
def name_market(row):
    if row["raw_market"] == "division":
        return "Win division"
    elif (row["raw_market"] == "make playoffs") & (row["result"] == "Yes"):
        return "Make playoffs - Yes"
    elif (row["raw_market"] == "make playoffs") & (row["result"] == "No"):
        return "Make playofffs - No"
    

def display_plus(s):
    if s[0] == "-":
        return s
    else:
        return "+"+s


# Columns for raw_data are:
# Seed, Team, Proportion, Make_playoffs, Equal_better
# Odds, Odds_Make_playoffs, Odds_Equal_better
def compare_market(raw_data):
    market = pd.read_csv("data/markets.csv")
    market.rename({"market": "raw_market"}, axis=1, inplace=True)
    ser_div = win_div(raw_data)
    ser_mp = make_playoffs(raw_data)
    market["prob"] = market.apply(lambda row: get_prob(row, ser_div, ser_mp), axis=1)
    market["market"] = market.apply(name_market, axis=1)
    market["kelly"] = market.apply(lambda row: kelly(row["prob"], row["odds"]), axis=1)
    rec = market[market["kelly"] > 0].sort_values("kelly", ascending=False)
    rec["odds"] = rec["odds"].astype(str).map(display_plus)
    return rec[["team", "market", "odds", "prob", "site", "kelly"]].reset_index(drop=True)