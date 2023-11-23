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


def get_prob(row, prob_dct):
    if row["raw_market"] == "division":
        ser = prob_dct["div"]
        return ser[row["team"]]
    elif (row["raw_market"] == "make playoffs") & (row["result"] == "Yes"):
        ser = prob_dct["mp"]
        return ser[row["team"]]
    elif (row["raw_market"] == "make playoffs") & (row["result"] == "No"):
        ser = prob_dct["mp"]
        return 1-ser[row["team"]]
    elif row["raw_market"] == "conference":
        ser = prob_dct["conf"]
        return ser[row["team"]]
    elif row["raw_market"] == "super bowl":
        ser = prob_dct["sb"]
        return ser[row["team"]]
    

def name_market(row):
    if row["raw_market"] == "division":
        return "Win division"
    elif (row["raw_market"] == "make playoffs") & (row["result"] == "Yes"):
        return "Make playoffs - Yes"
    elif (row["raw_market"] == "make playoffs") & (row["result"] == "No"):
        return "Make playoffs - No"
    elif row["raw_market"] == "conference":
        return "Conference Champion"
    elif row["raw_market"] == "super bowl":
        return "Super Bowl Champion"
    

def display_plus(s):
    if s[0] == "-":
        return s
    else:
        return "+"+s
    

def win_sb(champ_data):
    '''Returns a Series indicating the probability of winning the Super Bowl for each team.'''
    df = champ_data.set_index("Team", drop=True)
    df = df[df["Stage"] == "Win Super Bowl"].copy()
    return df["Proportion"]


def lose_sb(champ_data):
    '''Returns a Series indicating the probability of each team losing the Super Bowl.'''
    df = champ_data.set_index("Team", drop=True)
    df = df[df["Stage"] == "Lose in Super Bowl"].copy()
    return df["Proportion"] 


# Columns for raw_data are:
# Seed, Team, Proportion, Make_playoffs, Equal_better
# Odds, Odds_Make_playoffs, Odds_Equal_better
# Columns for champ_data are:
# Stage, Team, Proportion, Odds
def compare_market(raw_data, champ_data):
    market = pd.read_csv("data/markets.csv")
    market.rename({"market": "raw_market"}, axis=1, inplace=True)
    ser_div = win_div(raw_data)
    ser_mp = make_playoffs(raw_data)
    ser_sb = win_sb(champ_data)
    ser_lose = lose_sb(champ_data)
    # Winning the Conf = Win SB or Lose SB
    ser_conf = ser_sb + ser_lose
    prob_dct = {
        "div": ser_div,
        "mp": ser_mp,
        "sb": ser_sb,
        "conf": ser_conf
    }
    market["prob"] = market.apply(lambda row: get_prob(row, prob_dct), axis=1)
    market["market"] = market.apply(name_market, axis=1)
    market["kelly"] = market.apply(lambda row: kelly(row["prob"], row["odds"]), axis=1)
    rec = market[market["kelly"] > 0].sort_values("kelly", ascending=False)
    rec["odds"] = rec["odds"].astype(str).map(display_plus)
    return rec[["team", "market", "odds", "prob", "site", "kelly"]].reset_index(drop=True)
