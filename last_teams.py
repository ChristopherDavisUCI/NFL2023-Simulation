import pandas as pd

def outcome_dct(row):
    if row["home_score"] > row["away_score"]:
        winner = row["home_team"]
        loser = row["away_team"]
    elif row["away_score"] > row["home_score"]:
        winner = row["away_team"]
        loser = row["home_team"]
    else:
        # Tie game, no winner or loser
        return {}
    
    return {"winner": winner, "loser": loser}

def get_last(df):
    df = df.copy()
    df[["winner", "loser"]] = pd.DataFrame(list(df.apply(outcome_dct, axis=1)))
    
    winless = set(df["away_team"])
    undefeated = set(df["away_team"])
    
    for i in range(1,19):
        df_week = df[df["week"] == i]
        
        winners = set(df_week["winner"])
        temp = winless.difference(winners)
        if len(temp) == 0:
            last_winless = winless
            break
        else:
            winless = temp

        if i == 18:
            last_winless = winless
            

        
    for i in range(1,19):
        df_week = df[df["week"] == i]
        
        losers = set(df_week["loser"])
        temp = undefeated.difference(losers)
        if len(temp) == 0:
            last_undefeated = undefeated
            break
        else:
            undefeated = temp

        if i == 18:
            last_undefeated = undefeated
            
    return {"last_undefeated": tuple(last_undefeated), "last_winless": tuple(last_winless)}

