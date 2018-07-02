import pandas as pd
import numpy as np

class DataProcessor(object):
    
    def __init__(self, filename):
        self.filename = filename
        
    def run(self):
        
        df = pd.read_csv(self.filename)
        length = len(df)

        home_info = df[["League", "Home Team", "S.1", "EX.1", "EN.1", "F.1", "I.1", "R.1", "Away Team",
                        "S", "EX", "EN", "F", "I", "R", "Results.1"]]
        away_info = df[["League", "Away Team", "S", "EX", "EN", "F", "I", "R", "Home Team",
                        "S.1", "EX.1", "EN.1", "F.1", "I.1", "R.1", "Results"]]

        home_info = home_info.rename(index=str, columns={"League": "League", "Home Team": "Team","S.1": "S_own", 
                                            "EX.1": "EX_own", "EN.1": "EN_own", "F.1": "F_own", "I.1": "I_own",
                                            "R.1": "R_own", "Away Team":"Opponent Team", "S": "S_opp", "EX": "EX_opp", "EN": "EN_opp",
                                            "F": "F_opp", "I":"I_opp", "R": "R_opp", "Results.1": "Result"})
        away_info = away_info.rename(index=str, columns={"League":"League", "Away Team": "Team","S": "S_own", 
                                            "EX": "EX_own", "EN": "EN_own", "F": "F_own", "I": "I_own",
                                            "R": "R_own","Home Team":"Opponent Team", "S.1": "S_opp", "EX.1": "EX_opp", "EN.1": "EN_opp",
                                            "F.1": "F_opp", "I.1":"I_opp", "R.1": "R_opp", "Results": "Result"})


        information = home_info.append(away_info,ignore_index=True)

        information["Result"] = information["Result"].replace({'L':0, 'W':1})
        information["League"] = information["League"].replace({'NBA':0, 'NHL':1, 'NCAAB':2, 'NFL':3, 'MLB':4})

        ones = [1] * length
        zeros = [0] * length

        home_situation = ones + zeros
       
        home_situation = pd.DataFrame(home_situation)

        information["Home Situation"] = home_situation

        teams_home = information['Team'].drop_duplicates().values.tolist()
        teams_away = information['Opponent Team'].drop_duplicates().values.tolist()

        index = np.arange(0, len(teams_away),dtype=int)
        
        information["Team"] = information["Team"].replace({k:v for k,v in zip(teams_home,index)})
        information["Opponent Team"] = information["Opponent Team"].replace({k:v for k,v in zip(teams_home,index)})


        df = information.to_csv("input.csv", encoding="utf-8", index=False)

if __name__ == "__main__":
    csv_file = DataProcessor(filename='data.csv')
    csv_file.run()
