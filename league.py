import cassiopeia as cass
import pandas as pd
import numpy as np
import time
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import os.path

# ignore warning from sklearn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


API_KEY = 'RGAPI-12648c0e-7c50-4d99-8193-f088cdb1fccc'
GAMES = 100
# default_scalar = MinMaxScaler()

class LeagueClassify:
    def __init__(self, api=API_KEY, summoner_name='Lefty2323', collect_data=False):
        self.api = api
        self.cass_sets() # applut cassiopeia settings 
        self.summoner = cass.get_summoner(name=summoner_name)
        # self.scalar = MinMaxScaler()
        if collect_data:
            self.dat = self.collectData()
        else:
            self.dat = pd.read_csv(f"DataSets/{self.summoner.sanitized_name}_data.csv")
        # self.knn = self.model()


    def cass_sets(self):
        setting = cass.get_default_config()
        setting['logging']['print_calls'] = False
        cass.apply_settings(setting)

        cass.set_riot_api_key(self.api)
        cass.set_default_region("NA")

    def collectData(self):
        df = pd.read_csv(f"DataSets/{self.summoner.sanitized_name}_data.csv")
        
        for i in range(GAMES):
            curr_game = self.summoner.match_history[i]
            gtime = curr_game.duration.seconds/60
            stats_df = [] # list of seven things in order of 'kda', 'csd', 'gold', 'gtime', 'dmg', 'ward', 'outcome' 
            if curr_game.map.name != "Summoner's Rift":
                continue
            for p in curr_game.participants:
                if p.summoner.name == self.summoner.name:
                    raw_stats = p.stats # parse through raw_stats and put into stats_df **************************************
                    
                    stats_df = [raw_stats.kda, raw_stats.total_minions_killed, raw_stats.gold_earned, gtime,  raw_stats.total_damage_dealt_to_champions, raw_stats.vision_score, raw_stats.win*1]
                    stats_df = pd.DataFrame([stats_df], columns=['kda', 'csd', 'gold', 'gtime', 'dmg', 'ward', 'outcome'])
                    df = df.append(stats_df)
        
        df.to_csv(f"DataSets/{self.summoner.sanitized_name}_data.csv")

        return df

def logRegession(dat, y_label, X_labels, scalar):
    X_train, X_test, y_train, y_test = train_test_split(dat[X_labels], dat[y_label], test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    logreg.fit(X_train, y_train)
    
    return logreg

def my_pred(predList, logreg, scalar):
    '''Take a list of stats, and apply the model'''
    predictors = scalar.transform(np.array(predList).reshape(1, -1))
    pred = logreg.predict(predictors)
    return pred

def UI():
    features = ['KDA', 'Creep Score', 'Total Gold', 'Total damage (champions)', 'Vision Score']
    
    print('Hello!')
    summ_name = input("Enter summoner name: ").lower()
    
    coll_data = False
    if not os.path.exists(f"DataSets/{summ_name}_data.csv"):
        coll_data = True
    
    stats = []
    print("Enter the values for the following statistics")
    for stat in features:
        stats.append(input(f"Enter {stat}: "))

    lol = LeagueClassify(summoner_name=summ_name, collect_data=coll_data)
    scalar = StandardScaler()
    lol.logreg = logRegession(lol.dat, ['outcome'], ['kda', 'csd', 'gold', 'dmg', 'ward'], scalar)
    outcome = my_pred(stats, lol.logreg, scalar)
    
    if outcome == 1:
        print("The model predicts a win")
    else:
        print("The model predicts a loss")


if __name__ == "__main__":
    UI()