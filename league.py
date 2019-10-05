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


API_KEY = 'RGAPI-3d5ec4ec-507d-4546-8565-8741ea1ec442'
GAMES = 1000
# default_scalar = MinMaxScaler()

class LeagueClassify:
    def __init__(self, api=API_KEY, summoner_name='Lefty2323', collect_data=False):
        self.api = api
        self.cass_sets() # applut cassiopeia settings 
        self.summoner = cass.get_summoner(name=summoner_name)
        if collect_data:
            self.dat = self.collectData()
        else:
            self.dat = pd.read_csv(f"DataSets/{self.summoner.sanitized_name}_data.csv")

    def cass_sets(self):
        setting = cass.get_default_config()
        setting['logging']['print_calls'] = False
        cass.apply_settings(setting)

        cass.set_riot_api_key(self.api)
        cass.set_default_region("NA")

    def collectData(self):
        df = pd.DataFrame()

        print('Collecting Data')
        
        for i in range(GAMES):
            curr_game = self.summoner.match_history[i]
            gtime = curr_game.duration.seconds/60
            stats_df = [] # list of seven things in order of 'kda', 'csd', 'gold', 'gtime', 'dmg', 'ward', 'outcome' 
            if curr_game.map.name != "Summoner's Rift":
                continue
            for p in curr_game.participants:
                if p.summoner.name == self.summoner.name:
                    raw_stats = p.stats 
                    
                    stats_df = [raw_stats.kda, raw_stats.total_minions_killed, raw_stats.gold_earned, gtime,  raw_stats.total_damage_dealt_to_champions, raw_stats.vision_score, raw_stats.win*1]
                    stats_df = pd.DataFrame([stats_df], columns=['kda', 'csd', 'gold', 'gtime', 'dmg', 'ward', 'outcome'])
                    df = df.append(stats_df)
        
        print('Data Collection Complete')

        df.to_csv(f"DataSets/{self.summoner.sanitized_name}_data.csv")

        return df

def logRegession(dat, y_label, X_labels, scalar):
    print('Model Training Started')
    X_train, X_test, y_train, y_test = train_test_split(dat[X_labels], dat[y_label], test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    logreg.fit(X_train, y_train)
    
    print('Model training complete')

    return logreg

def UI():
    features = ['KDA', 'Creep Score', 'Total Gold', 'Total damage (champions)', 'Vision Score']
    
    print('Hello!')
    summ_name = input("Enter summoner name: ").lower()
    
    coll_data = False
    if not os.path.exists(f"DataSets/{summ_name}_data.csv"):
        coll_data = True
        print('Data Collection Required...')

    lol = LeagueClassify(summoner_name=summ_name, collect_data=coll_data)
    scalar = StandardScaler()
    lol.logreg = logRegession(lol.dat, ['outcome'], ['kda', 'csd', 'gold', 'dmg', 'ward'], scalar)
    coefficients = scalar.inverse_transform(lol.logreg.coef_, True)
    result = get_context(coefficients[0])
    print(result)


def get_context(coefficients):
    kda, cs, gold, dmg, vision = coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4]
    return f'''Based on the last 1000 games, you require the following stats to win a game:\n \
    KDA: {float("{0:.2f}".format(kda))}\n \
    Creep Score: {int(cs)}\n \
    Gold: {int(gold)}\n \
    Damage done to champions: {float("{0:.2f}".format(dmg))}\n \
    Vision Score: {int(vision)}\n \
    '''
    

if __name__ == "__main__":
    UI()