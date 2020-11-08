#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pandas as pd
from pandas import ExcelFile
from pandas import ExcelWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
import numpy as np
import math
from sklearn.metrics import classification_report, confusion_matrix


def OverUnder(row):
    if row['Open_x'] > 50 :
        return row['Open_x']
    else:
        return row['Open_y']

def OUresult(row):
    if row['Final_x'] + row['Final_y'] > row['OverUnder']:
        return 'O'
    else:
        return 'U'

def read_odds(odds_file="nba_odds_2018-19.xlsx"):
    pd.set_option('display.max_columns', None)
    odds = pd.read_excel(odds_file, sheet_name="Sheet1")
    odds.reset_index(inplace=True)

    game_nums = []
    for i in range(len(odds)):
        game_nums.append(int(i//2))

    odds["game_number"] = game_nums
    over_under = odds[['Date', 'Rot', 'VH', 'Team','Final', 'Open', 'game_number']]
    over_under_home = over_under[over_under["VH"] == "H"]
    over_under_visitor = over_under[over_under["VH"] == "V"]

    MasterTable = pd.merge(over_under_visitor, over_under_home, how='left', on="game_number")
    # MasterTable.set_index("game_number_home")
    #print(MasterTable.columns)
    # print(MasterTable.head(25))
    MasterTable.loc[(MasterTable.Open_x == 'PK'),'Open_x']=0
    MasterTable.loc[(MasterTable.Open_y == 'PK'),'Open_y']=0
    MasterTable.loc[(MasterTable.Open_x == 'pk'),'Open_x']=0
    MasterTable.loc[(MasterTable.Open_y == 'pk'),'Open_y']=0
    MasterTable["Open_x"] = pd.to_numeric(MasterTable["Open_x"])
    MasterTable["Open_y"] = pd.to_numeric(MasterTable["Open_y"])
    MasterTable['OverUnder'] = MasterTable.apply(lambda row: OverUnder(row), axis=1)
    MasterTable['OUresult'] = MasterTable.apply(lambda row: OUresult(row), axis=1)
    MasterTable = MasterTable.rename({'Date_x': 'Date', 'Final_x': 'Final_Visitor', 'Final_y': 'Final_Home', 'Team_x':'Visitor', 'Team_y':'Home'}, axis=1)  # new method
    MasterTable = MasterTable[['game_number', 'Date', 'Visitor', 'Final_Visitor', 'Home', 'Final_Home', 'OverUnder', 'OUresult']]
    MasterTable.to_excel("output.xlsx")
    # print(MasterTable.head(25))
    return MasterTable
    #return MasterTable


def get_game_odds():
    odds_table = read_odds()




def get_stats(team_stats_file="18_19_team_stats.csv", opp_stats_file='18_19_opponent_stats.csv'):
    # pd.set_option('display.max_columns', None)
    Tstat = pd.read_csv(team_stats_file, index_col=0)
    TOstat = pd.read_csv(opp_stats_file, index_col=0)
    TStats = Tstat[['Team', 'FG%', '3P%', 'FT%', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']].copy()
    TOStats = TOstat[['Team', 'FG%', '3P%', 'FT%', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']].copy()
    TOStats.columns = ['Team', 'OFG%', 'O3P%', 'OFT%', 'OFTA', 'OORB', 'OTRB', 'OAST', 'OSTL', 'OBLK', 'OTOV', 'OPTS']
    TStats.set_index('Team')
    TOStats.set_index('Team')
    TeamStats = pd.merge(TStats, TOStats, how='inner', on="Team")

    return TeamStats


def merge_normalize_Stats_Odds(odds, TeamStats):
    pd.set_option('display.max_rows', None)
    Master = pd.merge(odds, TeamStats, how='left', left_on='Visitor', right_on='Team')
    Master2 = pd.merge(Master, TeamStats, how = 'left', left_on='Home', right_on='Team')
    Master2.columns = ['game_number', 'Date', 'Visitor', 'Final_Visitor', 'Home', 'Final_Home',
       'OverUnder', 'OUresult','Team_V', 'FG%_V', '3P%_V', 'FT%_V', 'FTA_V', 'ORB_V',
       'TRB_V', 'AST_V', 'STL_V', 'BLK_V', 'TOV_V', 'PTS_V', 'OFG%_V',
       'O3P%_V', 'OFT%_V', 'OFTA_V', 'OORB_V', 'OTRB_V', 'OAST_V', 'OSTL_V',
       'OBLK_V', 'OTOV_V', 'OPTS_V', 'Team_H', 'FG%_H', '3P%_H', 'FT%_H',
       'FTA_H', 'ORB_H', 'TRB_H', 'AST_H', 'STL_H', 'BLK_H', 'TOV_H', 'PTS_H',
       'OFG%_H', 'O3P%_H', 'OFT%_H', 'OFTA_H', 'OORB_H', 'OTRB_H', 'OAST_H',
       'OSTL_H', 'OBLK_H', 'OTOV_H', 'OPTS_H']
    
    scalar = MinMaxScaler()
    
    Master2[['OverUnder', 'FG%_V', '3P%_V', 'FT%_V', 'FTA_V', 'ORB_V',
       'TRB_V', 'AST_V', 'STL_V', 'BLK_V', 'TOV_V', 'PTS_V', 'OFG%_V',
       'O3P%_V', 'OFT%_V', 'OFTA_V', 'OORB_V', 'OTRB_V', 'OAST_V', 'OSTL_V',
       'OBLK_V', 'OTOV_V', 'OPTS_V', 'FG%_H', '3P%_H', 'FT%_H',
       'FTA_H', 'ORB_H', 'TRB_H', 'AST_H', 'STL_H', 'BLK_H', 'TOV_H', 'PTS_H',
       'OFG%_H', 'O3P%_H', 'OFT%_H', 'OFTA_H', 'OORB_H', 'OTRB_H', 'OAST_H',
       'OSTL_H', 'OBLK_H', 'OTOV_H', 'OPTS_H']] = scalar.fit_transform(Master2[['OverUnder', 'FG%_V', '3P%_V', 'FT%_V', 'FTA_V', 'ORB_V',
       'TRB_V', 'AST_V', 'STL_V', 'BLK_V', 'TOV_V', 'PTS_V', 'OFG%_V',
       'O3P%_V', 'OFT%_V', 'OFTA_V', 'OORB_V', 'OTRB_V', 'OAST_V', 'OSTL_V',
       'OBLK_V', 'OTOV_V', 'OPTS_V', 'FG%_H', '3P%_H', 'FT%_H',
       'FTA_H', 'ORB_H', 'TRB_H', 'AST_H', 'STL_H', 'BLK_H', 'TOV_H', 'PTS_H',
       'OFG%_H', 'O3P%_H', 'OFT%_H', 'OFTA_H', 'OORB_H', 'OTRB_H', 'OAST_H',
       'OSTL_H', 'OBLK_H', 'OTOV_H', 'OPTS_H']])

    Master2.drop(columns=['Date', 'Team_H', 'Team_V'])
    Master2.set_index("game_number")

    return Master2


def trainTest(data):
    features = data[['OverUnder', 'FG%_V', '3P%_V', 'FT%_V', 'FTA_V', 'ORB_V',
       'TRB_V', 'AST_V', 'STL_V', 'BLK_V', 'TOV_V', 'PTS_V', 'OFG%_V',
       'O3P%_V','OORB_V', 'OTRB_V',
       'OTOV_V', 'OPTS_V', 'FG%_H', '3P%_H', 'FT%_H',
       'FTA_H', 'ORB_H', 'TRB_H', 'AST_H', 'STL_H', 'BLK_H', 'TOV_H', 'PTS_H',
       'OFG%_H', 'O3P%_H', 'OORB_H', 'OTRB_H', 'OTOV_H', 'OPTS_H']].copy()
    labels = data['OUresult']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)
    #classifier = RadiusNeighborsClassifier(radius=1.6)
    classifier = KNeighborsClassifier(n_neighbors=8)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("confusion_matrix: \n")
    print(confusion_matrix(y_test, y_pred))
    print("classification_report\n")
    return classification_report(y_test, y_pred)




odds17 = read_odds(odds_file="nba_odds_2017-18.xlsx")
odds18 = read_odds(odds_file="nba_odds_2018-19.xlsx")
odds19 = read_odds(odds_file="nba_odds_2019-20.xlsx")


stats17 = get_stats(team_stats_file="17TS.csv", opp_stats_file='17OPP.csv')
stats18 = get_stats(team_stats_file="18_19_team_stats.csv", opp_stats_file='18_19_opponent_stats.csv')
stats19 = get_stats(team_stats_file="19TS.csv", opp_stats_file='19OPP.csv')


data17 = merge_normalize_Stats_Odds(odds17, stats17)
data18 = merge_normalize_Stats_Odds(odds18, stats18)
data19 = merge_normalize_Stats_Odds(odds19, stats19)


def concatYears(data1, data2, data3):
    con1 = pd.concat([data1, data2])
    con2 = pd.concat([con1, data3])
    con2.to_excel("output.xlsx")
    return con2

#print(trainTest(concatYears(data17, data18, data19)))


def runTest():
    print(trainTest(concatYears(data17, data18, data19)))
