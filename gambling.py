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
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import math

def read_odds(odds_file="nba_odds_2018-19.xlsx"):
    pd.set_option('display.max_columns', None)
    odds = pd.read_excel(odds_file, sheet_name="Sheet1")
    game_nums = []
    for i in range(len(odds)):
        game_nums.append(int(i//2))
    odds["game_number"] = game_nums
    over_under = odds[['Date', 'Rot', 'VH', 'Team','Final', 'Open', 'game_number']]

    over_under_home = over_under[over_under["VH"] == "H"]
    #print(over_under_home)
    over_under_visitor = over_under[over_under["VH"] == "V"]
    #print(over_under_visitor)
    over_under_home.set_index("game_number")
    over_under_visitor.set_index("game_number")

    MasterTable = over_under_home.join(over_under_visitor, how='inner', on="game_number", lsuffix="_home", rsuffix="_visitor")
    MasterTable.set_index("game_number_home")
    #print(MasterTable.columns)
    print(MasterTable)
    #return MasterTable

def get_game_odds():
    odds_table = read_odds()




def get_stats(team_stats_file="18_19_team_stats.csv", opp_stats_file='18_19_opponent_stats.csv'):
    Tstat = pd.read_csv(team_stats_file, index_col=0)
    print(Tstat.columns)
    TeamStats = Tstat[['Team', 'FG%', '3P%', 'FT%', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']].copy()
    TeamStats.set_index('Team')
    scalar = MinMaxScaler()
    TeamStats[['FG%', '3P%', 'FT%', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']] = scalar.fit_transform(
        TeamStats[['FG%', '3P%', 'FT%', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']])

    Ostat = pd.read_csv(opp_stats_file, index_col=0)
    print(Ostat.columns)
    OppStats = Ostat[['Team', 'FG%', '3P%', 'FT%', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']].copy()
    OppStats.set_index('Team')
    scalar = MinMaxScaler()
    OppStats[['FG%', '3P%', 'FT%', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']] = scalar.fit_transform(
        OppStats[['FG%', '3P%', 'FT%', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']])

    MasterTable = TeamStats.join(OppStats, lsuffix="_team", rsuffix="_opp")
    print(MasterTable.columns)
    return MasterTable