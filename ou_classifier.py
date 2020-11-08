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
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
from sklearn.metrics import classification_report, confusion_matrix
import gambling

class OU_predictor():
    def __init__(self, model_param = 5, model_type = "knn"):
        odds17 = gambling.read_odds(odds_file="nba_odds_2017-18.xlsx")
        odds18 = gambling.read_odds(odds_file="nba_odds_2018-19.xlsx")
        odds19 = gambling.read_odds(odds_file="nba_odds_2019-20.xlsx")

        stats17 = gambling.get_stats(team_stats_file="17TS.csv", opp_stats_file='17OPP.csv')
        stats18 = gambling.get_stats(team_stats_file="18_19_team_stats.csv", opp_stats_file='18_19_opponent_stats.csv')
        stats19 = gambling.get_stats(team_stats_file="19TS.csv", opp_stats_file='19OPP.csv')

        data17 = gambling.merge_normalize_Stats_Odds(odds17, stats17)
        data18 = gambling.merge_normalize_Stats_Odds(odds18, stats18)
        data19 = gambling.merge_normalize_Stats_Odds(odds19, stats19)

        concated = gambling.concatYears(data17, data18, data19)
        #self.game_team_data = concated.drop("game_number")
        self.game_team_data = concated
        self.game_team_data.reset_index(drop=True, inplace=True)

        self.get_split()

        if model_type == "knn":
            self.num_neighbors = model_param
            self.set_up_knn()

        if model_type == "nn_radius":
            self.radius = model_param
            self.set_up_radius()

        if model_type == "custom":
            self.num_neighbors = model_param
            self.threshhold = self.num_neighbors // 2 #use this to decide how much greater the count of O needs to be than U for us to feel sure about the classification,  or vice versa
            self.set_up_custom

    def get_split(self):
        features = self.game_team_data[['OverUnder', 'FG%_V', '3P%_V', 'FT%_V', 'FTA_V', 'ORB_V',
                         'TRB_V', 'AST_V', 'STL_V', 'BLK_V', 'TOV_V', 'PTS_V', 'OFG%_V',
                         'O3P%_V', 'OORB_V', 'OTRB_V',
                         'OTOV_V', 'OPTS_V', 'FG%_H', '3P%_H', 'FT%_H',
                         'FTA_H', 'ORB_H', 'TRB_H', 'AST_H', 'STL_H', 'BLK_H', 'TOV_H', 'PTS_H',
                         'OFG%_H', 'O3P%_H', 'OORB_H', 'OTRB_H', 'OTOV_H', 'OPTS_H']].copy()
        labels = self.game_team_data['OUresult']
        self.X_train,self.X_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.05)
        self.X_train.reset_index(drop=True, inplace=True) #need to reset the indexes for consisten referencing
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)

    def getNN(self, points): #takes in a set of points. Gets the nearest neighbors for each point, and the classifies based on those neighbors and a threshold.
                            #returns a dictionary of where the keys are indices of the points, and the values are the classification for those points.
        OU_classifications = {}
        distances, indices = self.model.kneighbors(points)
        for i in range(indices):
            o_count = 0
            u_count = 0
            for neighbor in indices[i]:
                if self.y_train == "O":
                    o_count +=1
                else:
                    u_count += 1
            if o_count - self.threshhold > u_count:
                OU_classifications[i] = "O"
            elif u_count - self.threshhold > o_count:
                OU_classifications[i] = "U"
            else:
                OU_classifications[i] = "?"

        return OU_classifications


    def set_up_custom(self):
        self.get_split()
        self.model = NearestNeighbors(n_neighbors=self.num_neighbors, algorithm="ball_tree").fit(self.X_train)
        answer_dict = self. getNN(self.X_test)

    def set_up_radius(self):
        self.classifier = RadiusNeighborsClassifier(radius=self.radius)
        self.classifier.fit(self.X_train, self.y_train)
        y_pred = self.classifier.predict(self.X_test)
        print("confusion_matrix: \n")
        print(confusion_matrix(self.y_test, y_pred))
        print("classification_report\n")
        return classification_report(self.y_test, y_pred)

    def set_up_knn(self):
        self.classifier = KNeighborsClassifier(n_neighbors=self.num_neighbors)
        self.classifier.fit(self.X_train, self.y_train)
        y_pred = self.classifier.predict(self.X_test)
        print("confusion_matrix: \n")
        print(confusion_matrix(self.y_test, y_pred))
        print("classification_report\n")
        return classification_report(self.y_test, y_pred)