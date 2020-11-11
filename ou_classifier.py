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
from statistics import mean

class OU_predictor():
    def __init__(self, model_param = 5, model_type = "knn", ratThresh = 0.7):
        odds17 = gambling.read_odds(odds_file="nba_odds_2017-18.xlsx")
        odds18 = gambling.read_odds(odds_file="nba_odds_2018-19.xlsx")
        odds19 = gambling.read_odds(odds_file="nba_odds_2019-20.xlsx")

        stats17 = gambling.get_stats(team_stats_file="17TS.csv", opp_stats_file='17OPP.csv')
        stats18 = gambling.get_stats(team_stats_file="18_19_team_stats.csv", opp_stats_file='18_19_opponent_stats.csv')
        stats19 = gambling.get_stats(team_stats_file="19TS.csv", opp_stats_file='19OPP.csv')
        self.statsEx = stats19
        data17 = gambling.merge_normalize_Stats_Odds(odds17, stats17)
        data18 = gambling.merge_normalize_Stats_Odds(odds18, stats18)
        data19 = gambling.merge_normalize_Stats_Odds(odds19, stats19)
        
        self.dataEx = data19
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
            # self.threshhold = ratThresh #use this to decide how much greater the count of O needs to be than U for us to feel sure about the classification,  or vice versa






    def get_split(self):
        features = self.game_team_data[['OverUnder', 'FG%_V', '3P%_V', 'FT%_V', 'FTA_V', 'ORB_V',
                         'TRB_V', 'AST_V', 'STL_V', 'BLK_V', 'TOV_V', 'PTS_V', 'OFG%_V',
                         'O3P%_V', 'OORB_V', 'OTRB_V',
                         'OTOV_V', 'OPTS_V', 'FG%_H', '3P%_H', 'FT%_H',
                         'FTA_H', 'ORB_H', 'TRB_H', 'AST_H', 'STL_H', 'BLK_H', 'TOV_H', 'PTS_H',
                         'OFG%_H', 'O3P%_H', 'OORB_H', 'OTRB_H', 'OTOV_H', 'OPTS_H']].copy()
        
        # features = self.game_team_data[['OverUnder', 'FG%_V', '3P%_V', 'FT%_V', 'FTA_V', 'ORB_V',
        #                  'TRB_V', 'AST_V', 'BLK_V', 'TOV_V', 'PTS_V', 'OFG%_V',
        #                  'O3P%_V', 'OORB_V', 'OTRB_V',
        #                  'OTOV_V', 'OPTS_V', 'FG%_H', '3P%_H', 'FT%_H',
        #                  'FTA_H', 'ORB_H', 'TRB_H', 'AST_H', 'BLK_H', 'TOV_H', 'PTS_H',
        #                  'OFG%_H', 'O3P%_H', 'OORB_H', 'OTRB_H', 'OTOV_H', 'OPTS_H']].copy()
        # features['OverUnder'] = (features['OverUnder']*10)
        # features['OPTS_V'] = (features['OPTS_V']*2)
        # features['OPTS_H'] = (features['OPTS_H']*2)

        
        labels = self.game_team_data['OUresult']
        self.X_train,self.X_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.10)
        self.X_train.reset_index(drop=True, inplace=True) #need to reset the indexes for consistent referencing
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)




    def getNN(self, points): #takes in a set of points. Gets the nearest neighbors for each point, and the classifies based on those neighbors and a threshold.
                            #returns a dictionary of where the keys are indices of the points, and the values are the classification for those points.
        OU_classifications = {}
        # print(self.model.kneighbors(points))
        distances, indices = self.model.kneighbors(points)
        # print(distances)
        for i in range(len(indices)):
            #goes through each game in test data
            o_count = 0
            u_count = 0
            for neighbor in indices[i]:
                #for each game in test data, goes through its K (10) nearest neighbors
                if self.y_train[neighbor] == "O":
                    o_count += 1
                else:
                    u_count += 1

            tot = o_count + u_count
            if o_count/tot > self.threshhold:
                OU_classifications[i] = "O"
            elif u_count/tot > self.threshhold:
                OU_classifications[i] = "U"
            else:
                OU_classifications[i] = "?"

        return OU_classifications
    
    def getNN_sim(self, points): #takes in a set of points. Gets the nearest neighbors for each point, and the classifies based on those neighbors and a threshold.
                            #returns a dictionary of where the keys are indices of the points, and the values are the classification for those points.
        OU_classifications = {}
        # print(self.model.kneighbors(points))
        distances, indices = self.model.kneighbors(points)
        # print(distances)
        for i in range(len(indices)):
            #goes through each game in test data
            o_count = 0
            u_count = 0
            for neighbor in indices[i]:
                #for each game in test data, goes through its K (10) nearest neighbors
                if self.y_train[neighbor] == "O":
                    o_count += 1
                else:
                    u_count += 1

            tot = o_count + u_count
            if o_count > u_count:
                OU_classifications[i] = ["O", o_count/tot]
            elif u_count > o_count:
                OU_classifications[i] = ["U", u_count/tot]
            else:
                OU_classifications[i] = ["?", 0]

        return OU_classifications


    def getNN_dist(self, points, distThresh): #takes in a set of points. Gets the nearest neighbors for each point, and the classifies based on those neighbors and a threshold.
                            #returns a dictionary of where the keys are indices of the points, and the values are the classification for those points.
        OU_classifications = {}
        distances, indices = self.model.kneighbors(points)
        for i in range(len(indices)):
            LST = [distances[i], indices[i]]
            #goes through each game in test data
            o_count = 0
            u_count = 0
            for x in range(len(LST[0])):
                #for each game in test data, goes through its K (10) nearest neighbors
                neighbor = LST[1][x]
                #gives you neighbor game index
                distance = LST[0][x]
                #gives you neighbor distance
                if self.y_train[neighbor] == "O" and distance < distThresh:
                    o_count += 1
                    # print("o hit\n")
                elif self.y_train[neighbor] == "U" and distance < distThresh:
                    u_count += 1
                    # print("u hit\n")

            tot = o_count + u_count
            if o_count > u_count:
                OU_classifications[i] = "O"
            elif u_count > o_count:
                OU_classifications[i] = "U"
            else:
                OU_classifications[i] = "?"
        return OU_classifications


    def getNN_hybrid(self, points, distThresh, subThresh): #takes in a set of points. Gets the nearest neighbors for each point, and the classifies based on those neighbors and a threshold.
                            #returns a dictionary of where the keys are indices of the points, and the values are the classification for those points.
        OU_classifications = {}
        distances, indices = self.model.kneighbors(points)
        for i in range(len(indices)):
            LST = [distances[i], indices[i]]
            #goes through each game in test data
            o_count = 0
            u_count = 0
            for x in range(len(LST[0])):
                #for each game in test data, goes through its K (10) nearest neighbors
                neighbor = LST[1][x]
                #gives you neighbor game index
                distance = LST[0][x]
                #gives you neighbor distance
                if self.y_train[neighbor] == "O" and distance < distThresh:
                    o_count += 1
                    # print("o hit\n")
                elif self.y_train[neighbor] == "U" and distance < distThresh:
                    u_count += 1
                    # print("u hit\n")

            tot = o_count + u_count
            if o_count - u_count > subThresh:
                OU_classifications[i] = "O"
            elif u_count - o_count > subThresh:
                OU_classifications[i] = "U"
            else:
                OU_classifications[i] = "?"
        return OU_classifications

    def set_up_custom(self, model_type, distThresh = 0.75, ratThresh = 0.7, subThresh = 2):
        self.get_split()
        self.model = NearestNeighbors(n_neighbors=self.num_neighbors, algorithm="ball_tree").fit(self.X_train)
        if model_type == 'custom_1':
            self.threshhold = ratThresh
            self.answer_dict = self. getNN(self.X_test)
        elif model_type == 'custom_2':
            self.answer_dict = self. getNN_dist(self.X_test, distThresh)
        elif model_type == 'custom_3':
            self.answer_dict = self.getNN_hybrid(self.X_test, distThresh, subThresh)
        elif model_type == 'custom_sim':
            self.answer_dict = self.getNN_sim(self.X_test)
            return self.answer_dict
        # print(self.answer_dict)
        LetsGo = 0
        Shucks = 0
        Qs = 0
        for gameDex in self.answer_dict:
            if self.answer_dict[gameDex] == '?':
                Qs += 1
            elif self.answer_dict[gameDex] == self.y_test[gameDex]:
                LetsGo += 1
            elif self.answer_dict[gameDex] != self.y_test[gameDex]:
                Shucks += 1
        if LetsGo+Shucks == 0:
            return [100000000000000000, 10000000000000]
        return [LetsGo/(LetsGo+Shucks), LetsGo+Shucks]



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
    

    def testRatio(self, trials = 1000, model_type = 'custom_1'):
        lst = []
        for x in range(trials):
            lst.append(self.set_up_custom(model_type)[0])
        return mean(lst)

    def testExplore_dist(self, trials = 10):
        dict = {}
        dist = 1.2
        for x in range(trials):
            tempLst = []
            avgBets = []
            for y in range(25):
                tempLst.append(self.set_up_custom('custom_2', distThresh = dist)[0])
                avgBets.append(self.set_up_custom('custom_2', distThresh = dist)[1])
            dict[round(dist, 2)] = [round(mean(tempLst), 2), round(mean(avgBets), 2)]
            dist = dist - 0.1
        return dict

    def testExplore_rat(self, trials = 10):
        dict = {}
        rat = 0.5
        for x in range(trials):
            tempLst = []
            avgBets = []
            for y in range(20):
                print()
                result = (self.set_up_custom('custom_1', ratThresh = rat)[0])
                print(result)
                tempLst.append(result)
                avgBets.append(self.set_up_custom('custom_1', ratThresh = rat)[1])
            dict[round(rat, 2)] = [round(mean(tempLst), 2), round(mean(avgBets), 2)]
            rat = rat + 0.05
        return dict

    
    def simDict(self):
        #returns in form of [prediction, ratio, outcome]
        newDict = {}
        OU_classifications = self.set_up_custom(model_type='custom_sim')
        print(OU_classifications)
        for key in OU_classifications:
            actual = self.y_test[key]
            lst = OU_classifications[key]
            lst.append(actual)
            newDict[key] = lst
        return newDict
    
    # def simOne(self):





