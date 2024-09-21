# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:36:01 2024
@author: aaami
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import random
import csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
global labels
labels=['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 'anisic',
                                     'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'bergamot',
                                     'berry', 'bitter', 'black currant', 'brandy', 'burnt', 'buttery', 'cabbage',
                                     'camphoreous', 'caramellic', 'cedar', 'celery', 'chamomile', 'cheesy',
                                     'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean', 'clove', 'cocoa',
                                     'coconut', 'coffee', 'cognac', 'cooked', 'cooling', 'cortex', 'coumarinic',
                                     'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty',
                                     'fermented', 'fishy', 'floral', 'fresh', 'fruit skin', 'fruity', 'garlic',
                                     'gassy', 'geranium', 'grape', 'grapefruit', 'grassy', 'green', 'hawthorn',
                                     'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth', 'jasmin', 'juicy',
                                     'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery', 'lemon', 'lily',
                                     'malty', 'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'muguet',
                                     'mushroom', 'musk', 'musty', 'natural', 'nutty', 'odorless', 'oily', 'onion',
                                     'orange', 'orangeflower', 'orris', 'ozone', 'peach', 'pear', 'phenolic',
                                     'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'powdery', 'pungent',
                                     'radish', 'raspberry', 'ripe', 'roasted', 'rose', 'rummy', 'sandalwood',
                                     'savory', 'sharp', 'smoky', 'soapy', 'solvent', 'sour', 'spicy', 'strawberry',
                                     'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic', 'tobacco', 'tomato',
                                     'tropical', 'vanilla', 'vegetable', 'vetiver', 'violet', 'warm', 'waxy',
                                     'weedy', 'winey', 'woody']


def create_df_for_boruta(data):  # for boruta has to have a data frame with features names
    new_data = pd.DataFrame(data,
                            columns=labels)
    return new_data


def Algorithm_train(data):
    acurancy = [] #array that contains algorithm mane and its acurancy



    random.shuffle(data)  # Shuffle the data_vector randomly
    split_index = len(data) // 10  # Calculate the split index in order to split the the to train test 90:10
    test_group = data[:split_index]  # test group after splitting
    train_group = data[split_index:]  # train group after splitting
    # remove the unnececery data,(Delete the first two columns that contains smell chimical writing and true label name)
    train_group_for_KF = train_group.copy() #copy of train group to delete and still have the original data
    test_group_for_KF = test_group.copy() #copy of test group to delete and still have the original data
    train_group_for_KF = [row[3:] for row in train_group_for_KF]
    test_group_for_KF = [row[3:] for row in test_group_for_KF]
    # adding true label into new array
    train_true_label = []
    test_true_label = []

    for smell in train_group:
        train_true_label.append(smell[1])

    for smell in test_group:
        test_true_label.append(smell[1])
    #-----------------------------------------5 fold cross validation --------------------------------

    # choosing the model for 5 fold
    model = KNeighborsClassifier()
    #run 5 fold CV
    acurancy_for_cross_validation = cross_val_score(model, train_group_for_KF, train_true_label, scoring='accuracy', cv=5)

    print("\nCross-validation scores:", acurancy_for_cross_validation)
    print("Mean cross-validation score:", acurancy_for_cross_validation.mean())

    # ------------------------------------------------boruta------------------------------------------
    # choosing random forrest algorithm for boruta
    model_for_boruta = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

    #parameters of boruta
    feat_selector = BorutaPy(
        verbose=2,
        estimator=model_for_boruta,
        n_estimators='auto',
        max_iter=10  # number of iterations to perform
    )

    # train Boruta
    feat_selector.fit(np.array(train_group_for_KF, dtype=int), np.array(train_true_label, dtype=int))#changing the data to np array for running the algorithm
    df_for_boruta = create_df_for_boruta(train_group_for_KF)#creates the data frame for choosing best params
    df_for_boruta_test = create_df_for_boruta(test_group_for_KF)# creates the data frame for deleting the "unneccery data" for testing later

    # print the best params to console
    print("\n------Support and Ranking for each feature------")
    for i in range(len(feat_selector.support_)):
        if feat_selector.support_[i]:
            print("Passes the test: ", df_for_boruta.columns[i],
                  " - Ranking: ", feat_selector.ranking_[i])
    global selected_features
    selected_features = feat_selector.support_
    data_filtered = df_for_boruta.loc[:, selected_features] #filter the train group and stay only with the best params
    test_filtered = df_for_boruta_test.loc[:, selected_features] # #filter the test group and stay only with the best params


    # --------------------KNN-----------------------------------------

    # run KNN
    global knn
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train KNN with train group
    knn.fit(train_group_for_KF, train_true_label)

    # predict for test
    y_pred = knn.predict(test_group_for_KF)

    # culc acurancy for the KNN
    accuracy_for_knn = accuracy_score(test_true_label, y_pred)
    print(f"Accuracy for KNN: {100 * accuracy_for_knn:.2f}%")#print the result to consol
    acurancy.append(["KNN", accuracy_for_knn]) #add the result to the results array
    # ---------------------------knn with bo--------------------------------
    # run KNN with boruta
    global knn_bo
    knn_bo = KNeighborsClassifier(n_neighbors=3) # create new objest of KNN clasiifier for boruta

    # Train KNN wuth boruta with filtered data
    knn_bo.fit(data_filtered, train_true_label)

    # predict test group filtered
    y_pred_bo = knn_bo.predict(test_filtered)

    # culc acurancy for the KNN with boruta
    accuracy_for_knn_bo = accuracy_score(test_true_label, y_pred_bo)
    print(f"Accuracy for KNN with boruta: {accuracy_for_knn_bo * 100:.2f}%")#print the result to console
    acurancy.append(["KNN_bo", accuracy_for_knn_bo])#add the result to the results array

    # -----------------------------------------Random forest---------------------
    #run random forrest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)#create object of random forest classifier

    # Train random forrest
    rf.fit(train_group_for_KF, train_true_label)
    #  predict test group using random forrest
    y_pred_rf = rf.predict(test_group_for_KF)

    # culc acurancy for the RF
    accuracy_for_rf = accuracy_score(test_true_label, y_pred_rf)
    print(f"Accuracy for rf: {accuracy_for_rf * 100:.2f}%")#print the result to console
    acurancy.append(["Rf", accuracy_for_rf])#add the result to the results array
    # -----------------------------rf with boruta---------------------
    # run random forrest with boruta
    global rf_bo
    rf_bo = RandomForestClassifier(n_estimators=50, random_state=42)#create new object of random forest classifier for boruta

    # Train random forrest with filtered data
    rf_bo.fit(data_filtered, train_true_label)
    # predict test group using random forrest with filtered data
    y_pred_rf_bo = rf_bo.predict(test_filtered)

    #  culc acurancy for the RF with boruta
    accuracy_for_rf_bo = accuracy_score(test_true_label, y_pred_rf_bo)
    print(f"Accuracy for rf with bo: {accuracy_for_rf_bo * 100:.2f} %")#print the result to console
    acurancy.append(["Rf bo", accuracy_for_rf_bo])#add the result to the results array

    # -----------------------------------ADAboost----------------------------
    global adaboost
    #  run adaboost
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)#create object of adaboost classifier

    # Train adaboost
    adaboost.fit(train_group_for_KF, train_true_label)

    #  predict test group using adaboost
    y_pred_ada = adaboost.predict(test_group_for_KF)

    #culc acurancy for adaboost
    accuracy_adaboost = accuracy_score(test_true_label, y_pred_ada)
    print(f"Accuracy for adaboost: {accuracy_adaboost * 100:.2f}%")#print the result to console
    acurancy.append(["AdaBoost", accuracy_adaboost])#add the result to the results array
    # ---------------------------------Adaboost with boruta----------------------------
    global adaboost_bo
    # run adaboost with boruta
    adaboost_bo = AdaBoostClassifier(n_estimators=50, random_state=42)#create new object odadaboost classifier for boruta

    # Train adaboost using filtered data
    adaboost_bo.fit(data_filtered, train_true_label)

    #  predict test group using adaboost with filtered data
    y_pred_ada_bo = adaboost_bo.predict(test_filtered)

    #culc acurancy for adaboost
    accuracy_adaboost_bo = accuracy_score(test_true_label, y_pred_ada_bo)
    print(f"Accuracy for adaboost with bo: {accuracy_adaboost_bo * 100:.2f}%")#print the result to console
    acurancy.append(["AdaBoost_bo", accuracy_adaboost_bo])#add the result to the results array

    #-----------------------------------xgboost--------------------------
    global xgboost
    global label_encoder
    # run xgboost
    xgboost = xgb.XGBClassifier(use_label_encoder=True, eval_metric='logloss') # Create object of XGBoost Classifier
    label_encoder = LabelEncoder()#encode the true label in order them to be in a increasing series e.g :0 ,1, 2, 3,......
    true_label_encoded = label_encoder.fit_transform(train_true_label)  # Encode the labels
   # Train xgboost using encoded true label
    xgboost.fit(train_group_for_KF, true_label_encoded)

    #predict the result (retult in encoded true labels)
    y_pred_xg_encoded = xgboost.predict(test_group_for_KF)
    y_pred_xg_original = label_encoder.inverse_transform(y_pred_xg_encoded)  # Decode predictions labels
    # culc acurancy for xgboost
    accuracy_xgboost = accuracy_score(test_true_label, y_pred_xg_original)
    print(f"Accuracy XGBoost: {accuracy_xgboost * 100:.2f}%")#print the result to console
    acurancy.append(["xgboost", accuracy_xgboost]) #add the result to the results array
    #------------------------------------------xgboost bo------------------------------------
    global xgboost_bo
    # run xgboost with boruta
    xgboost_bo = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')#create new object xgboost classifier for boruta

    # Train xgboost using encoded true label and filtered data
    xgboost_bo.fit(data_filtered, true_label_encoded)
    # predict the result (retult in encoded true labels)
    y_pred_xg_bo_encoded = xgboost_bo.predict(test_filtered)
    y_pred_xg_bo_original = label_encoder.inverse_transform(y_pred_xg_bo_encoded)  # Decode predictions labels
    # Calculate accuracy for xgboost using boruta
    accuracy_xgboost = accuracy_score(test_true_label, y_pred_xg_bo_original)
    print(f"Accuracy  bo: {accuracy_xgboost * 100:.2f}%")#print the result to console
    acurancy.append(["xgboost bo", accuracy_xgboost])#add the result to the results array
    # find the best model by acurancy
    max_value = 0
    for i in range(len(acurancy)):
        if acurancy[i][1] > max_value:
            max_value = acurancy[i][1]
            max_index = i
    #write best model and date of training to a file
    with open("traindata.txt", "w") as file:

        file.write(acurancy[max_index][0] + "\n") #write the best model name
        file.write(datetime.now().strftime("%Y-%m-%d"))#write the dated of training
    return acurancy[max_index][0]# return best model


def main_train(data_to_pred):
    # -----------------------read the data---------------------------------------
    df = pd.read_csv('data_for_proj.csv')
    # Convert the Dataframe to a 2D array
    data = df.values.tolist()
    data2 = data.copy()#create copy to rearrange the data
    true_label = []# create an array for true label
    # remove the unnececery data,(Delete the first two columns that contains smell chimical writing and true label name)
    data2_fixed = [row[3:] for row in data2]
    for smell in data:
        true_label.append(smell[1])
    #open the train data file
    with open("traindata.txt", "r") as file:
        # Read all the lines(2 lines)
        lines = file.readlines()

    #get the last time trained and change it to a time object
    last_time_trained = datetime.strptime(lines[1], "%Y-%m-%d")
    best_model = lines[0]
    #check if traied in the past week
    if datetime.now() - last_time_trained >= timedelta(weeks=1):
        best_model = Algorithm_train(data)#run the train algorithm

    print("the best model is:", best_model)#print best model to console
    #find which model is the best and run it
    if best_model == "KNN" or best_model == "KNN\n":
        knn = KNeighborsClassifier(n_neighbors=3)

        # Train the classifier
        knn.fit(data2_fixed, true_label)
        prediction = knn.predict(data_to_pred)
        print("the prediction is:", prediction)
    elif best_model == "KNN_bo" or best_model == "KNN_bo\n":
        filtered_data = create_df_for_boruta(data_to_pred)
        filtered_data = filtered_data.loc[:, selected_features]
        prediction = knn_bo.predict(filtered_data)
        print("the prediction is:", prediction)
    elif best_model== "Rf" or best_model=="Rf\n":
        rf_final = RandomForestClassifier(n_estimators=50, random_state=42)
        # Train the classifier
        rf_final.fit(data2_fixed, true_label)
        prediction = rf_final.predict(data_to_pred)
        print("the prediction is:", prediction)
    elif  best_model == "Rf bo" or best_model == "Rf bo\n":
        filtered_data = create_df_for_boruta(data_to_pred)
        filtered_data = filtered_data.loc[:, selected_features]
        prediction = rf_bo.predict(filtered_data)
        print("the prediction is:", prediction)
    elif best_model == "AdaBoost" or best_model == "AdaBoost\n":
        prediction = adaboost.predict(data_to_pred)
        print("the prediction is:", prediction)
    elif best_model == "AdaBoost_bo" or best_model == "AdaBoost_bo\n":
        filtered_data = create_df_for_boruta(data_to_pred)
        filtered_data = filtered_data.loc[:, selected_features]
        prediction = adaboost_bo.predict(filtered_data)
        print("the prediction is:", prediction)
    elif best_model == "xgboost\n":
        prediction_for_xg = xgboost_bo.predict(data_to_pred)
        prediction = label_encoder.inverse_transform(prediction_for_xg)
        print("the prediction is:", prediction)
    elif best_model == "xgboost bo\n":
        filtered_data = create_df_for_boruta(data_to_pred)
        filtered_data = filtered_data.loc[:, selected_features]
        prediction_for_xg = xgboost_bo.predict(filtered_data)
        prediction=label_encoder.inverse_transform(prediction_for_xg)
        print("the prediction is:", prediction)

    data_true_label=[] #array to save the true label of all the csv to pred
    num_to_text_true_label = []#creates and 2D array to save true label number and text
    for smell in data:
        num_to_text_true_label.append([smell[1], smell[2]])
    # for each smell find the true label text
    for pred in prediction:

        for i in range(len(num_to_text_true_label)):
            if pred == num_to_text_true_label[i][0]:
                data_true_label.append(num_to_text_true_label[i][1])
                break
    # --------------------add the new smeel to data -------------------------------
    # open the dataset file to add the new smells
    os.chmod("data_for_proj.csv", 0o666) #change permission for writing
    #for each smell create an fixed aray made from true label number true label text and smell featuers
    for j in range(len(prediction)):
        add_data_arr = [" ", int(prediction[j]), data_true_label[j]]
        for i in range(len(data_to_pred[j])):
            add_data_arr.append(int(data_to_pred[j][i]))
        # Open the csv file in append mode
        with open("data_for_proj.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            # Append the data to the CSV file
            writer.writerow(add_data_arr)
    print("added to data seccesfuly")# print to console
    os.chmod("data_for_proj.csv", 0o644)  # terurn to normal permissions
    return data_true_label #return true label array



def find_data_per_smell(smell:str,data):
    apeareance=0 #variable count all apreance of smell contains smell
    just_name=0 # variable count all apreance of smell contains smell but smell feature is 0

    true_index = labels.index(smell) # find index of feature

    for odor in data: #run for each row in data

        if smell in odor[2]: # check if smell is part of the smell name
            apeareance+=1

            if  (odor[true_index+3]==0):# check if smell feature is 0
                just_name+=1
    return [smell,just_name,apeareance]


def data_for_bar_chart(data,smell_name):
    myarr=[]

    ingridients=smell_name.split(";")#split the long name to parts
    for ingridient in ingridients:
        myarr.append(data[labels.index(ingridient)])#add value of feature in smell sent to check
    return myarr
def count_unique(smell_name):
    #reading the original data
    df = pd.read_csv('data_for_proj.csv')
    # Convert the Dataframe to a 2D array
    data = df.values.tolist()
    data_for_unique=[row[1:] for row in data]#ignore first column
    arr_unique=[]
    for smell in data_for_unique:
        if smell[1]==smell_name:#put all the same smell in an arr
            arr_unique.append(smell)
    unique_items = []
    for item in arr_unique:
        if item not in unique_items:
            unique_items.append(item)#add only unique items

    return len(unique_items)#return len of unique




