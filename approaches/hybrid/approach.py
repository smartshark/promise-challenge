import sys
import numpy as np
import pandas as pd
import csv

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from joblib import dump

from utils import *

def approach():
    args = sys.argv

    data_path = args[1]
    score_path = args[2]
    approach_name = args[3]
    drop_months_end = int(args[4])
    num_test_commits = int(args[5])

    ######################################
    # Loop for within project prediction #
    # Loads only one project at a time   #
    ######################################

    for project_name in list_all_projects(path=data_path):
        print(project_name)
        data = load_project(path=data_path, project_name=project_name)

        train_df, test_df = prepare_within_project_data(data, drop_months_end=drop_months_end, num_test_commits=num_test_commits)

        #########################################
        # Build Classifier                      #
        # (should be adopted for your approach) #
        #########################################

        # extract columns with feature values from available data
        # we prepared the following feature lists for your convenience:
        #
        # ALL_FEATURES
        # STATIC_FEATURES
        # STATIC_FILE_FEATURES
        # STATIC_CLASS_FEATURES
        # STATIC_INTERFACE_FEATURES
        # STATIC_ENUM_FEATURES
        # STATIC_METHOD_FEATURES
        # FGJIT_FEATURES
        # JIT_FEATURES
        # WD_FEATURES
        # PMD_FEATURES
        #
        # please check the documentation to see which features are included in each list
        # https://github.com/smartshark/promise-challenge/blob/main/dataset.md
        # 
        # we use all available features for our baseline

        train_df = train_df.sample(frac=0.2)

        with open('200_selected_features.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            feature_names = next(spamreader)

        train_df_new = pd.concat([train_df[feat] for feat in feature_names], axis=1)
        test_df_new = pd.concat([test_df[feat] for feat in feature_names], axis=1)

        X_train = train_df_new.values
        X_test = test_df_new.values

        # binary labels are in the column 'is_inducing'
        y_train = train_df['is_inducing']
        y_test = test_df['is_inducing']

        # we recommend using a fixed random seed for reproducibility, but this is up to you
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)

        # we resample with SMOTE and build a random forest for our baseline
        X_res, y_res = SMOTE(random_state=RANDOM_SEED).fit_resample(X_train, y_train)
        # TODO: for
        rf = RandomForestClassifier(random_state=RANDOM_SEED, oob_score=True, n_estimators=50)
        rf.fit(X_res, y_res)

        importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

        y_pred = rf.predict_proba(X_test)
        # TODO: end of loop

        dump(rf, '200_important_features_rf.joblib')

        ######################################################
        # DO NOT TOUCH FROM HERE                             #
        # This is where the scores are calculated and stored #
        ######################################################

        scores = score_model(test_df, y_pred)
        print_summary(train_df, test_df, scores)
        write_scores(score_path, approach_name, project_name, scores)

        
if __name__ == '__main__':
    approach()