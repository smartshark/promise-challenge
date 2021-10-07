import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale

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

    for project_name in list_all_projects(path=data_path):
        print(project_name)
        data = load_project(path=data_path, project_name=project_name)

        train_df, test_df = prepare_within_project_data(data, drop_months_end=drop_months_end,
                                                        num_test_commits=num_test_commits)

        train_df = train_df.sample(frac=0.2)

        n_rows = train_df.shape[0]
        n_columns = train_df.shape[1]

        # train_df = scale(train_df)

        X_train = train_df[ALL_FEATURES].values
        X_test = test_df[ALL_FEATURES].values

        # binary labels are in the column 'is_inducing'
        y_train = train_df['is_inducing']
        y_test = test_df['is_inducing']

        # we recommend using a fixed random seed for reproducibility, but this is up to you
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)

        # X_train, y_train = scale(X_train), scale(y_train)

        # we resample with SMOTE and build a random forest for our baseline
        X_res, y_res = SMOTE(random_state=RANDOM_SEED).fit_resample(X_train, y_train)

        # X_res, y_res = scale(X_res), scale(y_res)

        rf = LogisticRegression(random_state=RANDOM_SEED, C=0.25)
        rf.fit(X_res, y_res)
        y_pred = rf.predict(X_test)

        dump(rf, 'logistic_regression_c_025_2.joblib')

        scores = score_model(test_df, y_pred)
        print_summary(train_df, test_df, scores)
        write_scores(score_path, approach_name, project_name, scores)

        
if __name__ == '__main__':
    approach()