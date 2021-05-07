import sys
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from utils import *

def approach():
    args = sys.argv

    data_path = args[1]
    score_path = args[2]
    approach_name = args[3]
    drop_months_end = int(args[4])
    num_test_commits = int(args[5])

    projects = load_all_projects(path=data_path)

    for project in projects:
        print(project)

        train_df, test_df = prepare_within_project_data(projects[project], drop_months_end=drop_months_end, num_test_commits=num_test_commits)

        #########################################
        # Build Classifier                      #
        # (should be adopted for your approach) #
        #########################################

        y_test = test_df['is_inducing']
        y_pred = y_test.copy().values
        y_pred.fill(0)   

        ######################################################
        # DO NOT TOUCH FROM HERE                             #
        # This is where the scores are calculated and stored #
        ######################################################

        scores = score_model(test_df, y_pred)
        print_summary(train_df, test_df, scores)
        write_scores(score_path, approach_name, project, scores)

        
if __name__ == '__main__':
    approach()