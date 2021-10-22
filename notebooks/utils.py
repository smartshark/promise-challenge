import csv
import os
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef


def bug_columns(df, label='induces'):
    """return all columns from the bug-matrix"""
    jlip = []
    for col in df.columns:
        if col.startswith('{}__'.format(label)):
            jlip.append(col)
    return jlip


def load_project(path, project_name):
    """load project from the supplied csv"""
    if not path.endswith('/') and len(path)>0:
        path += '/'
    df =  pd.read_csv(path+project_name+'.csv.gz')
    df['project'] = project_name
    df['committer_date'] = pd.to_datetime(df['committer_date'])
    return df


def load_all_projects(path):
    """loads all projects from a folder"""
    projects = {}
    for project_name in list_all_projects(path):
        projects[project_name] = load_project(path=path, project_name=project_name)
    return projects


def list_all_projects(path):
    """lists all projects from a folder"""
    project_names = []
    for file in os.listdir(path):
        if not os.path.isfile(os.path.join(path, file)):
            continue
        project_names.append(file.split('.')[0])
    return project_names


def last_commits(df, num_commits=500):
    """return last num_commits"""
    last_commits = []
    for c in df['commit'].unique()[-num_commits:]:  # order is preserved here (nice!)
        last_commits.append(c)
    return last_commits


def bugs_later_than(df, cutoff_date):
    """return columns from bug-matrix that are after a given cutoff date"""
    if not cutoff_date:
        raise Exception('please supply a cutoff date')

    remove = []
    for col in df.columns:
        if col.startswith('induces__'):
            label, issue_id, bugfix_commit, bugfix_date = col.split('__')
            if pd.to_datetime(bugfix_date, utc=True) > cutoff_date:
                remove.append(col)
    return remove


def prepare_within_project_data(test_project_df, drop_months_end=3, num_test_commits=250):
    """takes the data from a project and splits it into training and test data"""
    
    # drop end of project
    # there was no time for bug fixes, meaning the labels are not reliable
    latest_commit_date = test_project_df['committer_date'].max()
    cutoff_end = latest_commit_date-relativedelta(months=drop_months_end)
    test_project_df = test_project_df[test_project_df['committer_date']<cutoff_end]

    # use last 500 commits as test data
    lc = last_commits(test_project_df, num_commits=num_test_commits)
    test_df = test_project_df[test_project_df['commit'].isin(lc)].copy()
    train_df = test_project_df[~test_project_df['commit'].isin(lc)].copy()

    # drop all bugs that were fixed after the test period starts
    # this prevents a time travel information leak
    test_start_date = test_df['committer_date'].min()
    late_bugs = bugs_later_than(train_df, cutoff_date=test_start_date)
    train_df.drop(columns=late_bugs, inplace=True)

    # drop last three months of training data
    # there was no time for bug fixes, meaning the labels are not reliable
    # we use the start of the test data as reference
    cutoff_train = test_start_date - relativedelta(months=3)
    train_df = train_df[train_df['committer_date']<cutoff_train]

    # finally, we transform the detailed bug matrix into binary labels
    train_bugs = bug_columns(train_df)
    train_df['is_inducing'] = train_df[train_bugs].any(axis=1)
    test_bugs = bug_columns(test_df)
    test_df['is_inducing'] = test_df[test_bugs].any(axis=1)

    return train_df, test_df


def prepare_all_data(test_project_name, projects, drop_months_end=3, num_test_commits=250):
    """takes the data from the project and splits it into training and test data and also adds all data from other projects that are available"""

    test_project_df = projects[test_project_name]

    # drop end of project
    # there was no time for bug fixes, meaning the labels are not reliable
    latest_commit_date = test_project_df['committer_date'].max()
    cutoff_end = latest_commit_date-relativedelta(months=drop_months_end)
    test_project_df = test_project_df[test_project_df['committer_date']<cutoff_end]

    # use last 500 commits as test data
    lc = last_commits(test_project_df, num_commits=num_test_commits)
    test_df = test_project_df[test_project_df['commit'].isin(lc)].copy()

    # we compute binary labels for the test data
    test_bugs = bug_columns(test_df)
    test_df['is_inducing'] = test_df[test_bugs].any(axis=1)


    # now we prepare the training data
    # first, we take the commits prior to the test data from the target project
    train_df = test_project_df[~test_project_df['commit'].isin(lc)].copy()

    # we also drop all bugs, that were reported after the test period starts
    test_start_date = test_df['committer_date'].min()
    late_bugs = bugs_later_than(train_df, cutoff_date=test_start_date)
    train_df.drop(columns=late_bugs, inplace=True)

    # then we compute binary labels and drop the bug matrix completely
    train_bugs = bug_columns(train_df)
    train_df['is_inducing'] = train_df[train_bugs].any(axis=1)
    bug_matrix_cols = [col for col in train_df.columns if col.startswith('induces__')]
    train_df.drop(columns=bug_matrix_cols, inplace=True)

    # now we add all commits from other projects, prior to the cutoff date
    # the treatment of the data is the same
    for project in projects:
        if project==test_project_name:
            continue
        other_df = projects[project].copy()
        other_df = other_df[other_df['committer_date']<test_start_date]
        late_bugs = bugs_later_than(other_df, cutoff_date=test_start_date)
        other_df.drop(columns=late_bugs, inplace=True)
        train_bugs = bug_columns(other_df)
        other_df['is_inducing'] = other_df[train_bugs].any(axis=1)
        bug_matrix_cols = [col for col in other_df.columns if col.startswith('induces__')]
        other_df.drop(columns=bug_matrix_cols, inplace=True)
        train_df = train_df.append(other_df)

    # drop last three months of training data
    # there was no time for bug fixes, meaning the labels are not reliable
    # we use the start of the test data as reference
    cutoff_train = test_start_date - relativedelta(months=3)
    train_df = train_df[train_df['committer_date']<cutoff_train]

    return train_df, test_df


def lower_bound(test_df, predictions):
    """calculates the lower bound of the cost saving range"""
    bug_matrix_cols = [col for col in test_df.columns if col.startswith('induces__')]
    bug_matrix = test_df.loc[:,bug_matrix_cols]
    efforts = test_df['la']+test_df['ld']
    effort_true = efforts[predictions].sum()
    bugs_found = bug_matrix.sum().eq(bug_matrix[predictions].sum()).sum()
    return effort_true/bugs_found


def upper_bound(test_df, predictions):
    """calculates the upper bound of the cost saving range"""
    bug_matrix_cols = [col for col in test_df.columns if col.startswith('induces__')]
    bug_matrix = test_df.loc[:,bug_matrix_cols]
    efforts = test_df['la']+test_df['ld']
    effort_false = efforts[~predictions].sum()
    bugs_missed = len(bug_matrix.columns)-bug_matrix.sum().eq(bug_matrix[predictions].sum()).sum()
    return effort_false/bugs_missed


def costs(test_df, predictions, C):
    """calculates the costs given the cost of defects per line of code C"""
    bug_matrix_cols = [col for col in test_df.columns if col.startswith('induces__')]
    bug_matrix = test_df.loc[:,bug_matrix_cols]
    efforts = test_df['la']+test_df['ld']
    effort_true = efforts[predictions].sum()
    bugs_missed = len(bug_matrix.columns)-bug_matrix.sum().eq(bug_matrix[predictions].sum()).sum()
    return effort_true+C*bugs_missed


def score_model(test_df, y_pred):
    """calculates the scores for a model"""
    scores = {}
    scores['mcc'] = matthews_corrcoef(test_df['is_inducing'], y_pred)
    scores['c_lower'] = lower_bound(test_df, y_pred)
    scores['c_upper'] = upper_bound(test_df, y_pred)
    scores['cost_1000']  = costs(test_df, y_pred, 1000)
    scores['cost_10000'] = costs(test_df, y_pred, 10000)
    return scores


def print_summary(train_df, test_df, scores):
    """prints a summary of the data and scores"""
    print('train instances: {} ({} positive)'.format(len(train_df), sum(train_df['is_inducing'])))
    print('test instances:  {} ({} positive)'.format(len(test_df),  sum(test_df['is_inducing'])))
    
    for metric, score in scores.items():
        print(metric.ljust(16), score)
    print()

    
def write_scores(path, approach_name, project, scores):
    """writes the scores to a csv file"""
    if not path.endswith('/') and len(path)>0:
        path += '/'
    file_name = path+approach_name+'.csv'
    values = []
    values.append(project)
    for score_value in scores.values():
        values.append(score_value)

    write_header = True
    if os.path.exists(file_name) and os.path.getsize(file_name)>0:
        write_header = False

    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        if write_header:
            header = []
            header.append('project')
            for score_name in scores:
                header.append(score_name)
            writer.writerow(header)
        writer.writerow(values)

# Constants for feature sets

PMD_RULES = [{'type': 'Basic Rules', 'rule': 'Avoid Branching Statement As Last In Loop', 'abbrev': 'PMD_ABSALIL', 'severity': 'Major'}, {'type': 'Basic Rules', 'rule': 'Avoid Decimal Literals In Big Decimal Constructor', 'abbrev': 'PMD_ADLIBDC', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Avoid Multiple Unary Operators', 'abbrev': 'PMD_AMUO', 'severity': 'Major'}, {'type': 'Basic Rules', 'rule': 'Avoid Thread Group', 'abbrev': 'PMD_ATG', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Avoid Using Hard Coded IP', 'abbrev': 'PMD_AUHCIP', 'severity': 'Major'}, {'type': 'Basic Rules', 'rule': 'Avoid Using Octal Values', 'abbrev': 'PMD_AUOV', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Big Integer Instantiation', 'abbrev': 'PMD_BII', 'severity': 'Minor'}, {'type': 'Basic Rules', 'rule': 'Boolean Instantiation', 'abbrev': 'PMD_BI', 'severity': 'Minor'}, {'type': 'Basic Rules', 'rule': 'Broken Null Check', 'abbrev': 'PMD_BNC', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Check Result Set', 'abbrev': 'PMD_CRS', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Check Skip Result', 'abbrev': 'PMD_CSR', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Class Cast Exception With To Array', 'abbrev': 'PMD_CCEWTA', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Collapsible If Statements', 'abbrev': 'PMD_CIS', 'severity': 'Minor'}, {'type': 'Basic Rules', 'rule': 'Dont Call Thread Run', 'abbrev': 'PMD_DCTR', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Dont Use Float Type For Loop Indices', 'abbrev': 'PMD_DUFTFLI', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Double Checked Locking', 'abbrev': 'PMD_DCL', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Empty Catch Block', 'abbrev': 'PMD_ECB', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Empty Finally Block', 'abbrev': 'PMD_EFB', 'severity': 'Minor'}, {'type': 'Basic Rules', 'rule': 'Empty If Stmt', 'abbrev': 'PMD_EIS', 'severity': 'Major'}, {'type': 'Basic Rules', 'rule': 'Empty Statement Block', 'abbrev': 'PMD_EmSB', 'severity': 'Minor'}, {'type': 'Basic Rules', 'rule': 'Empty Statement Not In Loop', 'abbrev': 'PMD_ESNIL', 'severity': 'Minor'}, {'type': 'Basic Rules', 'rule': 'Empty Static Initializer', 'abbrev': 'PMD_ESI', 'severity': 'Minor'}, {'type': 'Basic Rules', 'rule': 'Empty Switch Statements', 'abbrev': 'PMD_ESS', 'severity': 'Major'}, {'type': 'Basic Rules', 'rule': 'Empty Synchronized Block', 'abbrev': 'PMD_ESB', 'severity': 'Major'}, {'type': 'Basic Rules', 'rule': 'Empty Try Block', 'abbrev': 'PMD_ETB', 'severity': 'Major'}, {'type': 'Basic Rules', 'rule': 'Empty While Stmt', 'abbrev': 'PMD_EWS', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Extends Object', 'abbrev': 'PMD_EO', 'severity': 'Minor'}, {'type': 'Basic Rules', 'rule': 'For Loop Should Be While Loop', 'abbrev': 'PMD_FLSBWL', 'severity': 'Minor'}, {'type': 'Basic Rules', 'rule': 'Jumbled Incrementer', 'abbrev': 'PMD_JI', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Misplaced Null Check', 'abbrev': 'PMD_MNC', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Override Both Equals And Hashcode', 'abbrev': 'PMD_OBEAH', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Return From Finally Block', 'abbrev': 'PMD_RFFB', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Unconditional If Statement', 'abbrev': 'PMD_UIS', 'severity': 'Major'}, {'type': 'Basic Rules', 'rule': 'Unnecessary Conversion Temporary', 'abbrev': 'PMD_UCT', 'severity': 'Minor'}, {'type': 'Basic Rules', 'rule': 'Unused Null Check In Equals', 'abbrev': 'PMD_UNCIE', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Useless Operation On Immutable', 'abbrev': 'PMD_UOOI', 'severity': 'Critical'}, {'type': 'Basic Rules', 'rule': 'Useless Overriding Method', 'abbrev': 'PMD_UOM', 'severity': 'Minor'}, {'type': 'Brace Rules', 'rule': 'For Loops Must Use Braces', 'abbrev': 'PMD_FLMUB', 'severity': 'Minor'}, {'type': 'Brace Rules', 'rule': 'If Else Stmts Must Use Braces', 'abbrev': 'PMD_IESMUB', 'severity': 'Minor'}, {'type': 'Brace Rules', 'rule': 'If Stmts Must Use Braces', 'abbrev': 'PMD_ISMUB', 'severity': 'Minor'}, {'type': 'Brace Rules', 'rule': 'While Loops Must Use Braces', 'abbrev': 'PMD_WLMUB', 'severity': 'Minor'}, {'type': 'Clone Implementation Rules', 'rule': 'Clone Throws Clone Not Supported Exception', 'abbrev': 'PMD_CTCNSE', 'severity': 'Major'}, {'type': 'Clone Implementation Rules', 'rule': 'Proper Clone Implementation', 'abbrev': 'PMD_PCI', 'severity': 'Critical'}, {'type': 'Controversial Rules', 'rule': 'Assignment In Operand', 'abbrev': 'PMD_AIO', 'severity': 'Minor'}, {'type': 'Controversial Rules', 'rule': 'Avoid Accessibility Alteration', 'abbrev': 'PMD_AAA', 'severity': 'Major'}, {'type': 'Controversial Rules', 'rule': 'Avoid Prefixing Method Parameters', 'abbrev': 'PMD_APMP', 'severity': 'Minor'}, {'type': 'Controversial Rules', 'rule': 'Avoid Using Native Code', 'abbrev': 'PMD_AUNC', 'severity': 'Major'}, {'type': 'Controversial Rules', 'rule': 'Default Package', 'abbrev': 'PMD_DP', 'severity': 'Minor'}, {'type': 'Controversial Rules', 'rule': 'Do Not Call Garbage Collection Explicitly', 'abbrev': 'PMD_DNCGCE', 'severity': 'Major'}, {'type': 'Controversial Rules', 'rule': 'Dont Import Sun', 'abbrev': 'PMD_DIS', 'severity': 'Major'}, {'type': 'Controversial Rules', 'rule': 'One Declaration Per Line', 'abbrev': 'PMD_ODPL', 'severity': 'Minor'}, {'type': 'Controversial Rules', 'rule': 'Suspicious Octal Escape', 'abbrev': 'PMD_SOE', 'severity': 'Major'}, {'type': 'Controversial Rules', 'rule': 'Unnecessary Constructor', 'abbrev': 'PMD_UC', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Abstract Class Without Abstract Method', 'abbrev': 'PMD_ACWAM', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Abstract Class Without Any Method', 'abbrev': 'PMD_AbCWAM', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Assignment To Non Final Static', 'abbrev': 'PMD_ATNFS', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Avoid Constants Interface', 'abbrev': 'PMD_ACI', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Avoid Instanceof Checks In Catch Clause', 'abbrev': 'PMD_AICICC', 'severity': 'Major'}, {'type': 'Design Rules', 'rule': 'Avoid Protected Field In Final Class', 'abbrev': 'PMD_APFIFC', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Avoid Protected Method In Final Class Not Extending', 'abbrev': 'PMD_APMIFCNE', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Avoid Reassigning Parameters', 'abbrev': 'PMD_ARP', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Avoid Synchronized At Method Level', 'abbrev': 'PMD_ASAML', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Bad Comparison', 'abbrev': 'PMD_BC', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Class With Only Private Constructors Should Be Final', 'abbrev': 'PMD_CWOPCSBF', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Close Resource', 'abbrev': 'PMD_ClR', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Constructor Calls Overridable Method', 'abbrev': 'PMD_CCOM', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Default Label Not Last In Switch Stmt', 'abbrev': 'PMD_DLNLISS', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Empty Method In Abstract Class Should Be Abstract', 'abbrev': 'PMD_EMIACSBA', 'severity': 'Major'}, {'type': 'Design Rules', 'rule': 'Equals Null', 'abbrev': 'PMD_EN', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Field Declarations Should Be At Start Of Class', 'abbrev': 'PMD_FDSBASOC', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Final Field Could Be Static', 'abbrev': 'PMD_FFCBS', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Idempotent Operations', 'abbrev': 'PMD_IO', 'severity': 'Major'}, {'type': 'Design Rules', 'rule': 'Immutable Field', 'abbrev': 'PMD_IF', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Instantiation To Get Class', 'abbrev': 'PMD_ITGC', 'severity': 'Major'}, {'type': 'Design Rules', 'rule': 'Logic Inversion', 'abbrev': 'PMD_LI', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Missing Break In Switch', 'abbrev': 'PMD_MBIS', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Missing Static Method In Non Instantiatable Class', 'abbrev': 'PMD_MSMINIC', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Non Case Label In Switch Statement', 'abbrev': 'PMD_NCLISS', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Non Static Initializer', 'abbrev': 'PMD_NSI', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Non Thread Safe Singleton', 'abbrev': 'PMD_NTSS', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Optimizable To Array Call', 'abbrev': 'PMD_OTAC', 'severity': 'Major'}, {'type': 'Design Rules', 'rule': 'Position Literals First In Case Insensitive Comparisons', 'abbrev': 'PMD_PLFICIC', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Position Literals First In Comparisons', 'abbrev': 'PMD_PLFIC', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Preserve Stack Trace', 'abbrev': 'PMD_PST', 'severity': 'Major'}, {'type': 'Design Rules', 'rule': 'Return Empty Array Rather Than Null', 'abbrev': 'PMD_REARTN', 'severity': 'Major'}, {'type': 'Design Rules', 'rule': 'Simple Date Format Needs Locale', 'abbrev': 'PMD_SDFNL', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Simplify Boolean Expressions', 'abbrev': 'PMD_SBE', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Simplify Boolean Returns', 'abbrev': 'PMD_SBR', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Simplify Conditional', 'abbrev': 'PMD_SC', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Singular Field', 'abbrev': 'PMD_SF', 'severity': 'Major'}, {'type': 'Design Rules', 'rule': 'Switch Stmts Should Have Default', 'abbrev': 'PMD_SSSHD', 'severity': 'Major'}, {'type': 'Design Rules', 'rule': 'Too Few Branches For ASwitch Statement', 'abbrev': 'PMD_TFBFASS', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Uncommented Empty Constructor', 'abbrev': 'PMD_UEC', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Uncommented Empty Method', 'abbrev': 'PMD_UEM', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Unnecessary Local Before Return', 'abbrev': 'PMD_ULBR', 'severity': 'Minor'}, {'type': 'Design Rules', 'rule': 'Unsynchronized Static Date Formatter', 'abbrev': 'PMD_USDF', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Use Collection Is Empty', 'abbrev': 'PMD_UCIE', 'severity': 'Major'}, {'type': 'Design Rules', 'rule': 'Use Locale With Case Conversions', 'abbrev': 'PMD_ULWCC', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Use Notify All Instead Of Notify', 'abbrev': 'PMD_UNAION', 'severity': 'Critical'}, {'type': 'Design Rules', 'rule': 'Use Varargs', 'abbrev': 'PMD_UV', 'severity': 'Minor'}, {'type': 'Finalizer Rules', 'rule': 'Avoid Calling Finalize', 'abbrev': 'PMD_ACF', 'severity': 'Major'}, {'type': 'Finalizer Rules', 'rule': 'Empty Finalizer', 'abbrev': 'PMD_EF', 'severity': 'Minor'}, {'type': 'Finalizer Rules', 'rule': 'Finalize Does Not Call Super Finalize', 'abbrev': 'PMD_FDNCSF', 'severity': 'Critical'}, {'type': 'Finalizer Rules', 'rule': 'Finalize Only Calls Super Finalize', 'abbrev': 'PMD_FOCSF', 'severity': 'Minor'}, {'type': 'Finalizer Rules', 'rule': 'Finalize Overloaded', 'abbrev': 'PMD_FO', 'severity': 'Critical'}, {'type': 'Finalizer Rules', 'rule': 'Finalize Should Be Protected', 'abbrev': 'PMD_FSBP', 'severity': 'Critical'}, {'type': 'Import Statement Rules', 'rule': 'Dont Import Java Lang', 'abbrev': 'PMD_DIJL', 'severity': 'Minor'}, {'type': 'Import Statement Rules', 'rule': 'Duplicate Imports', 'abbrev': 'PMD_DI', 'severity': 'Minor'}, {'type': 'Import Statement Rules', 'rule': 'Import From Same Package', 'abbrev': 'PMD_IFSP', 'severity': 'Minor'}, {'type': 'Import Statement Rules', 'rule': 'Too Many Static Imports', 'abbrev': 'PMD_TMSI', 'severity': 'Major'}, {'type': 'Import Statement Rules', 'rule': 'Unnecessary Fully Qualified Name', 'abbrev': 'PMD_UFQN', 'severity': 'Minor'}, {'type': 'J2EE Rules', 'rule': 'Do Not Call System Exit', 'abbrev': 'PMD_DNCSE', 'severity': 'Critical'}, {'type': 'J2EE Rules', 'rule': 'Local Home Naming Convention', 'abbrev': 'PMD_LHNC', 'severity': 'Major'}, {'type': 'J2EE Rules', 'rule': 'Local Interface Session Naming Convention', 'abbrev': 'PMD_LISNC', 'severity': 'Major'}, {'type': 'J2EE Rules', 'rule': 'MDBAnd Session Bean Naming Convention', 'abbrev': 'PMD_MDBASBNC', 'severity': 'Major'}, {'type': 'J2EE Rules', 'rule': 'Remote Interface Naming Convention', 'abbrev': 'PMD_RINC', 'severity': 'Major'}, {'type': 'J2EE Rules', 'rule': 'Remote Session Interface Naming Convention', 'abbrev': 'PMD_RSINC', 'severity': 'Major'}, {'type': 'J2EE Rules', 'rule': 'Static EJBField Should Be Final', 'abbrev': 'PMD_SEJBFSBF', 'severity': 'Critical'}, {'type': 'JUnit Rules', 'rule': 'JUnit Assertions Should Include Message', 'abbrev': 'PMD_JUASIM', 'severity': 'Minor'}, {'type': 'JUnit Rules', 'rule': 'JUnit Spelling', 'abbrev': 'PMD_JUS', 'severity': 'Critical'}, {'type': 'JUnit Rules', 'rule': 'JUnit Static Suite', 'abbrev': 'PMD_JUSS', 'severity': 'Critical'}, {'type': 'JUnit Rules', 'rule': 'JUnit Test Contains Too Many Asserts', 'abbrev': 'PMD_JUTCTMA', 'severity': 'Minor'}, {'type': 'JUnit Rules', 'rule': 'JUnit Tests Should Include Assert', 'abbrev': 'PMD_JUTSIA', 'severity': 'Major'}, {'type': 'JUnit Rules', 'rule': 'Simplify Boolean Assertion', 'abbrev': 'PMD_SBA', 'severity': 'Minor'}, {'type': 'JUnit Rules', 'rule': 'Test Class Without Test Cases', 'abbrev': 'PMD_TCWTC', 'severity': 'Minor'}, {'type': 'JUnit Rules', 'rule': 'Unnecessary Boolean Assertion', 'abbrev': 'PMD_UBA', 'severity': 'Minor'}, {'type': 'JUnit Rules', 'rule': 'Use Assert Equals Instead Of Assert True', 'abbrev': 'PMD_UAEIOAT', 'severity': 'Major'}, {'type': 'JUnit Rules', 'rule': 'Use Assert Null Instead Of Assert True', 'abbrev': 'PMD_UANIOAT', 'severity': 'Minor'}, {'type': 'JUnit Rules', 'rule': 'Use Assert Same Instead Of Assert True', 'abbrev': 'PMD_UASIOAT', 'severity': 'Minor'}, {'type': 'JUnit Rules', 'rule': 'Use Assert True Instead Of Assert Equals', 'abbrev': 'PMD_UATIOAE', 'severity': 'Minor'}, {'type': 'Jakarta Commons Logging Rules', 'rule': 'Guard Debug Logging', 'abbrev': 'PMD_GDL', 'severity': 'Major'}, {'type': 'Jakarta Commons Logging Rules', 'rule': 'Guard Log Statement', 'abbrev': 'PMD_GLS', 'severity': 'Minor'}, {'type': 'Jakarta Commons Logging Rules', 'rule': 'Proper Logger', 'abbrev': 'PMD_PL', 'severity': 'Minor'}, {'type': 'Jakarta Commons Logging Rules', 'rule': 'Use Correct Exception Logging', 'abbrev': 'PMD_UCEL', 'severity': 'Major'}, {'type': 'Java Logging Rules', 'rule': 'Avoid Print Stack Trace', 'abbrev': 'PMD_APST', 'severity': 'Major'}, {'type': 'Java Logging Rules', 'rule': 'Guard Log Statement Java Util', 'abbrev': 'PMD_GLSJU', 'severity': 'Minor'}, {'type': 'Java Logging Rules', 'rule': 'Logger Is Not Static Final', 'abbrev': 'PMD_LINSF', 'severity': 'Minor'}, {'type': 'Java Logging Rules', 'rule': 'More Than One Logger', 'abbrev': 'PMD_MTOL', 'severity': 'Major'}, {'type': 'Java Logging Rules', 'rule': 'System Println', 'abbrev': 'PMD_SP', 'severity': 'Major'}, {'type': 'JavaBean Rules', 'rule': 'Missing Serial Version UID', 'abbrev': 'PMD_MSVUID', 'severity': 'Major'}, {'type': 'Naming Rules', 'rule': 'Avoid Dollar Signs', 'abbrev': 'PMD_ADS', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Avoid Field Name Matching Method Name', 'abbrev': 'PMD_AFNMMN', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Avoid Field Name Matching Type Name', 'abbrev': 'PMD_AFNMTN', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Boolean Get Method Name', 'abbrev': 'PMD_BGMN', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Class Naming Conventions', 'abbrev': 'PMD_CNC', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Generics Naming', 'abbrev': 'PMD_GN', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Method Naming Conventions', 'abbrev': 'PMD_MeNC', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Method With Same Name As Enclosing Class', 'abbrev': 'PMD_MWSNAEC', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'No Package', 'abbrev': 'PMD_NP', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Package Case', 'abbrev': 'PMD_PC', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Short Class Name', 'abbrev': 'PMD_SCN', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Short Method Name', 'abbrev': 'PMD_SMN', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Suspicious Constant Field Name', 'abbrev': 'PMD_SCFN', 'severity': 'Minor'}, {'type': 'Naming Rules', 'rule': 'Suspicious Equals Method Name', 'abbrev': 'PMD_SEMN', 'severity': 'Critical'}, {'type': 'Naming Rules', 'rule': 'Suspicious Hashcode Method Name', 'abbrev': 'PMD_SHMN', 'severity': 'Critical'}, {'type': 'Naming Rules', 'rule': 'Variable Naming Conventions', 'abbrev': 'PMD_VNC', 'severity': 'Minor'}, {'type': 'Optimization Rules', 'rule': 'Add Empty String', 'abbrev': 'PMD_AES', 'severity': 'Minor'}, {'type': 'Optimization Rules', 'rule': 'Avoid Array Loops', 'abbrev': 'PMD_AAL', 'severity': 'Major'}, {'type': 'Optimization Rules', 'rule': 'Redundant Field Initializer', 'abbrev': 'PMD_RFI', 'severity': 'Minor'}, {'type': 'Optimization Rules', 'rule': 'Unnecessary Wrapper Object Creation', 'abbrev': 'PMD_UWOC', 'severity': 'Major'}, {'type': 'Optimization Rules', 'rule': 'Use Array List Instead Of Vector', 'abbrev': 'PMD_UALIOV', 'severity': 'Minor'}, {'type': 'Optimization Rules', 'rule': 'Use Arrays As List', 'abbrev': 'PMD_UAAL', 'severity': 'Major'}, {'type': 'Optimization Rules', 'rule': 'Use String Buffer For String Appends', 'abbrev': 'PMD_USBFSA', 'severity': 'Major'}, {'type': 'Security Code Guideline Rules', 'rule': 'Array Is Stored Directly', 'abbrev': 'PMD_AISD', 'severity': 'Major'}, {'type': 'Security Code Guideline Rules', 'rule': 'Method Returns Internal Array', 'abbrev': 'PMD_MRIA', 'severity': 'Major'}, {'type': 'Strict Exception Rules', 'rule': 'Avoid Catching Generic Exception', 'abbrev': 'PMD_ACGE', 'severity': 'Major'}, {'type': 'Strict Exception Rules', 'rule': 'Avoid Catching NPE', 'abbrev': 'PMD_ACNPE', 'severity': 'Critical'}, {'type': 'Strict Exception Rules', 'rule': 'Avoid Catching Throwable', 'abbrev': 'PMD_ACT', 'severity': 'Major'}, {'type': 'Strict Exception Rules', 'rule': 'Avoid Losing Exception Information', 'abbrev': 'PMD_ALEI', 'severity': 'Major'}, {'type': 'Strict Exception Rules', 'rule': 'Avoid Rethrowing Exception', 'abbrev': 'PMD_ARE', 'severity': 'Minor'}, {'type': 'Strict Exception Rules', 'rule': 'Avoid Throwing New Instance Of Same Exception', 'abbrev': 'PMD_ATNIOSE', 'severity': 'Minor'}, {'type': 'Strict Exception Rules', 'rule': 'Avoid Throwing Null Pointer Exception', 'abbrev': 'PMD_ATNPE', 'severity': 'Critical'}, {'type': 'Strict Exception Rules', 'rule': 'Avoid Throwing Raw Exception Types', 'abbrev': 'PMD_ATRET', 'severity': 'Major'}, {'type': 'Strict Exception Rules', 'rule': 'Do Not Extend Java Lang Error', 'abbrev': 'PMD_DNEJLE', 'severity': 'Critical'}, {'type': 'Strict Exception Rules', 'rule': 'Do Not Throw Exception In Finally', 'abbrev': 'PMD_DNTEIF', 'severity': 'Critical'}, {'type': 'Strict Exception Rules', 'rule': 'Exception As Flow Control', 'abbrev': 'PMD_EAFC', 'severity': 'Major'}, {'type': 'String and StringBuffer Rules', 'rule': 'Avoid Duplicate Literals', 'abbrev': 'PMD_ADL', 'severity': 'Major'}, {'type': 'String and StringBuffer Rules', 'rule': 'Avoid String Buffer Field', 'abbrev': 'PMD_ASBF', 'severity': 'Minor'}, {'type': 'String and StringBuffer Rules', 'rule': 'Consecutive Appends Should Reuse', 'abbrev': 'PMD_CASR', 'severity': 'Minor'}, {'type': 'String and StringBuffer Rules', 'rule': 'Consecutive Literal Appends', 'abbrev': 'PMD_CLA', 'severity': 'Minor'}, {'type': 'String and StringBuffer Rules', 'rule': 'Inefficient String Buffering', 'abbrev': 'PMD_ISB', 'severity': 'Minor'}, {'type': 'String and StringBuffer Rules', 'rule': 'String Buffer Instantiation With Char', 'abbrev': 'PMD_SBIWC', 'severity': 'Critical'}, {'type': 'String and StringBuffer Rules', 'rule': 'String Instantiation', 'abbrev': 'PMD_StI', 'severity': 'Minor'}, {'type': 'String and StringBuffer Rules', 'rule': 'String To String', 'abbrev': 'PMD_STS', 'severity': 'Minor'}, {'type': 'String and StringBuffer Rules', 'rule': 'Unnecessary Case Change', 'abbrev': 'PMD_UCC', 'severity': 'Minor'}, {'type': 'String and StringBuffer Rules', 'rule': 'Use Equals To Compare Strings', 'abbrev': 'PMD_UETCS', 'severity': 'Critical'}, {'type': 'Type Resolution Rules', 'rule': 'Clone Method Must Implement Cloneable', 'abbrev': 'PMD_ClMMIC', 'severity': 'Major'}, {'type': 'Type Resolution Rules', 'rule': 'Loose Coupling', 'abbrev': 'PMD_LoC', 'severity': 'Major'}, {'type': 'Type Resolution Rules', 'rule': 'Signature Declare Throws Exception', 'abbrev': 'PMD_SiDTE', 'severity': 'Major'}, {'type': 'Type Resolution Rules', 'rule': 'Unused Imports', 'abbrev': 'PMD_UnI', 'severity': 'Minor'}, {'type': 'Unnecessary and Unused Code Rules', 'rule': 'Unused Local Variable', 'abbrev': 'PMD_ULV', 'severity': 'Major'}, {'type': 'Unnecessary and Unused Code Rules', 'rule': 'Unused Private Field', 'abbrev': 'PMD_UPF', 'severity': 'Major'}, {'type': 'Unnecessary and Unused Code Rules', 'rule': 'Unused Private Method', 'abbrev': 'PMD_UPM', 'severity': 'Major'}]

STATIC = ['PDA', 'LOC', 'CLOC', 'PUA', 'McCC', 'LLOC',  'LDC', 'NOS', 'MISM', 'CCL', 'TNOS', 'TLLOC',
          'NLE', 'CI', 'HPL', 'MI', 'HPV', 'CD', 'NOI', 'NUMPAR', 'MISEI', 'CC', 'LLDC', 'NII', 'CCO', 'CLC', 'TCD', 'NL', 'TLOC',  'CLLC', 'TCLOC', 'MIMS', 'HDIF', 'DLOC', 'NLM', 'DIT', 'NPA', 'TNLPM', 
          'TNLA', 'NLA', 'AD', 'TNLPA', 'NM', 'TNG', 'NLPM', 'TNM', 'NOC', 'NOD', 'NOP', 'NLS', 'NG', 'TNLG', 'CBOI', 'RFC', 'NLG', 'TNLS', 'TNA', 'NLPA', 'NOA', 'WMC', 'NPM', 'TNPM', 'TNS', 'NA', 'LCOM5', 'NS', 'CBO', 'TNLM', 'TNPA']

STATIC_FILE = ['McCC', 'PDA', 'PUA', 'LOC', 'LLOC']
STATIC_CLASS = ['LCOM5', 'NL', 'NLE', 'WMC', 'CBO', 'CBOI', 'NII', 'NOI', 'RFC', 'AD', 'CD', 'CLOC', 'DLOC', 'PDA', 'PUA', 'TCD', 'TCLOC', 'DIT', 'NOA', 'NOC', 'NOD', 'NOP', 'LOC', 'LLOC', 'NA', 'NG', 'NLA', 'NLG', 'NLM', 'NLPA', 'NLPM', 'NLS', 'NM', 'NPA', 'NPM', 'NS', 'NOS', 'TLOC', 'TLLOC', 'TNA', 'TNG', 'TNLA', 'TNLG', 'TNLM', 'TNLPA', 'TNLPM', 'TNLS', 'TNM', 'TNPA', 'TNPM', 'TNS', 'TNOS'] + ['LDC', 'CCL', 'CI', 'CC', 'LLDC', 'CCO', 'CLC', 'CLLC']
STATIC_INTERFACE = ['LCOM5', 'NL', 'NLE', 'WMC', 'CBO', 'CBOI', 'NII', 'NOI', 'RFC', 'AD', 'CD', 'CLOC', 'DLOC', 'PDA', 'PUA', 'TCD', 'TCLOC', 'DIT', 'NOA', 'NOC', 'NOD', 'NOP', 'LOC', 'LLOC', 'NA', 'NG', 'NLA', 'NLG', 'NLM', 'NLPA', 'NLPM', 'NLS', 'NM', 'NPA', 'NPM', 'NS', 'NOS', 'TLOC', 'TLLOC', 'TNA', 'TNG', 'TNLA', 'TNLG', 'TNLM', 'TNLPA', 'TNLPM', 'TNLS', 'TNM', 'TNPA', 'TNPM', 'TNS', 'TNOS']
STATIC_ENUM = ['LCOM5', 'NL', 'NLE', 'WMC', 'CBO', 'CBOI', 'NII', 'NOI', 'RFC', 'AD', 'CD', 'CLOC', 'DLOC', 'PDA', 'PUA', 'TCD', 'TCLOC', 'DIT', 'NOA', 'NOC', 'NOD', 'NOP', 'LOC', 'LLOC', 'NA', 'NG', 'NLA', 'NLG', 'NLM', 'NLPA', 'NLPM', 'NLS', 'NM', 'NPA', 'NPM', 'NS', 'NOS', 'TLOC', 'TLLOC', 'TNA', 'TNG', 'TNLA', 'TNLG', 'TNLM', 'TNLPA', 'TNLPM', 'TNLS', 'TNM', 'TNPA', 'TNPM', 'TNS', 'TNOS']
STATIC_METHOD = ['MIMS', 'MI', 'MISEI', 'MISM', 'McCC', 'NL', 'NLE', 'NII', 'NOI', 'CD', 'CLOC', 'DLOC', 'TCD', 'TCLOC', 'LOC', 'LLOC', 'NUMPAR', 'NOS', 'TLOC', 'TLLOC', 'TNOS'] + ['LDC', 'CCL', 'CI', 'HPV', 'CC', 'LLDC', 'CCO', 'CLC', 'CLLC']

# not in all versions: 'HCPL', 'HDIF', 'HEFF', 'HNDB', 'HPL', 'HLV', 'HTRP', 'HVOL', 
STATIC_AGGREGATIONS = ['min', 'max', 'avg', 'median', 'sum']


FGJIT_FEATURES = ['comm', 'adev', 'ddev', 'nddev', 'add', 'del', 'own', 'minor', 'sctr', 'nadev', 'ncomm', 'nsctr', 'oexp', 'exp', 'nd', 'entropy', 'la', 'ld', 'lt', 'age', 'nuc', 'cexp', 'sexp', 'rexp', 'fix_bug']
JIT_FEATURES = ['kamei_ns', 'kamei_nd', 'kamei_nf', 'kamei_entropy', 'kamei_la', 'kamei_ld', 'kamei_lt', 'kamei_fix', 'kamei_fix', 'kamei_ndev', 'kamei_age', 'kamei_nuc', 'kamei_exp', 'kamei_sexp', 'kamei_rexp']
WD_FEATURES = ['sm_current_WD', 'sm_parent_WD', 'sm_delta_WD', 'sm_system_WD', 'sm_parent_system_WD']

PMD_FEATURES = []
for p in PMD_RULES:
    PMD_FEATURES.append('current_{}'.format(p['abbrev']))
    PMD_FEATURES.append('parent_{}'.format(p['abbrev']))
    PMD_FEATURES.append('delta_{}'.format(p['abbrev']))

STATIC_FILE_FEATURES = []
for s in STATIC_FILE:
    STATIC_FILE_FEATURES.append('current_{}_file'.format(s))
    STATIC_FILE_FEATURES.append('parent_{}_file'.format(s))
    STATIC_FILE_FEATURES.append('delta_{}_file'.format(s))

STATIC_CLASS_FEATURES = []
for s in STATIC_CLASS:
    for a in STATIC_AGGREGATIONS:
        STATIC_CLASS_FEATURES.append('current_{}_class_{}'.format(s, a))
        STATIC_CLASS_FEATURES.append('parent_{}_class_{}'.format(s, a))
        STATIC_CLASS_FEATURES.append('delta_{}_class_{}'.format(s, a))

STATIC_INTERFACE_FEATURES = []
for s in STATIC_INTERFACE:
    for a in STATIC_AGGREGATIONS:
        STATIC_INTERFACE_FEATURES.append('current_{}_interface_{}'.format(s, a))
        STATIC_INTERFACE_FEATURES.append('parent_{}_interface_{}'.format(s, a))
        STATIC_INTERFACE_FEATURES.append('delta_{}_interface_{}'.format(s, a))
    
STATIC_ENUM_FEATURES = []
for s in STATIC_ENUM:
    for a in STATIC_AGGREGATIONS:
        STATIC_ENUM_FEATURES.append('current_{}_enum_{}'.format(s, a))
        STATIC_ENUM_FEATURES.append('parent_{}_enum_{}'.format(s, a))
        STATIC_ENUM_FEATURES.append('delta_{}_enum_{}'.format(s, a))

STATIC_METHOD_FEATURES = []
for s in STATIC_METHOD:
    for a in STATIC_AGGREGATIONS:
        STATIC_METHOD_FEATURES.append('current_{}_method_{}'.format(s, a))
        STATIC_METHOD_FEATURES.append('parent_{}_method_{}'.format(s, a))
        STATIC_METHOD_FEATURES.append('delta_{}_method_{}'.format(s, a))

STATIC_FEATURES = STATIC_FILE_FEATURES + STATIC_CLASS_FEATURES + STATIC_INTERFACE_FEATURES + STATIC_ENUM_FEATURES + STATIC_METHOD_FEATURES
ALL_FEATURES = STATIC_FEATURES + FGJIT_FEATURES + JIT_FEATURES + WD_FEATURES + PMD_FEATURES

