# Data Description

The data we provide is for *file-level just-in-time defect prediction*. This means that we have one instance in our data for every file that was changed within each commit. We describe the data in detail below. 

## Data Source and Tools

The data was extracted from the [SmartSHARK database 2.0](https://smartshark.github.io/dbreleases/) using the tool [Gierlappen](https://github.com/atrautsch/Gierlappen).

## Labels (Training Data)

The labels for the training data were determined such that we marked commits as bug fixing if
- they contain a link to a Jira issue and
- the type of the issue is a bug and the [fastText based classification](https://doi.org/10.1007/s10664-020-09885-w) approach with 95% recall confirms this.

We then use git blames to determine the inducing changes for each changed line in a Java file following SZZ. The suspect boundary is based on the newest linked bug issues. Empty lines, comment lines, and lines determined as refactoring by [RefactoringMiner 1.0](https://github.com/tsantalis/RefactoringMiner) are ignored. 

## Label (Test Data)

Follows when we release the test data :)

## Description of the file format

The data is stored as compressed csv files. `pandas` can natively read this. For other CSV readers you probably need to unpack the data first. Below, we describe the columns of the files. You can also find code for loading the data within our samples. The utils.py we provide with the samples also contains constants that we state here, which you can use for filtering the features. 

|Field | Description|
|------|------------|
|commit|SHA hash of the commit|
|committer_date|Date of the commit|
|file|Current filename|
|oldest_name|Oldest filename known for this file, if a file is renamed from A to B the oldest_name is A while file is B|
|change_type|Pydriller type of change, e.g., ModificationType.ADD, ModificationType.MODIFY, ModificationType.DELETE, ModificationType.RENAME|
|comm,adev,ddev,add,del,own,minor,sctr,nd,entropy,la,ld,cexp,rexp,sexp,nuc,age,oexp,exp,nsctr,ncom,nadev,nddev,lt,fix_bug| Fine-grained just-in-time features after [Pascarella et al.](https://www.lucapascarella.com/articles/2018/Pascarella_JSS_2018.pdf)|
|kamei_\*|Just-in-time features after [Kamei et al.](https://ieeexplore.ieee.org/document/6341763)|
|sm_current_WD,sm_parent_WD,sm_delta_WD|Warning density of the file, the previous warning density of the file and the difference between both|
|current_PMD_\*,parent_PMD_\*,delta_PMD_\*|Number of [PMD](https://pmd.github.io/) warnings for the file, previous and difference between both. Via OpenStaticAnalyzers [PMD](https://raw.githubusercontent.com/sed-inf-u-szeged/OpenStaticAnalyzer/master/OpenStaticAnalyzer/java/doc/usersguide/md/PMDRef.md) integration.|
|current_METRIC_CODETYPE_AGGREGATION, parent_M*_C*_A*, delta_M*_C*_A*|METRIC includes [Static source code metrics](https://raw.githubusercontent.com/sed-inf-u-szeged/OpenStaticAnalyzer/master/OpenStaticAnalyzer/java/doc/usersguide/md/SourceCodeMetricsRef.md), [clone metrics](https://raw.githubusercontent.com/sed-inf-u-szeged/OpenStaticAnalyzer/master/OpenStaticAnalyzer/java/doc/usersguide/md/CodeDuplicationMetricsRef.md) from [OpenStaticAnalyzer](https://github.com/sed-inf-u-szeged/OpenStaticAnalyzer). CODETYPE includes file,class,method,interface,enum. AGGREGATION includes min,max,sum,avg,median (only for non-file metrics)|
|inducing_JIRAKEY_\*_\*|Bug matrix, which file change induces bug JIRAKEY.
The field name includes the bugfix revision and the bugfix date.|
