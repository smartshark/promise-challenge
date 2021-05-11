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
