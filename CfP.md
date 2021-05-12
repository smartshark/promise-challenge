# PROMISE 2021 Defect Prediction Challenge: Call for Papers

The [International Conference on Predictive Models and Data Analytics in Software Engineering](https://promiseconf.github.io/2021/) is hosting a defect prediction challenge track. With this challenge, we call upon everyone interested in defect prediction, to submit their favorite models and see how they compare to the competition. This challenge is also well suited for term projects of university students. 

Participants can submit any technique they want: This can be a new and unpublished approach, a previously published approach by the same authors, or even an approach that was suggested by somebody else. 

For this challenge, we provide a new and previously unpublished data set for *File Level Just-in-Time Defect Prediction*. This first part of this data set is published together with this challenge and can be used for training the models. The second part of this data set will be released after the submission deadline and will be used for the evaluation. This means that we have *real* test data, that was never touched by anyone for training or improving their models!

# How to Participate in the Challenge

Participants should first familiarize themselves with the challenge.
- Read the [description of the data](dataset.md).
- Read the [requirements for the submissions](requirements.md). 
- Check out the [baselines as samples](approaches).

Afterwards, you can start to develop your own model. We suggest to use the sample as template for your model. If you start from scratch, e.g., because you are not using Python, please ensure that you provide a [Docker container that conforms to our requirements](requirements.md). You can contact us anytime for help with other languages and to ensure conformity prior to the review. 

## Important Dates

- June 20th: Submission Deadline
- July 2nd: Notification of Acceptance
- July 8th: Camera Ready Version

All dates are AoE. 

## Submission

The submission consists of two parts. This first part is a short description of the defect prediction model, which will be published in the PROMISE proceedings.
- Must not be longer than 2 pages + 1 page for references. 
- Must be written in English.
- Must be conform to the [ACM Sigsoft conference proceedings template](https://www.acm.org/publications/proceedings-template). 
- Must be submitted via [HotCRP](https://promise-2021-challenge.hotcrp.com/)
- While you are allowed to submit work that you have not originally developed, you are required to correctly attribute this work to the original authors within this paper. Otherwise, your contribution will be rejected due to plagiarism. 

These descriptions do not require a description of the data set or any empirical results and should focus only on the description of the model itself. 

The second part is the executable artifact, i.e., the defect prediction model submitted to the challenge.
- Must be provided [following our description](requirements.md).
- Must be in line with the description from the paper. 
- Must be must be submitted to the challenge repository via a GitHub Pull Request. Hence, you must fork the repository, add your approach in your own branch, and then create a pull request. Please note that the challenge repository uses the permissive Apache 2.0 License. Please contact the challenge chairs if this is a problem for your approach (e.g., due to copy-left issues). 

## Review Guidelines

The review will be lean and only check the compliance with the submission guidelines. All valid submissions will be accepted.

# Training and Evaluation

We drop the last three month of each project and then use the 250 commits before that for scoring. 

All data that is older than the commit that is predicted may be used for training. This means that only bugs that were reported and fixed prior to the commit that is predicted may be used as bug labels to prevent a time-travel information leak. We therefore recommend to leave a time-gap between the training and test data. Our sample demonstrates how this can be done by using a three-month gap before the last 250 commits of a project. 

You may use data from any project, but the time contraint still applies, i.e., you are not allowed to use data from other projects, that was not available at the time of the commit for which you are running the prediction. 

All models will be ranked in three categories:
- Best overall performance, measured with Matthews Correlation Coefficient.
- Lowest  costs for projects with "cheap" defects. 
- Lowest  costs for projects with "expensive" defects. 

The cost saving potential will be measured assuming that cheap defects are as expensive as quality assurance (e.g., code review) for 1000 Logical Lines of Code (LLOC defined as non-empty, non-comment lines). For expensive defects, we assume that a defect is as expensive as quality assurance for 10,000 LLOC. The costs are computed using Equation 44 of [this article](https://doi.org/10.1109/TSE.2019.2957794) (preprint: https://arxiv.org/abs/1911.04309). 

Additionally, we use four baselines within the challenge:
- The random forest with SMOTE oversampling trained on old commits of the same projects. [You can find the code here](approaches/baseline_rf_wp).
- The random forest on older commits of all projects available. This baseline does not use SMOTE to deal with the class level imbalance due to the amount of memory required. [You can find the code here](approaches/baseline_rf_all).
- A trivial model that predicts everything as defective. [You can find the code here](approaches/baseline_all).
- A trivial model that predicts nothing as defective. [You can find the code here](approaches/baseline_none).

We will use the [autorank](https://github.com/sherbold/autorank) package for the statistical comparison of submission based on Bayesian statistics with a significance level of alpha=0.95. All submissions that are practically equivalent or where the difference is inconclusive to the model with the best performance are considered as winners of the challenge. You can already view a mockup of the code for evaluation [here](evaluation/Challenge.ipynb).
