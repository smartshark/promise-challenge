# PROMISE 2021 Defect Prediction Challenge: Call for Papers


The International Conference on Predictive Models and Data Analytics in Software Engineering is hosting a defect prediction challenge track. With this challenge, we call upon everyone interested in defect prediction, to submit their favorite models and see how they compare to the competion. 

Participants can submit any technique they want: This can be a new and unpublished approach, a previously published approach by the same authors, or even an approach that was suggested by somebody else. 

For this challenge, we provide a new and previously unpublished data set for *File Level Just-in-Time Defect Prediction*. This first part of this data set is published together with this challenge and can be used for training the models. The second part of this data set will be released after the submission deadline and will be used for the evaluation. This means that we have *real* test data, that was never touched by anyone for training or improving their models!

# How to Participate in the Challenge

Participants should first familiarize themselves with the challenge.
- Read the [description of the data](LINK_MISSING).
- Read the [requirements for the training and prediction](LINK_MISSING). 
- Check out the [sample of using a Random Forest](LINK_MISSING).

Afterwards, you can start to develop your own model. We suggest to use the sample as template for your model. If you start from scratch, e.g., because you are not using Python, please ensure that you provide a [Docker container that conforms to our requirements](LINK_MISSING).

## Important Dates

- May 15th: Publication of the Call for Papers and the Challenge
- July 2nd: Submission Deadline
- July 16th: Notification of Acceptance

## Submission

The submission consists of two parts. This first part is a short description of the defect prediction model, which will be published in the PROMISE proceedings.
- Must not be longer than 2 pages + 1 page for references. 
- Must be written in English.
- Must be conform to the [ACM Sigsoft conference proceedings template](LINK_MISSING). 
- Must be submitted via [HotCRP](LINK_MISSING). 
- While you are allowed to submit work that you have not originally developed, you are required to correctly attribute this work to the original authors within this paper. Otherwise, your contribution will be rejected due to plagiarism. 


These descriptions do not require a description of the data set or any empirical results and should focus only on the description of the model itself. 

The second part is the executable artifact, i.e., the defect prediction model submitted to the challenge.
- Must be provided as a Docker container [following our description](LINK_MISSING).
- Must be inline with the description from the paper. 
- Must be must be submitted to the [challenge repository via a GitHub Pull Request](LINK_MISSING). Please note that the challenge repository uses the permissive Apache 2.0 Licence. Please contact the challenge chairs if this is a problem for your approach (e.g., due to copy-left issues). 

## Review Guidelines

The review will be lean and only check the compliance with the submission guidelines. All valid submissions will be accepted.

# Evaluation

All models will be ranked in three categories:
- Best overall performance, measured with Matthews Correlation Coefficient.
- Lowests costs for projects with "cheap" defects. 
- Lowests costs for projects with "expensive" defects. 

The cost saving potential will be measured assuming that cheap defects are as expensive as quality assurance (e.g., code review) for 1000 Logical Lines of Code (LLOC defined as non-empty, non-comment lines). For expensive defects, we assume that a defect is as expensive as quality assurance for 10,000 LLOC. The costs are computed using Equation 44 of [this article](https://doi.org/10.1109/TSE.2019.2957794) (preprint: https://arxiv.org/abs/1911.04309). 

For all three categories, we will compute the scores for each project of the test data with all submitted models. Addionally, we use three baselines within the challenge:
- The simple random forest that is also used as template for submission in Python. 
- A trivial model that predicts everything as defective. 
- A trivial model that predicts nothing as defective. 

We will use the [autorank](https://github.com/sherbold/autorank) package for the statistical comparison of submission based on bayesian statistics with a significance level of alpha=0.95. All submissions that are practically equivalent or where the difference is inconclusive to the model with the best performance are considered as winners of the challenge. 
