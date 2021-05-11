# Requirements

In order to allow us to run all approaches without problems, we have a couple requirements that submissions must fulfill. 

- All submissions must be in a subfolder with a describing name of the approaches folder.
- All submissions must use five parameters as input:
  - The input folder with the data. This should usually have the value `../../data`.
  - The output folder for the results as csv. This should usually have the value `../../scores`.
  - The name of the approach. Must not have whitespaces.
  - The months that are ignored at the end of the test data. Will be `3` for this challenge.
  - The number of commits used for the test data. Will be `250` for this challenge. 
- All submissions must have a training time for a single test project of less than 12 hours. 
- Submissions should be designed such that they run with 32 GB memory and may use additional virtual RAM. The virtual RAM will result in a slow down, which should be accounted for. We will try to run the challenge on a very strong machine (several hundred GB memory), but cannot guarantee that approaches with huge memory consumption can be executed by us.
- If you require dedicated hardware (e.g., GPUs) for training, please contact us before submission so we can ensure that we can adequately execute your approach. 

The submissions must also be packaged in a way that we can easily execute them. Please find the guidance for this below. 

## For Python

If you are using Python 3.6, 3.7, or 3.8 an up to date requirements.txt is sufficient. We will use a virtual environment to run your model. If you want to use this approach, you must meet the following requirements. You must have a command line python script that we can call. In the baselines, this is the `approach.py`. Please go the the folder of your approach and use the following bash calls to verify that this is working. The example is for Ubuntu 20.04 and may need to be adopted for other environemnts (e.g. Mac). 

```
python3 -m venv venv
source venv/bin/active
pip install -r requirements
python approach.py ../../data ../../scores YOUR_APPROACH_NAME 3 250
```

A simple way to achieve this is to copy one ouf our samples and simply modify the `approach.py` with your approach and add any additional libraries to the requirements.txt. 

## For other technologies (R, Java, Python 3.9, ...)

For other technologies, you must provide a Docker container. 

- The container must be available on Dockerhub and the Dockerfile should be part of the approach.
- There must be a `.sh` script for pulling the container. 
- There must be `.sh` script that takes as input the five parameters described below and that runs the approach in the container. You should use the `-v` or `--mount` flag to read access the folders for input and output. 

If you plan to use Docker, we suggest you contact us so we can ensure on both sides that this works as expected and prevent trouble during the very short review period. 
