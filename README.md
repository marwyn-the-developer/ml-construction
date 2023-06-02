# Homework: Machine Learning 4 Construction

Dear candidate,
welcome to the second interview round. This round will assess your technical skills.

In this homework, you will analyse the [Iris Plants](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) dataset and develop a classifier for your customer.

Tech Stack to be used in this homework:
- Python/R
- [Pandas](https://pandas.pydata.org)
- [Scikit-Learn](https://scikit-learn.org/stable/index.html)
- [FastAPI](https://fastapi.tiangolo.com)
- Git

Comment: Powerpoint will not be accepted

## Data Exploration

Understanding datasets is vital to build effective Data-Science products. Hence, we ask you to analyse the iris dataset and
summarize your findings either in a Jupyter Notebook or R Markdown file. As concerns this task, you are free to choose between Python and R, though R would be preferred.

You are free to design the report as you deem adequate. Please ship all your code and convert the notebook to before submission PDF.


## Classification

Your customer wants to classify plants based on input data similar to the one found in the dataset (same structure). Your task is to develop and train a classifier using scikit-learn.
We prepared a class `sidclassifier.py` to get started. 


What we expect:
- Implement a class that can be used to download the iris dataset, train a scikit-learn classifier, test the accuracy of a classifier object, and export a trained classifier to file. You are free to choose the classifier. However, please explain your choice briefly
- Leverage countermeasures against over-fitting
- Measure and communicate the accuracy of your classifier
- Explain how you export the classifier after training and why this step matters

A few questions:
- How would you assess the quality of your source code?
- How would you ship the trained ML model to the customer?
- Two week after shipping your product your customer calls you and complains about low accuracy of your product. How would you react?


## API

Instead of shipping the ML model, your customer wants you to provide a FastAPI. The API will be used to make classifications using web requests similar to `requests.get("https://iris.strabag.com/predict...")`.
See `iris-api.py` to get started.

What we expect:
- Develop an API using FastAPI that uses the ML model trained in the previous step to classify plants. 
- The customer will use web request to interact with the API. You can assume that the input is similar to the iris dataset and one prediction per request is expected.
- You must not train the ML model using the API. Use exported models only!

## Submission

We kindly ask you to submit your homework via Git, as our team uses Git on a daily basis. Please create a public repository
on Bitbucket/Github. Please share the URL with us. In case you are not familiar with Git, you are free to submit your code via e-mail.

Good Luck!
