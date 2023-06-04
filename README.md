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

In the __model_training.ipynb__ notebook a number of different ML algorithms are tested and compared. The model that was chosen is a simple MLP as it provided the best accuracy and inference runtime.

- Leverage countermeasures against over-fitting

One measure is the limited model capacity with just one hidden layer and a hidden size of just 5. Another measure is the use of L2 weight decay with a decay rate of 1e-05.

- Measure and communicate the accuracy of your classifier

Details on model performance are discussed in __model_training.ipynb__.

- Explain how you export the classifier after training and why this step matters

The classifier is exported as a file with the __skops__ library which is a more secure option for model persistence as opposed to __pickle__. This step matters to avoid constant refitting which can be expensive when the dataset is large.

A few questions:
- How would you assess the quality of your source code?

One crucial aspect would be readability. Code that is easier to understand is also easier to maintain. This can include among other things descriptive names of variables and functions, good formatting and good use of comments to provide some reasoning to some parts of the code.

Minimization of code reuse. This can require upholding SOLID principles.

- How would you ship the trained ML model to the customer?

This would depend on what has been agreed with the customer. It could vary in levels of abstraction beginning from a persisted version of the model parameters to an API or a complete software solution with a user interface.

- Two week after shipping your product your customer calls you and complains about low accuracy of your product. How would you react?

Ask concrete questions as to how the customer came to this conclusion. Depending on the customer's answer if this conclusion came from certain inputs that the model has provided wrong outputs to. I would look at these inputs and try to reason why the model performed so poorly on those examples and adapt either the training algorithm or rather the training data to accomodate such shortcomings.

Another reason for a model's poor performance that is related to the first point, is that the input data systematically changed over time and the model trained on older data cannot make accurate predictions anymore. If this is the case the model has to be retrained on newer data.

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
