# SID DDA

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from skops.io import dump


class IsolationForestOutlierRemover:
    def __init__(self, contamination):
        self.contamination = contamination

    def transform(self, X, y):
        iforest = IsolationForest(
            n_estimators=100, contamination=self.contamination, random_state=0
        )
        pred = iforest.fit_predict(X)
        return X.iloc[pred == 1], y.iloc[pred == 1]


class SIDCLASSIFIER:
    def __init__(self):
        """
        init the class
        """
        self.outlier_remover = IsolationForestOutlierRemover(0.05)

        param_grid = {
            "hidden_layer_sizes": [(5,), (10,), (20,)],
            "alpha": [1e-05, 1e-03, 1e-02, 1e-01, 0],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": [1e-05, 1e-03, 1e-02, 1e-01],
        }
        mlp_classifier = MLPClassifier(
            solver="sgd", max_iter=10000000000, random_state=0
        )
        self.mlp_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.99)),
                ("gb", GridSearchCV(mlp_classifier, param_grid, cv=5, n_jobs=-1)),
            ]
        )

    def fetch_dataset(self):
        """
        fetch dataset and store to file
        """
        iris = datasets.load_iris()
        features_df = pd.DataFrame(
            iris.data, columns=[x.replace(" (cm)", "") for x in iris.feature_names]
        )
        target_df = pd.DataFrame(iris.target, columns=["species.code"])
        target_df["species"] = target_df["species.code"].map(
            lambda x: iris.target_names[x]
        )

        iris_df = pd.concat([features_df, target_df], axis=1)
        iris_df.to_csv("iris.csv")

    def iris_train_test_split(self, train_size=0.8, random_state=None):
        features = ["sepal length", "sepal width", "petal length", "petal width"]
        iris_df = pd.read_csv("iris.csv")

        return train_test_split(
            iris_df[features],
            iris_df.species,
            train_size=train_size,
            random_state=random_state,
            stratify=iris_df.species,
        )

    def train(self):
        """
        train the classifier on the dataset
        """
        X_train, _, y_train, _ = self.iris_train_test_split(train_size=0.8, random_state=0)
        X_train, y_train = self.outlier_remover.transform(X_train, y_train)
        self.mlp_pipe.fit(X_train, y_train.values.ravel())

    def assess_accuracy(self):
        """
        assess the accuracy of a trained classifier
        """
        _, X_test, _, y_test = self.iris_train_test_split(train_size=0.8, random_state=0)
        y_pred = self.mlp_pipe.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def store_to_file(self):
        """
        store trained classifier to file
        """
        dump(self.mlp_pipe, "iris_classifier.skops")


def main():
    """
    main steps
    """

    iris = SIDCLASSIFIER()

    iris.fetch_dataset()

    iris.train()

    iris.assess_accuracy()

    iris.store_to_file()


if __name__ == "__main__":
    main()
