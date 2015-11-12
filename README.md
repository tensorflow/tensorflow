# Scikit Flow

This is a simplified interface for TensorFlow, to get people started on predictive analytics and data mining.

## Installation

First, make sure you have TensorFlow and Scikit Learn installed, then just run:

    pip install git+git://github.com/ilblackdragon/skflow.git

## Usage

Example usage:

    iris = datasets.load_iris()
    classifier = skflow.TensorFlowClassifier(n_classes=3)
    classifier.fit(iris.data, iris.target)
    score = accuracy_score(classifier.predict(iris.data), iris.target)



