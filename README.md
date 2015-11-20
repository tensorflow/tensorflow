# Scikit Flow

This is a simplified interface for TensorFlow, to get people started on predictive analytics and data mining.

## Installation

First, make sure you have TensorFlow and Scikit Learn installed, then just run:

    pip install git+git://github.com/google/skflow.git

## Usage

Example usage:

### Linear Classifier

Simple linear classification.

```Python
iris = datasets.load_iris()
classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
classifier.fit(iris.data, iris.target)
score = accuracy_score(classifier.predict(iris.data), iris.target)
```

### Deep Neural Network

Example of 3 layer network with 10, 20 and 10 hidden units respectively:

```Python
dnn = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
dnn.fit(iris.data, iris.target)
score = accuracy_score(dnn.predict(iris.data), iris.target)
```

## Coming soon

* Easy way to handle categorical variables
* Text categorization
* Images (CNNs)
* More & deeper

