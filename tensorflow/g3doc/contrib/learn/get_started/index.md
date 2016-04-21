# Introduction

Below are few simple examples of the API to get you started with TensorFlow Learn.
For more examples, please see [examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/skflow).

## General tips

-  It's useful to re-scale dataset before passing to estimator to 0 mean and unit standard deviation. Stochastic Gradient Descent doesn't always do the right thing when variable are very different scale.

-  Categorical variables should be managed before passing input to the estimator.

## Linear Classifier

Simple linear classification:

    from tensorflow.contrib import learn
    from sklearn import datasets, metrics

    iris = datasets.load_iris()
    classifier = learn.TensorFlowLinearClassifier(n_classes=3)
    classifier.fit(iris.data, iris.target)
    score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
    print("Accuracy: %f" % score)

## Linear Regressor

Simple linear regression:

    from tensorflow.contrib import learn
    from sklearn import datasets, metrics, preprocessing

    boston = datasets.load_boston()
    X = preprocessing.StandardScaler().fit_transform(boston.data)
    regressor = learn.TensorFlowLinearRegressor()
    regressor.fit(X, boston.target)
    score = metrics.mean_squared_error(regressor.predict(X), boston.target)
    print ("MSE: %f" % score)

## Deep Neural Network

Example of 3 layer network with 10, 20 and 10 hidden units respectively:

    from tensorflow.contrib import learn
    from sklearn import datasets, metrics

    iris = datasets.load_iris()
    classifier = learn.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
    classifier.fit(iris.data, iris.target)
    score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
    print("Accuracy: %f" % score)

## Custom model

Example of how to pass a custom model to the TensorFlowEstimator:

    from tensorflow.contrib import learn
    from sklearn import datasets, metrics

    iris = datasets.load_iris()

    def my_model(X, y):
        """This is DNN with 10, 20, 10 hidden layers, and dropout of 0.5 probability."""
        layers = learn.ops.dnn(X, [10, 20, 10], keep_prob=0.5)
        return learn.models.logistic_regression(layers, y)

    classifier = learn.TensorFlowEstimator(model_fn=my_model, n_classes=3)
    classifier.fit(iris.data, iris.target)
    score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
    print("Accuracy: %f" % score)

## Saving / Restoring models

Each estimator has a ``save`` method which takes folder path where all model information will be saved. For restoring you can just call ``learn.TensorFlowEstimator.restore(path)`` and it will return object of your class.

Some example code:

    from tensorflow.contrib import learn

    classifier = learn.TensorFlowLinearRegression()
    classifier.fit(...)
    classifier.save('/tmp/tf_examples/my_model_1/')

    new_classifier = TensorFlowEstimator.restore('/tmp/tf_examples/my_model_2')
    new_classifier.predict(...)

## Summaries

To get nice visualizations and summaries you can use ``logdir`` parameter on ``fit``. It will start writing summaries for ``loss`` and histograms for variables in your model. You can also add custom summaries in your custom model function by calling ``tf.summary`` and passing Tensors to report.

    classifier = learn.TensorFlowLinearRegression()
    classifier.fit(X, y, logdir='/tmp/tf_examples/my_model_1/')

Then run next command in command line:

    tensorboard --logdir=/tmp/tf_examples/my_model_1

and follow reported url.

Graph visualization: Text classification RNN Graph image

Loss visualization: Text classification RNN Loss image


## More examples

See [examples folder](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/skflow) for:

-  Easy way to handle categorical variables - words are just an example of categorical variable.
-  Text Classification - see examples for RNN, CNN on word and characters.
-  Language modeling and text sequence to sequence.
-  Images (CNNs) - see example for digit recognition.
-  More & deeper - different examples showing DNNs and CNNs
