# tf.estimator Quickstart

TensorFlow’s high-level machine learning API (tf.estimator) makes it easy to
configure, train, and evaluate a variety of machine learning models. In this
tutorial, you’ll use tf.estimator to construct a
[neural network](https://en.wikipedia.org/wiki/Artificial_neural_network)
classifier and train it on the
[Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) to
predict flower species based on sepal/petal geometry. You'll write code to
perform the following five steps:

TensorFlow的高级机器学习API(tf.estimator)可以轻松配置, 训练和评估各种机器学习模型. 在本教程中, 您将使用tf.estimator构建神经网络分类器并在[Iris数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set)上进行训练, 以基于萼片/花瓣几何来预测花种. 您将编写代码以执行以下五个步骤：

1.  Load CSVs containing Iris training/test data into a TensorFlow `Dataset`
2.  Construct a @{tf.estimator.DNNClassifier$neural network classifier}
3.  Train the model using the training data
4.  Evaluate the accuracy of the model
5.  Classify new samples


1. 将包含Iris训练/测试数据的CSV加载到TensorFlow数据集中
1. 构建神经网络分类器
1. 使用训练数据拟合模型
1. 评估模型的准确性
1. 分类新样本

NOTE: Remember to @{$install$install TensorFlow on your machine}
before getting started with this tutorial.

## Complete Neural Network Source Code

Here is the full code for the neural network classifier:

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "w") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urllib.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "w") as f:
      f.write(raw)

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))

if __name__ == "__main__":
    main()
```

The following sections walk through the code in detail.

下面将对代码如何运行进行详细解释.

## Load the Iris CSV data to TensorFlow

The [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) contains
150 rows of data, comprising 50 samples from each of three related Iris species:
*Iris setosa*, *Iris virginica*, and *Iris versicolor*.

Iris数据集包含150行数据, 包括来自三种相关虹膜物种中的每一种的50个样本.

![Petal geometry compared for three iris species: Iris setosa, Iris virginica, and Iris versicolor](https://www.tensorflow.org/images/iris_three_species.jpg) **From left to right,
[*Iris setosa*](https://commons.wikimedia.org/w/index.php?curid=170298) (by
[Radomil](https://commons.wikimedia.org/wiki/User:Radomil), CC BY-SA 3.0),
[*Iris versicolor*](https://commons.wikimedia.org/w/index.php?curid=248095) (by
[Dlanglois](https://commons.wikimedia.org/wiki/User:Dlanglois), CC BY-SA 3.0),
and [*Iris virginica*](https://www.flickr.com/photos/33397993@N05/3352169862)
(by [Frank Mayfield](https://www.flickr.com/photos/33397993@N05), CC BY-SA
2.0).**

从左到右分别为Iris setosa, Iris versicolor, Iris virginica.

Each row contains the following data for each flower sample:
[sepal](https://en.wikipedia.org/wiki/Sepal) length, sepal width,
[petal](https://en.wikipedia.org/wiki/Petal) length, petal width, and flower
species. Flower species are represented as integers, with 0 denoting *Iris
setosa*, 1 denoting *Iris versicolor*, and 2 denoting *Iris virginica*.

每行包含以下每个花样品的数据：萼片长度, 萼片宽度, 花瓣长度, 花瓣宽度和花种. 花种以整数表示, 0表示Iris setosa, 1表示Iris versicolor, 2表示Iris virginica.

Sepal Length | Sepal Width | Petal Length | Petal Width | Species
:----------- | :---------- | :----------- | :---------- | :-------
5.1          | 3.5         | 1.4          | 0.2         | 0
4.9          | 3.0         | 1.4          | 0.2         | 0
4.7          | 3.2         | 1.3          | 0.2         | 0
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
7.0          | 3.2         | 4.7          | 1.4         | 1
6.4          | 3.2         | 4.5          | 1.5         | 1
6.9          | 3.1         | 4.9          | 1.5         | 1
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
6.5          | 3.0         | 5.2          | 2.0         | 2
6.2          | 3.4         | 5.4          | 2.3         | 2
5.9          | 3.0         | 5.1          | 1.8         | 2

萼片长度 | 萼片宽度 | 花瓣长度 | 花瓣宽度 | 花种
--- | --- | --- | --- | ---
5.1 | 3.5 | 1.4 | 0.2 | 0
4.9 | 3.0 | 1.4 | 0.2 | 0
4.7 | 3.2 | 1.3 | 0.2 | 0
... | ... | ... | ... | ...
7.0 | 3.2 | 4.7 | 1.4 | 1
6.4 | 3.2 | 4.5 | 1.5 | 1
6.9 | 3.1 | 4.9 | 1.5 | 1
... | ... | ... | ... | ...
6.5 | 3.0 | 5.2 | 2.0 | 2
6.2 | 3.4 | 5.4 | 2.3 | 2
5.9 | 3.0 | 5.1 | 1.8 | 2

For this tutorial, the Iris data has been randomized and split into two separate
CSVs:

对于本教程, Iris数据已被随机分为两个独立的CSV：

*   A training set of 120 samples
    ([iris_training.csv](http://download.tensorflow.org/data/iris_training.csv))
*   A test set of 30 samples
    ([iris_test.csv](http://download.tensorflow.org/data/iris_test.csv)).

To get started, first import all the necessary modules, and define where to
download and store the dataset:

要开始, 首先导入所有必要的模块, 并定义下载和存储数据集的位置：

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
```

Then, if the training and test sets aren't already stored locally, download
them.

然后, 如果培训和测试集尚未存储在本地, 请下载.

```python
if not os.path.exists(IRIS_TRAINING):
  raw = urllib.urlopen(IRIS_TRAINING_URL).read()
  with open(IRIS_TRAINING,'w') as f:
    f.write(raw)

if not os.path.exists(IRIS_TEST):
  raw = urllib.urlopen(IRIS_TEST_URL).read()
  with open(IRIS_TEST,'w') as f:
    f.write(raw)
```

Next, load the training and test sets into `Dataset`s using the
[`load_csv_with_header()`](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/datasets/base.py)
method in `learn.datasets.base`. The `load_csv_with_header()` method takes three
required arguments:

*   `filename`, which takes the filepath to the CSV file
*   `target_dtype`, which takes the
    [`numpy` datatype](http://docs.scipy.org/doc/numpy/user/basics.types.html)
    of the dataset's target value.
*   `features_dtype`, which takes the
    [`numpy` datatype](http://docs.scipy.org/doc/numpy/user/basics.types.html)
    of the dataset's feature values.

接下来, 使用`learn.datasets.base`中的`load_csv_with_header()`方法将训练和测试集加载到Datasets中. `load_csv_with_header()`方法需要三个必需的参数：

- `filename`, 将文件路径引入CSV文件
- `target_dtype`, 它指定数据集目标值的numpy数据类型.
- `features_dtype`, 它指定数据集的特征值的numpy数据类型.


Here, the target (the value you're training the model to predict) is flower
species, which is an integer from 0&ndash;2, so the appropriate `numpy` datatype
is `np.int`:

在这里, 目标(你正在训练模型预测的值)是花种, 它是0-2的整数, 所以适当的numpy数据类型是`np.int`：

```python
# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)
```

`Dataset`s in tf.contrib.learn are
[named tuples](https://docs.python.org/2/library/collections.html#collections.namedtuple);
you can access feature data and target values via the `data` and `target`
fields. Here, `training_set.data` and `training_set.target` contain the feature
data and target values for the training set, respectively, and `test_set.data`
and `test_set.target` contain feature data and target values for the test set.

`tf.contrib.learn`中的数据集被命名为元组; 您可以通过`data `和`target`字段访问特征数据和目标值. 这里, `training_set.data`和`training_set.target`分别包含训练集的特征数据和目标值, `test_set.data`和`test_set.target`包含测试集的要素数据和目标值.

Later on, in
["Fit the DNNClassifier to the Iris Training Data,"](#fit-dnnclassifier)
you'll use `training_set.data` and
`training_set.target` to train your model, and in
["Evaluate Model Accuracy,"](#evaluate-accuracy) you'll use `test_set.data` and
`test_set.target`. But first, you'll construct your model in the next section.

稍后, 在“将DNNC分类器安装到Iris训练数据”中, 您将使用`training_set.data`和`training_set.target`来训练您的模型, 并在“评估模型精度”中使用`test_set.data`和`test_set.target`. 但首先, 您将在下一节中构建您的模型.

## Construct a Deep Neural Network Classifier

tf.estimator offers a variety of predefined models, called `Estimator`s, which
you can use "out of the box" to run training and evaluation operations on your
data.
Here, you'll configure a Deep Neural Network Classifier model to fit the Iris
data. Using tf.estimator, you can instantiate your
@{tf.estimator.DNNClassifier} with just a couple lines of code:

`tf.estimator`提供了各种预定义的模型, 称为`Estimators`, 您可以使用“开箱即用”来对数据运行培训和评估操作. 在这里, 您将配置深层神经网络分类器模型以适应Iris数据. 使用tf.estimator, 您可以使用几行代码实例化`tf.estimator.DNNClassifier`：

```python
# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 20, 10],
                                        n_classes=3,
                                        model_dir="/tmp/iris_model")
```

The code above first defines the model's feature columns, which specify the data
type for the features in the data set. All the feature data is continuous, so
`tf.feature_column.numeric_column` is the appropriate function to use to
construct the feature columns. There are four features in the data set (sepal
width, sepal height, petal width, and petal height), so accordingly `shape`
must be set to `[4]` to hold all the data.

上面的代码首先定义了模型的特征列, 它们指定数据集中的features的数据类型. 所有的特征数据是连续的, 所以`tf.contrib.layers.real_valued_column`是用来构造特征列的适当函数. 数据集中有四个特征(萼片宽度, 萼片高度, 花瓣宽度和花瓣高度), 因此尺寸必须设置为4以保存所有数据.

Then, the code creates a `DNNClassifier` model using the following arguments:

*   `feature_columns=feature_columns`. The set of feature columns defined above.
*   `hidden_units=[10, 20, 10]`. Three
    [hidden layers](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw),
    containing 10, 20, and 10 neurons, respectively.
*   `n_classes=3`. Three target classes, representing the three Iris species.
*   `model_dir=/tmp/iris_model`. The directory in which TensorFlow will save
    checkpoint data during model training. For more on logging and monitoring
    with TensorFlow, see
    @{$monitors$Logging and Monitoring Basics with tf.estimator}.

然后, 代码使用以下参数创建一个DNNClassifier模型：

- `feature_columns= feature_columns`. 上面定义的特征列集合.
- `hidden_units = [10, 20, 10]`. 三个隐藏层, 分别含有10,20和10个神经元.
- `n_classes= 3`. 三个目标课程, 代表三种虹膜物种.
- `model_dir=/TMP/iris_model`. TensorFlow将在模型训练期间保存检查点数据的目录. 有关使用TensorFlow进行日志记录和监视的更多信息, 请参阅使用tf.estimator进行记录和监视基础知识.

## Describe the training input pipeline {#train-input}

The `tf.estimator` API uses input functions, which create the TensorFlow
operations that generate data for the model.
We can use `tf.estimator.inputs.numpy_input_fn` to produce the input pipeline:

`tf.estimator` API使用输入函数, 它创建为模型生成数据的TensorFlow操作. 在这种情况下, 数据足够小, 可以将其存储在TensorFlow常量中. 以下代码生成最简单的输入管道：

```python
# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)
```

## Fit the DNNClassifier to the Iris Training Data {#fit-dnnclassifier}

Now that you've configured your DNN `classifier` model, you can fit it to the
Iris training data using the @{tf.estimator.Estimator.train$`train`} method.
Pass `train_input_fn` as the `input_fn`, and the number of steps to train
(here, 2000):

现在您已经配置了DNN分类器模型, 您可以使用fit方法将其适用于Iris训练数据. 通过`get_train_inputs`作为`input_fn`, 以及训练的步骤数(这里, 2000)：

```python
# Train model.
classifier.train(input_fn=train_input_fn, steps=2000)
```

The state of the model is preserved in the `classifier`, which means you can
train iteratively if you like. For example, the above is equivalent to the
following:

模型的状态保留在分类器中, 这意味着如果你喜欢, 你可以迭代地训练. 例如, 以上相当于以下内容：

```python
classifier.train(input_fn=train_input_fn, steps=1000)
classifier.train(input_fn=train_input_fn, steps=1000)
```

However, if you're looking to track the model while it trains, you'll likely
want to instead use a TensorFlow @{tf.train.SessionRunHook$`SessionRunHook`}
to perform logging operations. See the tutorial
@{$monitors$Logging and Monitoring Basics with tf.estimator}
for more on this topic.

但是, 如果您希望在训练时跟踪模型, 则可能需要使用TensorFlow监视器来执行日志记录操作. 有关此主题的更多信息, 请参阅“使用tf.estimator记录和监视基础知识”教程.

## Evaluate Model Accuracy {#evaluate-accuracy}

You've trained your `DNNClassifier` model on the Iris training data; now, you
can check its accuracy on the Iris test data using the
@{tf.estimator.Estimator.evaluate$`evaluate`} method. Like `train`,
`evaluate` takes an input function that builds its input pipeline. `evaluate`
returns a `dict`s with the evaluation results. The following code passes the
Iris test data&mdash;`test_set.data` and `test_set.target`&mdash;to `evaluate`
and prints the `accuracy` from the results:

您已经将您的DNNClassifier模型适用于Iris训练数据; 现在, 您可以使用评估方法检查其对Iris测试数据的准确性. 像适合, 评估需要一个构建其输入管道的输入函数. 评估结果返回一个dict. 以下代码通过Iris测试`data-test_set.data`和`test_set.target`来评估和打印结果的准确性：

```python
# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
```

Note: The `num_epochs=1` argument to `numpy_input_fn` is important here.
`test_input_fn` will iterate over the data once, and then raise
`OutOfRangeError`. This error signals the classifier to stop evaluating, so it
will evaluate over the input once.

Note: 评估的步骤参数很重要. 评估正常运行, 直到达到输入的结尾. 这是评估一组文件的理想选择, 但这里使用的常量将永远不会抛出它所期望的OutOfRangeError或StopIteration.

When you run the full script, it will print something close to:

运行结果大约为

```
Test Accuracy: 0.966667
```

Your accuracy result may vary a bit, but should be higher than 90%. Not bad for
a relatively small data set!

您的准确性结果可能有所不同, 但应高于90％. 对于相对较小的数据集来说不错！

## Classify New Samples

Use the estimator's `predict()` method to classify new samples. For example, say
you have these two new flower samples:

使用估计器的predict()方法对新样本进行分类. 例如, 说你有这两个新花：

Sepal Length | Sepal Width | Petal Length | Petal Width
:----------- | :---------- | :----------- | :----------
6.4          | 3.2         | 4.5          | 1.5
5.8          | 3.1         | 5.0          | 1.7

萼片长度 | 萼片宽度 | 花瓣长度 | 花瓣宽度
--- | --- | --- | ---
6.4 | 3.2 | 4.5 | 1.5
5.8 | 3.1 | 5.0 | 1.7

You can predict their species using the `predict()` method. `predict` returns a
generator of dicts, which can easily be converted to a list. The following code
retrieves and prints the class predictions:

您可以使用predict()方法预测其物种. 预测返回一个生成器, 可以轻松地将其转换为列表. 以下代码检索并打印类预测：

```python
# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5],
     [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": new_samples},
    num_epochs=1,
    shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [p["classes"] for p in predictions]

print(
    "New Samples, Class Predictions:    {}\n"
    .format(predicted_classes))
```

Your results should look as follows:

结果应该是

```
New Samples, Class Predictions:    [1 2]
```

The model thus predicts that the first sample is *Iris versicolor*, and the
second sample is *Iris virginica*.

因此, 该模型预测第一个样品是Iris versicolor, 第二个样品是Iris virginica.

## Additional Resources

*   To learn more about using tf.estimator to create linear models, see
    @{$linear$Large-scale Linear Models with TensorFlow}.

*   To build your own Estimator using tf.estimator APIs, check out
    @{$estimators$Creating Estimators in tf.estimator}.

*   To experiment with neural network modeling and visualization in the browser,
    check out [Deep Playground](http://playground.tensorflow.org/).

*   For more advanced tutorials on neural networks, see
    @{$deep_cnn$Convolutional Neural Networks} and @{$recurrent$Recurrent Neural
    Networks}.


- 有关tf.estimator的进一步参考资料, 请参阅官方[API docs](https://www.tensorflow.org/api_guides/python/estimator).
- 要了解有关使用tf.estimator创建线性模型的更多信息, 请参阅[Large-scale Linear Models with TensorFlow](https://www.tensorflow.org/tutorials/linear).
- 要使用tf.estimator API构建自己的Estimator, 请查看[Creating Estimators in tf.estimator](https://www.tensorflow.org/extend/estimators).
- 要在浏览器中进行神经网络建模和可视化实验, 请查看[Deep Playground](http://playground.tensorflow.org/)
- 有关神经网络的更多高级教程, 请参阅[ Convolutional Neural Networks](https://www.tensorflow.org/tutorials/deep_cnn)和[Recurrent Neural Networks](https://www.tensorflow.org/tutorials/recurrent).
