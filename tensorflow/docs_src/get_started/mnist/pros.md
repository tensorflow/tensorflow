# Deep MNIST for Experts

TensorFlow is a powerful library for doing large-scale numerical computation.
One of the tasks at which it excels is implementing and training deep neural
networks.  In this tutorial we will learn the basic building blocks of a
TensorFlow model while constructing a deep convolutional MNIST classifier.

TensorFlow是进行大规模数值计算的强大库。 其优点之一是实施和训练深层神经网络。 在本教程中，我们将在构造一个深卷积MNIST分类器的同时学习TensorFlow模型的基本构建模块。

*This introduction assumes familiarity with neural networks and the MNIST
dataset. If you don't have
a background with them, check out the
@{$beginners$introduction for beginners}. Be sure to
@{$install$install TensorFlow} before starting.*

这个介绍假设熟悉神经网络和MNIST数据集。 如果你没有他们的背景，请查看初学者的介绍。 开始之前，请务必安装TensorFlow。


## About this tutorial

The first part of this tutorial explains what is happening in the
[mnist_softmax.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_softmax.py)
code, which is a basic implementation of a Tensorflow model.  The second part
shows some ways to improve the accuracy.

本教程的第一部分解释了[mnist_softmax.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py)代码中发生了什么，这是Tensorflow模型的基本实现。 第二部分显示了一些提高准确性的方法。

You can copy and paste each code snippet from this tutorial into a Python
environment to follow along, or you can download the fully implemented deep net
from [mnist_deep.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_deep.py)
.

您可以将本教程中的每个代码段复制并粘贴到Python环境中，或者从[mnist_deep.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py)下载完全实现的深层网络。

What we will accomplish in this tutorial:

- Create a softmax regression function that is a model for recognizing MNIST
  digits, based on looking at every pixel in the image

- Use Tensorflow to train the model to recognize digits by having it "look" at
  thousands of examples (and run our first Tensorflow session to do so)

- Check the model's accuracy with our test data

- Build, train, and test a multilayer convolutional neural network to improve
  the results

我们将在本教程中完成什么：

- 创建一个softmax回归函数，该函数是基于查看图像中每个像素的MNIST数字识别模型
- 使用Tensorflow来训练模型来识别数字，方法是将其“查看”成千上万个示例（并运行我们的第一个Tensorflow会话）
- 使用我们的测试数据检查模型的精度
- 构建，训练和测试多层卷积神经网络以改善结果

## Setup

Before we create our model, we will first load the MNIST dataset, and start a
TensorFlow session.

在创建模型之前，我们首先加载MNIST数据集并且开始一个TensorFlow会话。

### Load MNIST Data

If you are copying and pasting in the code from this tutorial, start here with
these two lines of code which will download and read in the data automatically:

如果您在本教程的代码中复制和粘贴，请从这两个代码开始，这两行代码将自动下载和读取数据：

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

Here `mnist` is a lightweight class which stores the training, validation, and
testing sets as NumPy arrays.  It also provides a function for iterating through
data minibatches, which we will use below.

这里`mnist`是一个轻量级的类，它将训练、验证和测试的数据集存储为NumPy数组。 它还提供了一个迭代数据服务的功能，我们将在下面使用。

### Start TensorFlow InteractiveSession

TensorFlow relies on a highly efficient C++ backend to do its computation. The
connection to this backend is called a session.  The common usage for TensorFlow
programs is to first create a graph and then launch it in a session.

TensorFlow依靠高效的C++后端来进行计算。 与此后端的连接称为会话。 TensorFlow程序的常见用法是首先创建一个图形，然后在会话中启动它。

Here we instead use the convenient `InteractiveSession` class, which makes
TensorFlow more flexible about how you structure your code.  It allows you to
interleave operations which build a
@{$get_started/get_started#the_computational_graph$computation graph}
with ones that run the graph.  This is particularly convenient when working in
interactive contexts like IPython.  If you are not using an
`InteractiveSession`, then you should build the entire computation graph before
starting a session and
@{$get_started/get_started#the_computational_graph$launching the graph}.

这里我们使用方便的`InteractiveSession`类，这使得TensorFlow更加灵活地构建您的代码。 它允许您与运行图形的[计算图](https://www.tensorflow.org/get_started/get_started#the_computational_graph)交织操作。 在像IPython这样的交互式环境中工作时，这是非常方便的。 如果您不使用`InteractiveSession`，则应在开始会话并启动图形之前[构建整个计算图](https://www.tensorflow.org/get_started/get_started#the_computational_graph)。

```python
import tensorflow as tf
sess = tf.InteractiveSession()
```

#### Computation Graph

To do efficient numerical computing in Python, we typically use libraries like
[NumPy](http://www.numpy.org/) that do expensive operations such as matrix
multiplication outside Python, using highly efficient code implemented in
another language.  Unfortunately, there can still be a lot of overhead from
switching back to Python every operation. This overhead is especially bad if you
want to run computations on GPUs or in a distributed manner, where there can be
a high cost to transferring data.

为了在Python中进行有效的数值计算，我们通常使用像[NumPy](http://www.numpy.org/)这样的数据库，使用诸如Python之外的矩阵乘法等昂贵的操作，使用其他语言实现的高效代码。不幸的是，每次操作切换回Python时，仍然可能会有很多开销。如果要在GPU上运行计算或以分布式方式运行计算，那么这种开销是特别糟糕的，因为传输数据的成本很高。

TensorFlow also does its heavy lifting outside Python, but it takes things a
step further to avoid this overhead.  Instead of running a single expensive
operation independently from Python, TensorFlow lets us describe a graph of
interacting operations that run entirely outside Python.  This approach is
similar to that used in Theano or Torch.

TensorFlow也在Python之外做了很大的工作，但它需要进一步的工作来避免这种开销。 TensorFlow不是独立于Python运行单一昂贵的操作，而是可以描述完全在Python之外运行的交互操作的图形。这种方法类似于Theano或Torch所使用的方法。

The role of the Python code is therefore to build this external computation
graph, and to dictate which parts of the computation graph should be run. See
the @{$get_started/get_started#the_computational_graph$Computation Graph}
section of @{$get_started/get_started} for more detail.

因此，Python代码的作用是构建这个外部计算图，并且决定运行计算图的哪些部分。有关更多详细信息，请参阅“使用TensorFlow入门”的“计算图”部分。

## Build a Softmax Regression Model

In this section we will build a softmax regression model with a single linear
layer. In the next section, we will extend this to the case of softmax
regression with a multilayer convolutional network.

在本节中，我们将使用单个线性层构建一个softmax回归模型。 在下一节中，我们将使用多层卷积网络将其扩展到softmax回归的情况。

### Placeholders

We start building the computation graph by creating nodes for the
input images and target output classes.

我们通过创建输入图像和目标输出类的节点来开始构建计算图。

```python
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```

Here `x` and `y_` aren't specific values. Rather, they are each a `placeholder`
-- a value that we'll input when we ask TensorFlow to run a computation.

这里`x`和`y_`不是特定值。 相反，它们都是占位符 - 当我们要求TensorFlow运行计算时，我们将输入一个值。

The input images `x` will consist of a 2d tensor of floating point numbers.
Here we assign it a `shape` of `[None, 784]`, where `784` is the dimensionality
of a single flattened 28 by 28 pixel MNIST image, and `None` indicates that the
first dimension, corresponding to the batch size, can be of any size.  The
target output classes `y_` will also consist of a 2d tensor, where each row is a
one-hot 10-dimensional vector indicating which digit class (zero through nine)
the corresponding MNIST image belongs to.

输入图像`x`将由2d tensor的浮点数组成。 这里我们给它一个`[None，784]`的形状，其中784是单个扁平化的28x28像素MNIST图像的维度，而None表示对应于批量大小的第一维可以是任何大小。 目标输出类别`y_`也将由二维tensor组成，其中每一行都是one-hot的10维向量，指示对应的MNIST图像属于哪个数字类（0到9）。

The `shape` argument to `placeholder` is optional, but it allows TensorFlow
to automatically catch bugs stemming from inconsistent tensor shapes.

占位符的形状参数是可选的，但它允许TensorFlow自动捕获源于不一致的张量形状的错误。

### Variables

We now define the weights `W` and biases `b` for our model. We could imagine
treating these like additional inputs, but TensorFlow has an even better way to
handle them: `Variable`.  A `Variable` is a value that lives in TensorFlow's
computation graph.  It can be used and even modified by the computation. In
machine learning applications, one generally has the model parameters be
`Variable`s.

我们现在定义我们的模型的权重W和偏差b。 我们可以想象，像其他输入一样处理这些信息，但是TensorFlow有一个更好的处理方式：变量。 变量是居住在TensorFlow计算图中的值。 它可以被计算使用甚至修改。 在机器学习应用中，一般通常将模型参数设为变量。

```python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

We pass the initial value for each parameter in the call to `tf.Variable`.  In
this case, we initialize both `W` and `b` as tensors full of zeros. `W` is a
784x10 matrix (because we have 784 input features and 10 outputs) and `b` is a
10-dimensional vector (because we have 10 classes).

我们在调用`tf.Variable`时传递每个参数的初始值。 在这种情况下，我们初始化W和b作为充满零的tensor。 W是784x10矩阵（因为我们有784个输入特征和10个输出），而b是一个10维向量（因为我们有10个类）。

Before `Variable`s can be used within a session, they must be initialized using
that session.  This step takes the initial values (in this case tensors full of
zeros) that have already been specified, and assigns them to each
`Variable`. This can be done for all `Variables` at once:

在会话中可以使用变量之前，必须使用该会话初始化变量。 此步骤将已经指定的初始值（在这种情况下为零），并将其分配给每个变量。 可以一次完成所有变量：

```python
sess.run(tf.global_variables_initializer())
```

### Predicted Class and Loss Function

We can now implement our regression model. It only takes one line!  We multiply
the vectorized input images `x` by the weight matrix `W`, add the bias `b`.

我们现在可以实现我们的回归模型。 只需一行！ 我们将矢量化输入图像x乘以权重矩阵W，添加偏差b。

```python
y = tf.matmul(x,W) + b
```

We can specify a loss function just as easily. Loss indicates how bad the
model's prediction was on a single example; we try to minimize that while
training across all the examples. Here, our loss function is the cross-entropy
between the target and the softmax activation function applied to the model's
prediction.  As in the beginners tutorial, we use the stable formulation:

我们可以轻松地指定一个损失函数。 损失表明模型在一个例子上的预测有多糟糕; 我们尝试尽量减少所有这些例子的训练。 在这里，我们的损失函数是应用于模型预测的目标和softmax激活函数之间的交叉熵。 与初学者教程一样，我们使用稳定的公式：

```python
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```

Note that `tf.nn.softmax_cross_entropy_with_logits` internally applies the
softmax on the model's unnormalized model prediction and sums across all
classes, and `tf.reduce_mean` takes the average over these sums.

请注意，`tf.nn.softmax_cross_entropy_with_logits`内部将softmax应用于模型的非规范化模型预测和所有类别的总和，并且tf.reduce_mean取这些和的平均值。

## Train the Model

Now that we have defined our model and training loss function, it is
straightforward to train using TensorFlow.  Because TensorFlow knows the entire
computation graph, it can use automatic differentiation to find the gradients of
the loss with respect to each of the variables.  TensorFlow has a variety of
@{$python/train#optimizers$built-in optimization algorithms}.
For this example, we will use steepest gradient descent, with a step length of
0.5, to descend the cross entropy.

现在我们已经定义了我们的模型和训练损失函数，直接使用TensorFlow进行训练。 因为TensorFlow知道整个计算图，它可以使用自动差分来找出相对于每个变量的损失的梯度。 TensorFlow有各种内置优化算法。 对于这个例子，我们将使用最大梯度下降，步长为0.5来降低交叉熵。

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

What TensorFlow actually did in that single line was to add new operations to
the computation graph. These operations included ones to compute gradients,
compute parameter update steps, and apply update steps to the parameters.

TensorFlow在这一行代码中实际做的是向计算图添加新的操作。 这些操作包括计算梯度、计算参数更新步骤，以及对参数应用更新步骤。

The returned operation `train_step`, when run, will apply the gradient descent
updates to the parameters. Training the model can therefore be accomplished by
repeatedly running `train_step`.

返回的操作`train_step`在运行时将对参数应用梯度下降更新。 因此，训练模型可以通过重复运行`train_step`来实现。

```python
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```

We load 100 training examples in each training iteration. We then run the
`train_step` operation, using `feed_dict` to replace the `placeholder` tensors
`x` and `y_` with the training examples.  Note that you can replace any tensor
in your computation graph using `feed_dict` -- it's not restricted to just
`placeholder`s.

我们在每个训练迭代中加载了100个训练样例。 然后，我们运行`train_step`操作，使用`feed_dict`替换训练样例的占位符张量`x`和`y_`。 请注意，您可以使用`feed_dict`替换计算图中的任何张量 - 它不仅限于占位符。

### Evaluate the Model

How well did our model do?

我们的模型做得如何？

First we'll figure out where we predicted the correct label. `tf.argmax` is an
extremely useful function which gives you the index of the highest entry in a
tensor along some axis. For example, `tf.argmax(y,1)` is the label our model
thinks is most likely for each input, while `tf.argmax(y_,1)` is the true
label. We can use `tf.equal` to check if our prediction matches the truth.

首先，我们将弄清楚我们预测正确的标签。 `tf.argmax`是一个非常有用的功能，它给出沿某个轴的张量中最高条目的索引。 例如，`tf.argmax(y，1)`是我们的模型认为对于每个输入最有可能的标签，而`tf.argmax(y_，1)`是真实标签。 我们可以使用`tf.equal`来检查我们的预测是否符合真相。

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

That gives us a list of booleans. To determine what fraction are correct, we
cast to floating point numbers and then take the mean. For example,
`[True, False, True, True]` would become `[1,0,1,1]` which would become `0.75`.

这给了我们一个布尔值的列表。 为了确定哪个部分是正确的，我们转换为浮点数，然后取平均值。 例如，`[True，False，True，True]`将变为`[1,0,1,1]`，这将变为`0.75`。

```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

Finally, we can evaluate our accuracy on the test data. This should be about
92% correct.

```python
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

## Build a Multilayer Convolutional Network

Getting 92% accuracy on MNIST is bad. It's almost embarrassingly bad. In this
section, we'll fix that, jumping from a very simple model to something
moderately sophisticated: a small convolutional neural network. This will get us
to around 99.2% accuracy -- not state of the art, but respectable.

在MNIST获得92％的准确性是不好的。 几乎是尴尬的坏。 在本节中，我们将从一个非常简单的模型跳到中等复杂度的一个小卷积神经网络。 这将使我们达到约99.2%的准确度， 这虽然算不上最好的。

Here is a diagram, created with TensorBoard, of the model we will build:

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img src="https://www.tensorflow.org/images/mnist_deep.png">
</div>

### Weight Initialization

To create this model, we're going to need to create a lot of weights and biases.
One should generally initialize weights with a small amount of noise for
symmetry breaking, and to prevent 0 gradients. Since we're using
[ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) neurons, it is
also good practice to initialize them with a slightly positive initial bias to
avoid "dead neurons". Instead of doing this repeatedly while we build the model,
let's create two handy functions to do it for us.

要创建这个模型，我们将需要创建大量的权重和偏差。 通常应该用少量的噪声来初始化权重以进行对称断裂(symmetry breaking)，并且防止0梯度。 由于我们使用[ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))神经元，所以使用略高于0的偏置去初始化也是一个很好的做法，以避免“死神经元”。 而不是在构建模型时反复执行，我们使用两个方便的函数来实现。

```python
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```

### Convolution and Pooling

TensorFlow also gives us a lot of flexibility in convolution and pooling
operations. How do we handle the boundaries? What is our stride size?
In this example, we're always going to choose the vanilla version.
Our convolutions uses a stride of one and are zero padded so that the
output is the same size as the input. Our pooling is plain old max pooling
over 2x2 blocks. To keep our code cleaner, let's also abstract those operations
into functions.

TensorFlow还为卷积和集合操作提供了很大的灵活性。 我们如何处理边界？ 我们的步幅是多少？ 在这个例子中，我们总是选择vanilla版本。 我们的卷积使用步长为1，边距为0，使得输出的大小与输入相同。 我们的pooling是2x2个block的普通max pooling。 为了使代码更清洁，我们还将这些操作抽象为函数。

```python
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```

### First Convolutional Layer

We can now implement our first layer. It will consist of convolution, followed
by max pooling. The convolution will compute 32 features for each 5x5 patch.
Its weight tensor will have a shape of `[5, 5, 1, 32]`. The first two
dimensions are the patch size, the next is the number of input channels, and
the last is the number of output channels. We will also have a bias vector with
a component for each output channel.

我们现在可以实现我们的第一层。 它将包括卷积，其次是max pooling。 卷积将为每个5x5块计算32个特征。 它的权重tensor将具有`[5,5,1,32]`的形状。 前两个维度是block大小，下一个是输入通道的数量，最后一个是输出通道的数量。 我们还将为每个输出通道提供一个偏置矢量和一个分量。

```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

To apply the layer, we first reshape `x` to a 4d tensor, with the second and
third dimensions corresponding to image width and height, and the final
dimension corresponding to the number of color channels.

为了实现这一层，我们首先将`x`转换为`4d tensor`，第二和第三维对应于图像的宽度和高度，最后的尺寸对应于颜色通道的数量。

```python
x_image = tf.reshape(x, [-1, 28, 28, 1])
```

We then convolve `x_image` with the weight tensor, add the
bias, apply the ReLU function, and finally max pool. The `max_pool_2x2` method will
reduce the image size to 14x14.

然后，我们将`x_image`与权重tensor进行卷积，添加偏差，应用ReLU函数，最后使用max pool。 `max_pool_2x2`方法将图像大小减小到14x14。

```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

### Second Convolutional Layer

In order to build a deep network, we stack several layers of this type. The
second layer will have 64 features for each 5x5 patch.

为了构建一个深层次的网络，我们堆叠这种类型的层几次。 第二层将为每个5x5 block提供64个特征。

```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

### Densely Connected Layer

Now that the image size has been reduced to 7x7, we add a fully-connected layer
with 1024 neurons to allow processing on the entire image. We reshape the tensor
from the pooling layer into a batch of vectors,
multiply by a weight matrix, add a bias, and apply a ReLU.

现在图像尺寸已经缩小到7x7，我们添加了一个具有1024个神经元的完全连接的图层，以便对整个图像进行处理。 我们从pooling层对tensor进行reshape成向量集，乘以权重矩阵，添加偏置并且添加ReLU算法。

```python
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

#### Dropout

To reduce overfitting, we will apply [dropout](
https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) before the readout layer.
We create a `placeholder` for the probability that a neuron's output is kept
during dropout. This allows us to turn dropout on during training, and turn it
off during testing.
TensorFlow's `tf.nn.dropout` op automatically handles scaling neuron outputs in
addition to masking them, so dropout just works without any additional
scaling.<sup id="a1">[1](#f1)</sup>

为了减少过度拟合，我们将在readout层之前应用[dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)。 我们创建一个`placeholder`，使得在`dropout`期间神经元可以保持输出。 这样可以让我们在训练过程使用`dropout`，并在测试期间关闭它。 TensorFlow的`tf.nn.dropout`将自动处理缩放神经元输出，除了掩盖它们，所以`dropout`只是一项工作，但是没有任何额外的缩放。

```python
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

### Readout Layer

Finally, we add a layer, just like for the one layer softmax regression
above.

最后，我们添加一层，就像上一层softmax回归一样。

```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
```

### Train and Evaluate the Model

How well does this model do? To train and evaluate it we will use code that is
nearly identical to that for the simple one layer SoftMax network above.

这个模型有多好？ 为了训练和评估它，我们将使用与上述简单的一层SoftMax网络几乎相同的代码。

The differences are that:

- We will replace the steepest gradient descent optimizer with the more
  sophisticated ADAM optimizer.

- We will include the additional parameter `keep_prob` in `feed_dict` to control
  the dropout rate.

- We will add logging to every 100th iteration in the training process.

差异在于：

- 我们将用更复杂的ADAM优化器替换最陡峭的梯度下降优化器。
- 我们将在`feed_dict`中添加额外的参数`keep_prob`来控制辍学率。
- 在培训过程中，我们将每隔100次迭代添加日志记录。

We will also use tf.Session rather than tf.InteractiveSession. This better
separates the process of creating the graph (model specification) and the
process of evaluating the graph (model fitting). It generally makes for cleaner
code. The tf.Session is created within a [`with` block](https://docs.python.org/3/whatsnew/2.6.html#pep-343-the-with-statement)
so that it is automatically destroyed once the block is exited.

我们也将使用tf.Session而不是tf.InteractiveSession。 这更好地分离了创建图形（模型分离）的过程和评估图形（模型拟合）的过程。 它通常会使更清洁的代码。 tf.Session是在一个块中创建的，以便在块被退出后自动销毁。

Feel free to run this code. Be aware that it does 20,000 training iterations
and may take a while (possibly up to half an hour), depending on your processor.

随意运行这段代码。 请注意，它会执行20,000次训练迭代，并且可能需要一段时间（可能长达半小时），具体取决于您的处理器。

```python
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

The final test set accuracy after running this code should be approximately 99.2%.

运行此代码后的最终测试集精度应为大约99.2％。

We have learned how to quickly and easily build, train, and evaluate a
fairly sophisticated deep learning model using TensorFlow.

我们已经学会了如何使用TensorFlow快速轻松地构建，训练和评估一个相当复杂的深度学习模型。

<b id="f1">1</b>: For this small convolutional network, performance is actually nearly identical with and without dropout. Dropout is often very effective at reducing overfitting, but it is most useful when training very large neural networks. [↩](#a1)
