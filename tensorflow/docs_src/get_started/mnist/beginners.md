# MNIST For ML Beginners

*This tutorial is intended for readers who are new to both machine learning and
TensorFlow. If you already know what MNIST is, and what softmax (multinomial
logistic) regression is, you might prefer this
@{$pros$faster paced tutorial}.  Be sure to
@{$install$install TensorFlow} before starting either
tutorial.*

本教程适用于新来的机器学习和TensorFlow的读者。 如果你已经知道MNIST是什么，以及什么softmax（多项式逻辑）回归，那么你可能更喜欢这个更快节奏的教程。 在开始任一教程之前，请务必安装TensorFlow。

When one learns how to program, there's a tradition that the first thing you do
is print "Hello World." Just like programming has Hello World, machine learning
has MNIST.

MNIST is a simple computer vision dataset. It consists of images of handwritten
digits like these:

当学习如何编程时，有一个传统，你所做的第一件事是打印“Hello World”。 就像编程有Hello World，机器学习有MNIST。

MNIST是一个简单的计算机视觉数据集。 它由以下手写数字的图像组成：

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/MNIST.png">
</div>

It also includes labels for each image, telling us which digit it is. For
example, the labels for the above images are 5, 0, 4, and 1.

它还包括每个图像的标签，告诉我们哪个数字。 例如，上述图像的标签是5,0,4和1。

In this tutorial, we're going to train a model to look at images and predict
what digits they are. Our goal isn't to train a really elaborate model that
achieves state-of-the-art performance -- although we'll give you code to do that
later! -- but rather to dip a toe into using TensorFlow. As such, we're going
to start with a very simple model, called a Softmax Regression.

The actual code for this tutorial is very short, and all the interesting
stuff happens in just three lines. However, it is very
important to understand the ideas behind it: both how TensorFlow works and the
core machine learning concepts. Because of this, we are going to very carefully
work through the code.

在本教程中，我们将训练一个模型来查看图像并预测它们的数字。 我们的目标不是训练一个真正精致的模型，而是实现最先进的性能，我们稍后会给你代码！我们将从一个非常简单的模型开始，称为Softmax回归。

本教程的实际代码很短，所有有趣的东西都发生在三行。 然而，了解其背后的想法是非常重要的：TensorFlow如何运作和核心机器学习概念。 因此，我们将非常仔细地编写代码。

## About this tutorial

This tutorial is an explanation, line by line, of what is happening in the
[mnist_softmax.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_softmax.py) code.

本教程将逐行解释[mnist_softmax.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py)代码中发生的情况。

You can use this tutorial in a few different ways, including:

- Copy and paste each code snippet, line by line, into a Python environment as
  you read through the explanations of each line.

- Run the entire `mnist_softmax.py` Python file either before or after reading
  through the explanations, and use this tutorial to understand the lines of
  code that aren't clear to you.

What we will accomplish in this tutorial:

- Learn about the MNIST data and softmax regressions

- Create a function that is a model for recognizing digits, based on looking at
  every pixel in the image

- Use TensorFlow to train the model to recognize digits by having it "look" at
  thousands of examples (and run our first TensorFlow session to do so)

- Check the model's accuracy with our test data

您可以通过以下几种不同的方式使用本教程，其中包括：

- 当您阅读每行的说明时，将每个代码段逐行复制并粘贴到Python环境中。
- 在阅读说明之前或之后运行整个mnist_softmax.py Python文件，并使用本教程来了解不清楚的代码行。

我们将在本教程中完成什么：

- 了解MNIST数据和softmax回归
- 基于查看图像中的每个像素来创建一个用于识别数字的模型的函数
- 使用TensorFlow来训练模型来识别数字，方法是将其“查看”成千上万个示例（并运行我们的第一个TensorFlow session）
- 使用我们的测试数据检查型号的精度

## The MNIST Data

The MNIST data is hosted on
[Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).  If you are copying and
pasting in the code from this tutorial, start here with these two lines of code
which will download and read in the data automatically:

MNIST数据托管在[Yann LeCun的网站](http://yann.lecun.com/exdb/mnist/)上。 如果您在本教程的代码中复制和粘贴，请从这两个代码开始，这两行代码将自动下载和读取数据：

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

The MNIST data is split into three parts: 55,000 data points of training
data (`mnist.train`), 10,000 points of test data (`mnist.test`), and 5,000
points of validation data (`mnist.validation`). This split is very important:
it's essential in machine learning that we have separate data which we don't
learn from so that we can make sure that what we've learned actually
generalizes!

As mentioned earlier, every MNIST data point has two parts: an image of a
handwritten digit and a corresponding label. We'll call the images "x"
and the labels "y". Both the training set and test set contain images and their
corresponding labels; for example the training images are `mnist.train.images`
and the training labels are `mnist.train.labels`.

Each image is 28 pixels by 28 pixels. We can interpret this as a big array of
numbers:

MNIST数据分为三部分：训练数据（mnist.train）55,000个数据点，10,000点测试数据（mnist.test）和5,000点验证数据（mnist.validation）。 这种区分是非常重要的：在机器学习中，我们有不用来学习的独立数据，因此我们可以确保我们所学到的知识在实际上被概括！

如前所述，每个MNIST数据点都有两部分：手写数字的图像和相应的标签。 我们将调用图像“x”和标签“y”。 训练集和测试集都包含图像及其相应的标签; 例如，训练图像是mnist.train.images，训练标签是mnist.train.labels。

每个图像是28像素乘以28像素。 我们可以把它解释为一个大数组：

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/MNIST-Matrix.png">
</div>

We can flatten this array into a vector of 28x28 = 784 numbers. It doesn't
matter how we flatten the array, as long as we're consistent between images.
From this perspective, the MNIST images are just a bunch of points in a
784-dimensional vector space, with a
[very rich structure](https://colah.github.io/posts/2014-10-Visualizing-MNIST/)
(warning: computationally intensive visualizations).

我们可以把这个数组变成一个28×28 = 784数字的向量。 只要我们在图像之间保持一致，那么我们如何平坦化数组并不重要。 从这个角度来看，MNIST图像只是一个784维向量空间中的一个点，结构非常丰富（警告：计算密集的可视化）。

Flattening the data throws away information about the 2D structure of the image.
Isn't that bad? Well, the best computer vision methods do exploit this
structure, and we will in later tutorials. But the simple method we will be
using here, a softmax regression (defined below), won't.

平铺数据会丢弃有关图像2D结构的信息。 那样好吗？ 那么，最好的计算机视觉方法会利用这个结构，我们将在以后的教程中。 但是我们将在这里使用的简单方法，一个softmax回归（下面定义）不会。

The result is that `mnist.train.images` is a tensor (an n-dimensional array)
with a shape of `[55000, 784]`. The first dimension is an index into the list
of images and the second dimension is the index for each pixel in each image.
Each entry in the tensor is a pixel intensity between 0 and 1, for a particular
pixel in a particular image.

结果是mnist.train.images是一个形状为[55000,784]的张量（n维数组）。 第一个维度是图像列表中的索引，第二个维度是每个图像中每个像素的索引。 对于特定图像中的特定像素，张量中的每个条目是0和1之间的像素强度。

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/mnist-train-xs.png">
</div>

Each image in MNIST has a corresponding label, a number between 0 and 9
representing the digit drawn in the image.

MNIST中的每个图像都具有相应的标签，0到9之间的数字表示图像中的数字。

For the purposes of this tutorial, we're going to want our labels as "one-hot
vectors". A one-hot vector is a vector which is 0 in most dimensions, and 1 in a
single dimension. In this case, the \\(n\\)th digit will be represented as a
vector which is 1 in the \\(n\\)th dimension. For example, 3 would be
\\([0,0,0,1,0,0,0,0,0,0]\\).  Consequently, `mnist.train.labels` is a
`[55000, 10]` array of floats.

为了本教程的目的，我们将要将我们的标签称为“one-hot向量”。 一个one-hot向量是大多数维数为0的向量，单个维度为1。 在这种情况下，第n个数字将被表示为在第n维中为1的向量。 例如，3将是[0,0,0,1,0,0,0,0,0,0]。 因此，mnist.train.labels是一个[55000,10]的浮点数组。

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/mnist-train-ys.png">
</div>

We're now ready to actually make our model!

我们现在准备实际制作我们的模型了！

## Softmax Regressions

We know that every image in MNIST is of a handwritten digit between zero and
nine.  So there are only ten possible things that a given image can be. We want
to be able to look at an image and give the probabilities for it being each
digit. For example, our model might look at a picture of a nine and be 80% sure
it's a nine, but give a 5% chance to it being an eight (because of the top loop)
and a bit of probability to all the others because it isn't 100% sure.

我们知道MNIST中的每个图像都是零到九之间的手写数字。所以给定的图像只有十种可能。我们希望能够看到一个图像，并给出它作为每个数字的概率。例如，我们的模型可能会看到一个9的图片，并且有80％的人肯定它是一个9，但是给出5％的机会是8（因为顶级循环），并且有一点概率给所有其他人，因为它不是100%确定的。

This is a classic case where a softmax regression is a natural, simple model.
If you want to assign probabilities to an object being one of several different
things, softmax is the thing to do, because softmax gives us a list of values
between 0 and 1 that add up to 1. Even later on, when we train more sophisticated
models, the final step will be a layer of softmax.

softmax回归是一种自然且简单的经典模型。softmax要将概率分配给几个不同的对象，因为softmax给出了一个0到1之间的值列表，这些值加起来为1。即使在以后，当我们训练更复杂型号，最后一步也将是softmax的一层。

A softmax regression has two steps: first we add up the evidence of our input
being in certain classes, and then we convert that evidence into probabilities.

softmax回归有两个步骤：首先，我们将输入的evidence加在某些确认的类别中，然后将该evidence转换为概率。

To tally up the evidence that a given image is in a particular class, we do a
weighted sum of the pixel intensities. The weight is negative if that pixel
having a high intensity is evidence against the image being in that class, and
positive if it is evidence in favor.

为了统计给定图像在特定类别中的evidence，我们进行像素强度的加权和。如果具有高强度的像素是针对该类中的图像的evidence，那么权重是负的。如果是有利的证据则权重为正。

The following diagram shows the weights one model learned for each of these
classes. Red represents negative weights, while blue represents positive
weights.

下图显示了为每个类别学习的模型的权重。红色代表负权重，而蓝色代表正权重。

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/softmax-weights.png">
</div>

We also add some extra evidence called a bias. Basically, we want to be able
to say that some things are more likely independent of the input. The result is
that the evidence for a class \\(i\\) given an input \\(x\\) is:

我们还增加了一些称为bias的额外evidence。 我们希望能够加一些独立于输入的东西。 最终结果为，给定输入x，是某个类别i的evidence是

$$\text{evidence}_i = \sum_j W_{i,~ j} x_j + b_i$$

where \\(W_i\\) is the weights and \\(b_i\\) is the bias for class \\(i\\),
and \\(j\\) is an index for summing over the pixels in our input image \\(x\\).
We then convert the evidence tallies into our predicted probabilities
\\(y\\) using the "softmax" function:

W\_i是类别i权重，b\_i是类别i偏置，j是对输入图像x中的像素求和的索引。 然后我们使用“softmax”函数将evidence计数转换为我们的预测概率y：

$$y = \text{softmax}(\text{evidence})$$

Here softmax is serving as an "activation" or "link" function, shaping
the output of our linear function into the form we want -- in this case, a
probability distribution over 10 cases.
You can think of it as converting tallies
of evidence into probabilities of our input being in each class.
It's defined as:

这里，softmax用作“激活”或“链接”功能，将我们的线性函数的输出转换为我们想要的形式 - 在这种情况下，概率分布超过了10种。 您可以将其视为将evidence的一切转化为输入图像是某个类别的概率。 它定义为：

$$\text{softmax}(x) = \text{normalize}(\exp(x))$$

If you expand that equation out, you get:

展开等式后变为

$$\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

But it's often more helpful to think of softmax the first way: exponentiating
its inputs and then normalizing them.  The exponentiation means that one more
unit of evidence increases the weight given to any hypothesis multiplicatively.
And conversely, having one less unit of evidence means that a hypothesis gets a
fraction of its earlier weight. No hypothesis ever has zero or negative
weight. Softmax then normalizes these weights, so that they add up to one,
forming a valid probability distribution. (To get more intuition about the
softmax function, check out the
[section](http://neuralnetworksanddeeplearning.com/chap3.html#softmax) on it in
Michael Nielsen's book, complete with an interactive visualization.)

但是通过展开前的公式来考虑softmax通常更有帮助：将其输入指数化，然后对它们进行归一化。 求幂意味着evidence小小的增大都会带来指数级的权重增长。 相反，如果evidence比较小，则以为着该部分的权重只有别人的一部分。 没有假设有零或负重。 Softmax然后对这些权重进行归一化，使得它们加起来等于一，这样可以形成有效的概率分布。 （要获得关于softmax功能的更多直观信息，请查看[Michael Nielsen的书中的部分](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)，并附有交互式可视化文件。）

You can picture our softmax regression as looking something like the following,
although with a lot more \\(x\\)s. For each output, we compute a weighted sum of
the \\(x\\)s, add a bias, and then apply softmax.

您可以将我们的softmax回归图像看起来像以下内容，尽管还有更多x。 对于每个输出，我们计算x的加权和，添加偏差，然后应用softmax。

<div style="width:55%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/softmax-regression-scalargraph.png">
</div>

If we write that out as equations, we get:

写成公式如下

<div style="width:52%; margin-left:25%; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/softmax-regression-scalarequation.png"
   alt="[y1, y2, y3] = softmax(W11*x1 + W12*x2 + W13*x3 + b1,  W21*x1 + W22*x2 + W23*x3 + b2,  W31*x1 + W32*x2 + W33*x3 + b3)">
</div>

We can "vectorize" this procedure, turning it into a matrix multiplication
and vector addition. This is helpful for computational efficiency. (It's also
a useful way to think.)

我们可以“矢量化”这个过程，把它变成矩阵乘法和向量加法。 这有助于计算效率。 （这也是一个有用的思考方式。）

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/softmax-regression-vectorequation.png"
 alt="[y1, y2, y3] = softmax([[W11, W12, W13], [W21, W22, W23], [W31, W32, W33]]*[x1, x2, x3] + [b1, b2, b3])">
</div>

More compactly, we can just write:

更紧凑，我们可以写：

$$y = \text{softmax}(Wx + b)$$

Now let's turn that into something that TensorFlow can use.

现在让我们把它变成TensorFlow可以使用的东西。

## Implementing the Regression


To do efficient numerical computing in Python, we typically use libraries like
[NumPy](http://www.numpy.org) that do expensive operations such as matrix
multiplication outside Python, using highly efficient code implemented in
another language.  Unfortunately, there can still be a lot of overhead from
switching back to Python every operation. This overhead is especially bad if you
want to run computations on GPUs or in a distributed manner, where there can be
a high cost to transferring data.

为了在Python中进行有效的数值计算，我们通常使用像[NumPy](http://www.numpy.org/)这样的库，可以利用另一种语言的高效率代码在Python之外进行诸如矩阵乘法等昂贵的操作。 不幸的是，每次操作切换回Python时，仍然可能会有很多开销。 如果要在GPU上运行计算或以分布式方式运行计算，那么这种开销是特别糟糕的，因为传输数据的成本很高。

TensorFlow also does its heavy lifting outside Python, but it takes things a
step further to avoid this overhead.  Instead of running a single expensive
operation independently from Python, TensorFlow lets us describe a graph of
interacting operations that run entirely outside Python. (Approaches like this
can be seen in a few machine learning libraries.)

TensorFlow也在Python之外做了很大的工作，但它需要进一步的工作来避免这种开销。 TensorFlow不是独立于Python运行单一昂贵的操作，而是可以描述完全在Python之外运行的交互操作的图形。 （这样的方法可以在几台机器学习库中看到。）

To use TensorFlow, first we need to import it.

要使用TensorFlow，首先我们需要导入它。

```python
import tensorflow as tf
```

We describe these interacting operations by manipulating symbolic variables.
Let's create one:

我们通过操纵符号变量来描述这些交互操作。 我们创建一个：

```python
x = tf.placeholder(tf.float32, [None, 784])
```

`x` isn't a specific value. It's a `placeholder`, a value that we'll input when
we ask TensorFlow to run a computation. We want to be able to input any number
of MNIST images, each flattened into a 784-dimensional vector. We represent
this as a 2-D tensor of floating-point numbers, with a shape `[None, 784]`.
(Here `None` means that a dimension can be of any length.)

`x`不是一个特定的值。 这是一个`占位符`，当我们要求TensorFlow运行计算时，我们将输入一个值。 我们希望能够输入任意数量的MNIST图像，每个图像被平铺成784维的向量。 我们将其表示为2-D tensor的浮点数，形状为`[None，784]`。 （这里无意味着尺寸可以是任何长度。）

We also need the weights and biases for our model. We could imagine treating
these like additional inputs, but TensorFlow has an even better way to handle
it: `Variable`.  A `Variable` is a modifiable tensor that lives in TensorFlow's
graph of interacting operations. It can be used and even modified by the
computation. For machine learning applications, one generally has the model
parameters be `Variable`s.

我们还需要模型的权重和偏差。 我们可以想像其他输入一样处理这些信息，但是TensorFlow有一个更好的处理方式：`变量`。 `变量`是一个可修改的tensor，它存在于TensorFlow的交互操作图中。 它可以被计算使用甚至修改。 对于机器学习应用程序，一般通常将模型参数设为`变量`。

```python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

We create these `Variable`s by giving `tf.Variable` the initial value of the
`Variable`: in this case, we initialize both `W` and `b` as tensors full of
zeros. Since we are going to learn `W` and `b`, it doesn't matter very much
what they initially are.

通过给`tf.Variable`函数在创建`变量`的时候进行初始化：在这种情况下，我们初始化`W`和`b`作为充满零的tensor。 由于我们要学习训练`W`和`B`，所以它们最初是多少并没有关系。

Notice that `W` has a shape of [784, 10] because we want to multiply the
784-dimensional image vectors by it to produce 10-dimensional vectors of
evidence for the difference classes. `b` has a shape of [10] so we can add it
to the output.

请注意，`W`具有[784,10]的形状，因为我们要将784维图像向量乘以它，以产生差分类别的10维evidence向量。 `b`具有[10]的形状，所以我们可以将其添加到输出。

We can now implement our model. It only takes one line to define it!

我们现在可以实施我们的模型。 它只需要一行来定义它！

```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

First, we multiply `x` by `W` with the expression `tf.matmul(x, W)`. This is
flipped from when we multiplied them in our equation, where we had \\(Wx\\), as
a small trick to deal with `x` being a 2D tensor with multiple inputs. We then
add `b`, and finally apply `tf.nn.softmax`.

首先，我们将`x`乘以`W`，表达式为`tf.matmul(x，W)`。 当我们在我们的方程中乘以它们时，我们已经转过来了。Wx作为处理x是具有多个输入的2D tensor的小技巧。 然后我们添加`b`，最后应用`tf.nn.softmax`。

That's it. It only took us one line to define our model, after a couple short
lines of setup. That isn't because TensorFlow is designed to make a softmax
regression particularly easy: it's just a very flexible way to describe many
kinds of numerical computations, from machine learning models to physics
simulations. And once defined, our model can be run on different devices:
your computer's CPU, GPUs, and even phones!

经过几个短暂的设置，我们只需要一条线来定义我们的模型。 这并不是因为TensorFlow旨在使softmax回归特别容易：它只是一种非常灵活的方式来描述从机器学习模型到物理模拟的多种数值计算。 一旦定义，我们的模型可以在不同的设备上运行：您的计算机的CPU，GPU，甚至手机！


## Training

In order to train our model, we need to define what it means for the model to be
good. Well, actually, in machine learning we typically define what it means for
a model to be bad. We call this the cost, or the loss, and it represents how far
off our model is from our desired outcome. We try to minimize that error, and
the smaller the error margin, the better our model is.

为了训练我们的模型，我们需要定义什么样的模型是好的。 那么实际上，在机器学习中，我们通常定义一个模型对于坏的意义。 我们称之为成本或损失，它代表了我们的模型与我们所期望的结果有多远。 我们尝试最小化该错误，并且错误边距越小，我们的模型就越好。

One very common, very nice function to determine the loss of a model is called
"cross-entropy." Cross-entropy arises from thinking about information
compressing codes in information theory but it winds up being an important idea
in lots of areas, from gambling to machine learning. It's defined as:

确定模型损失的一个非常常见的非常好的功能称为“交叉熵”。 交叉熵源于对信息理论中的信息压缩代码的思考，它是从赌博到机器学习在很多领域都是一个重要的思想。 它定义为：

$$H_{y'}(y) = -\sum_i y'_i \log(y_i)$$

Where \\(y\\) is our predicted probability distribution, and \\(y'\\) is the true
distribution (the one-hot vector with the digit labels).  In some rough sense, the
cross-entropy is measuring how inefficient our predictions are for describing
the truth. Going into more detail about cross-entropy is beyond the scope of
this tutorial, but it's well worth
[understanding](https://colah.github.io/posts/2015-09-Visual-Information).

其中`y`是我们的预测概率分布，`y'`是真实分布（带有数字标签的one-hot矢量）。 在某种粗略的意义上，交叉熵正在衡量我们的预言是如何无效地描述真相。 关于交叉熵的更多细节超出了本教程的范围，但它是非常值得[理解](http://colah.github.io/posts/2015-09-Visual-Information/)的。

To implement cross-entropy we need to first add a new placeholder to input the
correct answers:

为了实现交叉熵，我们需要先添加一个新的占位符来输入正确答案：

```python
y_ = tf.placeholder(tf.float32, [None, 10])
```

Then we can implement the cross-entropy function, \\(-\sum y'\log(y)\\):

然后我们实现交叉熵函数：

```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```

First, `tf.log` computes the logarithm of each element of `y`. Next, we multiply
each element of `y_` with the corresponding element of `tf.log(y)`. Then
`tf.reduce_sum` adds the elements in the second dimension of y, due to the
`reduction_indices=[1]` parameter. Finally, `tf.reduce_mean` computes the mean
over all the examples in the batch.

首先，`tf.log`计算`y`中每个元素的对数。接下来，我们将`y_`中每个元素乘以`tf.log(y)`中的相应元素。然后`tf.reduce_sum`对`y`中的行向量（第二个维度）进行相加，这是由于`reduce_indices = [1]`。最后，`tf.reduce_mean`计算批次中所有示例的平均值。

Note that in the source code, we don't use this formulation, because it is
numerically unstable.  Instead, we apply
`tf.nn.softmax_cross_entropy_with_logits` on the unnormalized logits (e.g., we
call `softmax_cross_entropy_with_logits` on `tf.matmul(x, W) + b`), because this
more numerically stable function internally computes the softmax activation.  In
your code, consider using `tf.nn.softmax_cross_entropy_with_logits`
instead.

请注意，在源代码中，我们不使用这个公式，因为它在数值上是不稳定的。相反，我们对非规范化逻辑应用使用`tf.nn.softmax_cross_entropy_with_logits`（例如，我们在`tf.matmul(x，W)+ b）`上调用`softmax_cross_entropy_with_logits`，因为这个函数更加数值稳定，其在内部计算softmax激活。在您的代码中，请考虑使用`tf.nn.softmax_cross_entropy_with_logits`。

Now that we know what we want our model to do, it's very easy to have TensorFlow
train it to do so.  Because TensorFlow knows the entire graph of your
computations, it can automatically use the
[backpropagation algorithm](https://colah.github.io/posts/2015-08-Backprop) to
efficiently determine how your variables affect the loss you ask it to
minimize. Then it can apply your choice of optimization algorithm to modify the
variables and reduce the loss.

现在我们知道我们想要我们的模型做什么，很容易让TensorFlow训练它来做到这一点。因为TensorFlow知道您的计算的整个图形，它可以自动使用[反向传播算法](http://colah.github.io/posts/2015-08-Backprop/)来有效地确定您的变量如何影响您要求最小化的损失。那么它可以应用您选择的优化算法来修改变量并减少损失。

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

In this case, we ask TensorFlow to minimize `cross_entropy` using the
[gradient descent algorithm](https://en.wikipedia.org/wiki/Gradient_descent)
with a learning rate of 0.5. Gradient descent is a simple procedure, where
TensorFlow simply shifts each variable a little bit in the direction that
reduces the cost. But TensorFlow also provides
@{$python/train#Optimizers$many other optimization algorithms}:
using one is as simple as tweaking one line.

在这种情况下，我们要求TensorFlow使用学习速率为0.5的[梯度下降算法](https://en.wikipedia.org/wiki/Gradient_descent)来最小化`交叉熵`。 梯度下降是一个简单的过程，其中TensorFlow简单地将每个变量在方向上移动一点，从而降低成本。 但是TensorFlow还提供了许多其他的[优化算法](https://www.tensorflow.org/api_guides/python/train#Optimizers)：使用一个就像调整一行代码一样简单。

What TensorFlow actually does here, behind the scenes, is to add new operations
to your graph which implement backpropagation and gradient descent. Then it
gives you back a single operation which, when run, does a step of gradient
descent training, slightly tweaking your variables to reduce the loss.

TensorFlow在背后实现的操作有添加新的操作到您的图形、实现反向传播和梯度下降。 然后，当它运行时，它返回一个单一的操作，进行梯度下降训练的步骤，稍微调整您的变量以减少损失。


We can now launch the model in an `InteractiveSession`:

我们现在可以在`InteractiveSession`中启动该模型：

```python
sess = tf.InteractiveSession()
```

We first have to create an operation to initialize the variables we created:

我们首先必须创建一个操作来初始化我们创建的变量：

```python
tf.global_variables_initializer().run()
```


Let's train -- we'll run the training step 1000 times!

我们将训练1000次

```python
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

Each step of the loop, we get a "batch" of one hundred random data points from
our training set. We run `train_step` feeding in the batches data to replace
the `placeholder`s.

循环的每一步，我们从训练集中得到一百个随机数据点的“批次”。 我们在批次数据中运行train_step，feed将用来替换占位符。

Using small batches of random data is called stochastic training -- in this
case, stochastic gradient descent. Ideally, we'd like to use all our data for
every step of training because that would give us a better sense of what we
should be doing, but that's expensive. So, instead, we use a different subset
every time. Doing this is cheap and has much of the same benefit.

使用小批次的随机数据称为随机训练 - 在这种情况下，随机梯度下降。 理想情况下，我们希望将所有数据用于培训的每个步骤，因为这样可以让我们更好地了解我们应该做什么，但这很贵。 因此，我们每次使用不同的子集。 这样做是便宜的，有很多同样的好处。



## Evaluating Our Model

How well does our model do?

我们的模型做得如何？

Well, first let's figure out where we predicted the correct label. `tf.argmax`
is an extremely useful function which gives you the index of the highest entry
in a tensor along some axis. For example, `tf.argmax(y,1)` is the label our
model thinks is most likely for each input, while `tf.argmax(y_,1)` is the
correct label. We can use `tf.equal` to check if our prediction matches the
truth.

那么首先我们来弄清楚我们预测的正确标签。 `tf.argmax`是一个非常有用的功能，它给出沿某个轴的tensor中最高条目的索引。 例如，`tf.argmax(y，1)`是我们的模型认为对每个输入最有可能的标签，而`tf.argmax(y_，1)`是正确的标签。 我们可以使用`tf.equal`来检查我们的预测是否符合真相。

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

That gives us a list of booleans. To determine what fraction are correct, we
cast to floating point numbers and then take the mean. For example,
`[True, False, True, True]` would become `[1,0,1,1]` which would become `0.75`.

这给我们一个booleans的列表。 为了确定哪个部分是正确的，我们转换为浮点数，然后取平均值。 例如，`[True，False，True，True]`将变为`[1,0,1,1]`，这将变为0.75。

```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

Finally, we ask for our accuracy on our test data.

最后，我们在测试集上验证我们的准度。

```python
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

This should be about 92%.

输出应为92%。

Is that good? Well, not really. In fact, it's pretty bad. This is because we're
using a very simple model. With some small changes, we can get to 97%. The best
models can get to over 99.7% accuracy! (For more information, have a look at
this
[list of results](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results).)

因为我们使用一个非常简单的模型，所以这个结果很糟糕。 变化一些小东西，我们可以达到97%。 但是最好的型号可以达到99.7％的精度！ （有关详细信息，请查看此[结果列表](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)。）

What matters is that we learned from this model. Still, if you're feeling a bit
down about these results, check out
@{$pros$the next tutorial} where we do a lot
better, and learn how to build more sophisticated models using TensorFlow!

重要的是我们从这个模式中学到了。 不过，如果您对这些结果感到遗憾，请查看下一个教程，我们做得更好，并学习如何使用TensorFlow构建更复杂的模型！
