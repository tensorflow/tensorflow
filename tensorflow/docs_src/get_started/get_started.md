# Getting Started With TensorFlow

This guide gets you started programming in TensorFlow. Before using this guide,
@{$install$install TensorFlow}. To get the most out of
this guide, you should know the following:

*   How to program in Python.
*   At least a little bit about arrays.
*   Ideally, something about machine learning. However, if you know little or
    nothing about machine learning, then this is still the first guide you
    should read.

在教程之前，你应该知道以下一些知识：
- Python编程
- 数组知识
- 机器学习，若你了解很少或者不了解，本教程已然可以阅读

TensorFlow provides multiple APIs. The lowest level API--TensorFlow Core--
provides you with complete programming control. We recommend TensorFlow Core for
machine learning researchers and others who require fine levels of control over
their models. The higher level APIs are built on top of TensorFlow Core. These
higher level APIs are typically easier to learn and use than TensorFlow Core. In
addition, the higher level APIs make repetitive tasks easier and more consistent
between different users. A high-level API like tf.estimator helps you manage
data sets, estimators, training and inference.

TensorFlow提供多种API。 最低级API - `TensorFlow Core` - 为您提供完整的编程控制。 我们推荐机器学习研究人员和需要对其模型进行良好控制的其他人使用TensorFlow Core。 更高级别的API构建在TensorFlow Core之上。 这些更高级别的API通常比TensorFlow Core更容易学习和使用。 此外，较高级别的API使得在不同用户之间的重复任务更加容易和一致。 像`tf.estimator`这样的高级API可以帮助您管理`datasets`，`estimators`，`training`和`inference`。 请注意，一些高级TensorFlow API（其方法名称包含contrib的）仍在开发中。 在后续的TensorFlow版本中，有可能会有变化或变得过时。

This guide begins with a tutorial on TensorFlow Core. Later, we
demonstrate how to implement the same model in tf.estimator. Knowing
TensorFlow Core principles will give you a great mental model of how things are
working internally when you use the more compact higher level API.

本指南从TensorFlow Core教程开始。 稍后我们演示如何在tf.estimator中实现相同的模型。 了解TensorFlow核心原则将为您提供一个伟大的心理模型，以便您在使用更紧凑的更高级别的API时了解其内部的工作。

# Tensors

The central unit of data in TensorFlow is the **tensor**. A tensor consists of a
set of primitive values shaped into an array of any number of dimensions. A
tensor's **rank** is its number of dimensions. Here are some examples of
tensors:

TensorFlow中的中心数据单位是`tensor`。 tensor由一组原始值组成，这些值可以被组合为任意维度的数组。 tensor的等级是其维数。 以下是张量的一些例子：

```python
3 # a rank 0 tensor; a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

## TensorFlow Core tutorial

### Importing TensorFlow

The canonical import statement for TensorFlow programs is as follows:

TensorFlow程序的规范导入声明如下：

```python
import tensorflow as tf
```
This gives Python access to all of TensorFlow's classes, methods, and symbols.
Most of the documentation assumes you have already done this.

这使Python可以访问TensorFlow的所有类，方法和符号。 大多数文档假定您已经完成了这一import。

### The Computational Graph

You might think of TensorFlow Core programs as consisting of two discrete
sections:

1.  Building the computational graph.
2.  Running the computational graph.

您可能会认为TensorFlow Core程序由两个独立部分组成：

1. 构建计算图。
2. 运行计算图。

A **computational graph** is a series of TensorFlow operations arranged into a
graph of nodes.
Let's build a simple computational graph. Each node takes zero
or more tensors as inputs and produces a tensor as an output. One type of node
is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs
a value it stores internally. We can create two floating point Tensors `node1`
and `node2` as follows:

计算图是排列成节点图的一系列TensorFlow操作。 我们来构建一个简单的计算图。 每个节点采用零个或多个tensor作为输入，并产生tensor作为输出。 一种类型的节点是一个常数。 像所有TensorFlow常数一样，它不需要任何输入，它输出一个内部存储的值。 我们可以创建两个浮点类型tensor node1和node2，如下所示：

```python
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
```

The final print statement produces

最终的打印声明生成

```
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```

Notice that printing the nodes does not output the values `3.0` and `4.0` as you
might expect. Instead, they are nodes that, when evaluated, would produce 3.0
and 4.0, respectively. To actually evaluate the nodes, we must run the
computational graph within a **session**. A session encapsulates the control and
state of the TensorFlow runtime.

请注意，打印节点不会按预期输出值3.0和4.0。 相反，它们是在评估时分别产生3.0和4.0的节点。 要实际评估节点，我们必须在Session中运行计算图。 Session封装了TensorFlow运行时的控制和状态。

The following code creates a `Session` object and then invokes its `run` method
to run enough of the computational graph to evaluate `node1` and `node2`. By
running the computational graph in a session as follows:

以下代码创建一个`Session`对象，然后调用其`run`方法运行足够的计算图来评估node1和node2。 通过在会话中运行计算图如下：

```python
sess = tf.Session()
print(sess.run([node1, node2]))
```

we see the expected values of 3.0 and 4.0:

我们看到3.0和4.0的预期值：

```
[3.0, 4.0]
```

We can build more complicated computations by combining `Tensor` nodes with
operations (Operations are also nodes). For example, we can add our two
constant nodes and produce a new graph as follows:

我们可以通过将Tensor节点与操作（操作也是节点）组合来构建更复杂的计算。 例如，我们可以添加我们的两个常量节点并生成一个新的图，如下所示：

```python
from __future__ import print_function
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
```

The last two print statements produce

最后两个print语句输出

```
node3: Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3): 7.0
```

TensorFlow provides a utility called TensorBoard that can display a picture of
the computational graph. Here is a screenshot showing how TensorBoard
visualizes the graph:

TensorFlow提供了一个名为TensorBoard的实用程序，可以显示计算图的图片。 这是一个屏幕截图，显示TensorBoard如何可视化图形：

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_add.png)

As it stands, this graph is not especially interesting because it always
produces a constant result. A graph can be parameterized to accept external
inputs, known as **placeholders**. A **placeholder** is a promise to provide a
value later.

就这样，这个图并不是特别有趣，因为它总是产生一个恒定的结果。 一个图可以被参数化，用来接受外部输入，这称为占位符。 占位符保证之后会提供值。

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
```

The preceding three lines are a bit like a function or a lambda in which we
define two input parameters (a and b) and then an operation on them. We can
evaluate this graph with multiple inputs by using the feed_dict argument to
the [run method](https://www.tensorflow.org/api_docs/python/tf/Session#run)
to feed concrete values to the placeholders:

前面的三行有点像一个函数或一个lambda，其中我们定义了两个输入参数（a和b），然后对它们进行一个操作。 我们可以使用`feed_dict`参数来指定多个输入来评估图，`feed_dict`的参数将指定这些占位符具体值的Tensors：

```python
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
```
resulting in the output

输出如下

```
7.5
[ 3.  7.]
```

In TensorBoard, the graph looks like this:

TensorBoard中图如下：

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_adder.png)

We can make the computational graph more complex by adding another operation.
For example,

我们可以通过添加另一个操作来使计算图更加复杂。 例如:

```python
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
```
produces the output

输出如下
```
22.5
```

The preceding computational graph would look as follows in TensorBoard:

前面的计算图在TensorBoard中将如下所示：

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_triple.png)

In machine learning we will typically want a model that can take arbitrary
inputs, such as the one above.  To make the model trainable, we need to be able
to modify the graph to get new outputs with the same input.  **Variables** allow
us to add trainable parameters to a graph.  They are constructed with a type and
initial value:

在机器学习中，我们通常会想要一个可以接受任意输入的模型，比如上面的一个。 为了使模型可训练，我们需要能够修改图以获得相同输入却不同的输出。 变量允许我们向图中添加可训练的参数。 它们的构造类型和初始值如下：


```python
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
```

Constants are initialized when you call `tf.constant`, and their value can never
change. By contrast, variables are not initialized when you call `tf.Variable`.
To initialize all the variables in a TensorFlow program, you must explicitly
call a special operation as follows:

调用`tf.constant`时，常量被初始化，它们的值永远不会改变。 相比之下，当您调用`tf.Variable`时，变量不会被初始化。 要初始化TensorFlow程序中的所有变量，必须显式调用特殊操作，如下所示：

```python
init = tf.global_variables_initializer()
sess.run(init)
```
It is important to realize `init` is a handle to the TensorFlow sub-graph that
initializes all the global variables. Until we call `sess.run`, the variables
are uninitialized.

意识到`init`是TensorFlow子图的句柄是重要的，它将初始化所有的全局变量。 在我们调用`sess.run`之前，这些变量是未初始化的。


Since `x` is a placeholder, we can evaluate `linear_model` for several values of
`x` simultaneously as follows:

由于x是占位符，所以我们可以同时评估`linear_model`中x的几个值，如下所示：

```python
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
```
to produce the output

输出如下：
```
[ 0.          0.30000001  0.60000002  0.90000004]
```

We've created a model, but we don't know how good it is yet. To evaluate the
model on training data, we need a `y` placeholder to provide the desired values,
and we need to write a loss function.

A loss function measures how far apart the
current model is from the provided data. We'll use a standard loss model for
linear regression, which sums the squares of the deltas between the current
model and the provided data. `linear_model - y` creates a vector where each
element is the corresponding example's error delta. We call `tf.square` to
square that error. Then, we sum all the squared errors to create a single scalar
that abstracts the error of all examples using `tf.reduce_sum`:

我们创建了一个模型，但是我们不知道它有多好。 为了评估培训数据上的模型，我们需要一个`y`占位符来提供所期望的值，我们需要编写一个损失函数。

损失函数是用来衡量当前模型与提供的数据之间的距离。 我们将使用线性回归的标准损失模型，其将当前模型和提供数据之间的delta进行平方并将平方相加。 `linear_model - y`创建了一个向量，其中每个元素都是相应样例的错误delta。 我们调用`tf.square`来表示该错误。 然后，我们求和所有平方误差，创建一个标量，使用`tf.reduce_sum`抽象出所有示例的错误：

```python
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
```
producing the loss value

损失值如下：
```
23.66
```

We could improve this manually by reassigning the values of `W` and `b` to the
perfect values of -1 and 1. A variable is initialized to the value provided to
`tf.Variable` but can be changed using operations like `tf.assign`. For example,
`W=-1` and `b=1` are the optimal parameters for our model. We can change `W` and
`b` accordingly:

我们可以通过重新分配`W`和`b`的值来手动将损失值提高到-1和1之间的完美值。`tf.Variable`提供了变量的初始值，但可以使用像`tf.assign`这样的操作去进行更改。 例如，`W = -1`和`b = 1`是我们的模型的最优参数。 我们可以相应地改变`W`和`b`：

```python
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
```
The final print shows the loss now is zero.

最后显示当前的损失值为0
```
0.0
```

We guessed the "perfect" values of `W` and `b`, but the whole point of machine
learning is to find the correct model parameters automatically.  We will show
how to accomplish this in the next section.

我们猜测`W`和`b`的“完美”值，但机器学习的全部要点是自动找到正确的模型参数。 我们将在下一节中展示如何完成此项工作。

## tf.train API

A complete discussion of machine learning is out of the scope of this tutorial.
However, TensorFlow provides **optimizers** that slowly change each variable in
order to minimize the loss function. The simplest optimizer is **gradient
descent**. It modifies each variable according to the magnitude of the
derivative of loss with respect to that variable. In general, computing symbolic
derivatives manually is tedious and error-prone. Consequently, TensorFlow can
automatically produce derivatives given only a description of the model using
the function `tf.gradients`. For simplicity, optimizers typically do this
for you. For example,

机器学习的完整讨论超出了本教程的范围。 然而，TensorFlow提供了**优化器optimizers**，可以缓慢地更改每个变量，以便最大程度地减少损失函数。 最简单的优化器是**梯度下降gradient descent**。 它根据相对于该变量的损失导数的大小修改每个变量。 通常，手动计算符号导数是乏味且容易出错的。 因此，TensorFlow可以使用函数`tf.gradients`自动生成仅给出模型描述的导数。 为了简单起见，优化器通常为您做这个。 例如：

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

```python
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
```
results in the final model parameters:

最后的模型参数结果为：
```
[array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]
```

Now we have done actual machine learning!  Although this simple linear
regression model does not require much TensorFlow core code, more complicated
models and methods to feed data into your models necessitate more code. Thus,
TensorFlow provides higher level abstractions for common patterns, structures,
and functionality. We will learn how to use some of these abstractions in the
next section.

现在我们已经做了实际的机器学习！ 虽然这样做简单的线性回归并不需要太多的TensorFlow核心代码，但更复杂的模型和方法需要用更多的代码将数据输入到模型中。 因此，TensorFlow为常见的模式、结构和功能提供了更高级别的抽象。 我们将在下一节中学习如何使用其中的一些抽象。

### Complete program

The completed trainable linear regression model is shown here:

完整的训练线性回归模型的代码如下：

```python
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```
When run, it produces

输出如下：
```
W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```

Notice that the loss is a very small number (very close to zero). If you run 
this program, your loss may not be exactly the same as the aforementioned loss 
because the model is initialized with pseudorandom values.

请注意，损失是非常小的数字（非常接近零）。 如果您运行此程序，您的损失可能与上述损失完全不一样，因为模型是用伪随机值初始化的。

This more complicated program can still be visualized in TensorBoard

这个更加复杂的程序依然可以在TensorBoard中可视化:
![TensorBoard final model visualization](https://www.tensorflow.org/images/getting_started_final.png)

## `tf.estimator`

`tf.estimator` is a high-level TensorFlow library that simplifies the
mechanics of machine learning, including the following:

*   running training loops
*   running evaluation loops
*   managing data sets

tf.estimator defines many common models.

`tf.estimator`是一个高等级的TensorFlow库，其简化了机器学习的机制，包括：

- 运行训练循环
- 运行评估循环
- 管理数据集

`tf.estimator`定义了许多常见的模型。

### Basic usage

Notice how much simpler the linear regression program becomes with
`tf.estimator`:

注意`tf.estimator`使得线性回归程序变得多么简单：

```python
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import tensorflow as tf

# Declare list of features. We only have one numeric feature. There are many
# other types of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```
When run, it produces something like

输出如下：
```
train metrics: {'loss': 7.8912095e-09, 'average_loss': 1.9728024e-09, 'global_step': 1000}
eval metrics: {'loss': 0.010116501, 'average_loss': 0.0025291252, 'global_step': 1000}
```
Notice how our eval data has a higher loss, but it is still close to zero.
That means we are learning properly.

### A custom model

`tf.estimator` does not lock you into its predefined models. Suppose we
wanted to create a custom model that is not built into TensorFlow. We can still
retain the high level abstraction of data set, feeding, training, etc. of
`tf.estimator`. For illustration, we will show how to implement our own
equivalent model to `LinearRegressor` using our knowledge of the lower level
TensorFlow API.

To define a custom model that works with `tf.estimator`, we need to use
`tf.estimator.Estimator`. `tf.estimator.LinearRegressor` is actually
a sub-class of `tf.estimator.Estimator`. Instead of sub-classing
`Estimator`, we simply provide `Estimator` a function `model_fn` that tells
`tf.estimator` how it can evaluate predictions, training steps, and
loss. The code is as follows:

`tf.estimator`不会将您锁定到其预定义的模型中。 假设我们想创建一个没有内置到TensorFlow中的自定义模型。 我们仍然可以保留`tf.estimator`对数据集、feed和训练等的高级抽象。 为了说明，我们将展示如何实现我们自己的与`LinearRegressor`等效的模型，用我们掌握的对 较低级别的TensorFlow API的了解。

要定义与`tf.estimator`一起使用的自定义模型，我们需要使用`tf.estimator.Estimator`。 `tf.estimator.LinearRegressor`实际上是一个`tf.estimator.Estimator`的子类。 我们只是给`Estimator`提供一个函数`model_fn`来告诉`tf.estimator`如何评估预测、训练步骤和计算损失，而非对`Estimator`进行继承。 代码如下：

```python
import numpy as np
import tensorflow as tf

# Declare list of features, we only have one real-valued feature
def model_fn(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # EstimatorSpec connects subgraphs we built to the
  # appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# train
estimator.train(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```
When run, it produces

输出如下：
```
train metrics: {'loss': 1.227995e-11, 'global_step': 1000}
eval metrics: {'loss': 0.01010036, 'global_step': 1000}
```

Notice how the contents of the custom `model_fn()` function are very similar
to our manual model training loop from the lower level API.

请注意，自定义model()函数的内容与下一级API的手动模型训练循环非常相似。

## Next steps

Now you have a working knowledge of the basics of TensorFlow. We have several
more tutorials that you can look at to learn more. If you are a beginner in
machine learning see @{$beginners$MNIST for beginners},
otherwise see @{$pros$Deep MNIST for experts}.

现在您已经了解了TensorFlow的基础知识。 我们还有更多的教程，您可以查看以了解更多。 如果您是机器学习的初学者，请参阅MNIST的初学者，否则请查看深入MNIST的专家。
