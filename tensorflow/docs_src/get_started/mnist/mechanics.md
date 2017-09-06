# TensorFlow Mechanics 101

Code: [tensorflow/examples/tutorials/mnist/](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/)

GitHub Code: [tensorflow/examples/tutorials/mnist/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist/)

The goal of this tutorial is to show how to use TensorFlow to train and
evaluate a simple feed-forward neural network for handwritten digit
classification using the (classic) MNIST data set.  The intended audience for
this tutorial is experienced machine learning users interested in using
TensorFlow.

本教程的目标是展示如何使用TensorFlow来使用（经典的）MNIST数据集来训练和评估一个简单的前馈神经网络进行手写数字分类。 本教程的目标读者是有兴趣使用TensorFlow的经验丰富的机器学习用户。

These tutorials are not intended for teaching Machine Learning in general.

这些教程一般不用于教学机器学习。

Please ensure you have followed the instructions to
@{$install$install TensorFlow}.

请确保您已按照说明安装TensorFlow。

## Tutorial Files

This tutorial references the following files:

教程参照以下文件

File | Purpose
--- | ---
[`mnist.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist.py) | The code to build a fully-connected MNIST model.
[`fully_connected_feed.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/fully_connected_feed.py) | The main code to train the built MNIST model against the downloaded dataset using a feed dictionary.

文件 | 目的
-- | --
[`mnist.py`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist/mnist.py) | 建立全连接的MNIST模型代码
[`fully_connected_feed.py`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist/fully_connected_feed.py) | 使用Feed字典对下载的数据集训练构建的MNIST模型的主要代码

Simply run the `fully_connected_feed.py` file directly to start training:

运行`fully_connected_feed.py`来开始训练

```bash
python fully_connected_feed.py
```

## Prepare the Data

MNIST is a classic problem in machine learning. The problem is to look at
greyscale 28x28 pixel images of handwritten digits and determine which digit
the image represents, for all the digits from zero to nine.

MNIST是机器学习中的一个经典问题。 问题是查看手写数字的灰度28x28像素图像，并确定图像表示的数字，从0到9的所有数字。

![MNIST Digits](https://www.tensorflow.org/images/mnist_digits.png "MNIST Digits")

For more information, refer to [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)
or [Chris Olah's visualizations of MNIST](http://colah.github.io/posts/2014-10-Visualizing-MNIST/).

更多信息，参照[Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)或者[Chris Olah's visualizations of MNIST.](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)。

### Download

At the top of the `run_training()` method, the `input_data.read_data_sets()`
function will ensure that the correct data has been downloaded to your local
training folder and then unpack that data to return a dictionary of `DataSet`
instances.

在`run_training()`方法的顶部，`input_data.read_data_sets()`函数将确保将正确的数据下载到本地培训文件夹，然后解压缩该数据以返回`DataSet`实例的字典。

```python
data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
```

**NOTE**: The `fake_data` flag is used for unit-testing purposes and may be
safely ignored by the reader.

**NOTE**:fake_data标志用于单元测试，读取器可能会被忽略。

Dataset | Purpose
--- | ---
`data_sets.train` | 55000 images and labels, for primary training.
`data_sets.validation` | 5000 images and labels, for iterative validation of training accuracy.
`data_sets.test` | 10000 images and labels, for final testing of trained accuracy.

数据集 | 目的
-- | --
data_sets.train | 55000张图片和标签，用于初级训练。
data_sets.validation | 5000张图像和标签，用于训练准确性的迭代验证。
data_sets.test | 10000张图像和标签，用于最终测试训练的精度。

### Inputs and Placeholders

The `placeholder_inputs()` function creates two @{tf.placeholder}
ops that define the shape of the inputs, including the `batch_size`, to the
rest of the graph and into which the actual training examples will be fed.

`placeholder_inputs()`函数创建两个`tf.placeholder`操作，它们将输入的形状（包括batch_size）定义到图形的其余部分，并将实际的训练示例进行输入。

```python
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                       mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
```

Further down, in the training loop, the full image and label datasets are
sliced to fit the `batch_size` for each step, matched with these placeholder
ops, and then passed into the `sess.run()` function using the `feed_dict`
parameter.

接下来，在训练循环中，将完整图像和标签数据集切片，以适应每个步骤的`batch_size`，与这些占位符操作相匹配，然后使用`feed_dict`参数传递给`sess.run()`函数。

## Build the Graph

After creating placeholders for the data, the graph is built from the
`mnist.py` file according to a 3-stage pattern: `inference()`, `loss()`, and
`training()`.

1.  `inference()` - Builds the graph as far as required for running
the network forward to make predictions.
1.  `loss()` - Adds to the inference graph the ops required to generate
loss.
1.  `training()` - Adds to the loss graph the ops required to compute
and apply gradients.

在为数据创建占位符后，该图根据3阶段模式从`mnist.py`文件构建：`inference()`，`loss()`和`training()`。

1. `inference()` - 构建图表，直到运行网络前进来进行预测。
1. `loss()` - 将推理图添加到产生损失所需的ops。
1. `training()` -  将损失图添加到计算和应用渐变所需的操作。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/mnist_subgraph.png">
</div>

### Inference

The `inference()` function builds the graph as far as needed to
return the tensor that would contain the output predictions.

`inference()`函数根据需要构建图形，以返回包含输出预测的tensor。

It takes the images placeholder as input and builds on top
of it a pair of fully connected layers with [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation followed by a ten
node linear layer specifying the output logits.

它将图像占位符作为输入，并在其顶部构建一对完全连接的层，[ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))激活后跟一个指定输出逻辑的十个节点线性层。

Each layer is created beneath a unique @{tf.name_scope}
that acts as a prefix to the items created within that scope.

每个层都创建在唯一的[`tf.name_scope`](https://www.tensorflow.org/api_docs/python/tf/name_scope)之下，该`tf.name_scope`用作在该范围内创建的项目的前缀。

```python
with tf.name_scope('hidden1'):
```

Within the defined scope, the weights and biases to be used by each of these
layers are generated into @{tf.Variable}
instances, with their desired shapes:

在定义的范围内，每个这些层要使用的权重和偏差被生成为[tf.Variable](https://www.tensorflow.org/api_docs/python/tf/Variable)实例，并具有所需的形状：

```python
weights = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    name='weights')
biases = tf.Variable(tf.zeros([hidden1_units]),
                     name='biases')
```

When, for instance, these are created under the `hidden1` scope, the unique
name given to the weights variable would be "`hidden1/weights`".

例如，当这些在`hidden1`范围下创建时，给予权重变量的唯一名称将是"`hidden1/weights`"。

Each variable is given initializer ops as part of their construction.

每个变量都被赋予了初始化操作作为其构造的一部分。

In this most common case, the weights are initialized with the
@{tf.truncated_normal}
and given their shape of a 2-D tensor with
the first dim representing the number of units in the layer from which the
weights connect and the second dim representing the number of
units in the layer to which the weights connect.  For the first layer, named
`hidden1`, the dimensions are `[IMAGE_PIXELS, hidden1_units]` because the
weights are connecting the image inputs to the hidden1 layer.  The
`tf.truncated_normal` initializer generates a random distribution with a given
mean and standard deviation.

在这种最常见的情况下，权重用[`tf.truncated_normal`](https://www.tensorflow.org/api_docs/python/tf/truncated_normal)初始化，并给出它们的`2-D`tensor的形状，其中第一个dim表示权重连接的层中的单元数，第二个dim表示权重连接到的层中的单位。对于名为`hidden1`的第一层，尺寸为`[IMAGE_PIXELS，hidden1_units]`，因为权重将图像输入连接到hidden1图层。 `tf.truncated_normal`初始化器生成具有给定平均值和标准偏差的随机分布。

Then the biases are initialized with @{tf.zeros}
to ensure they start with all zero values, and their shape is simply the number
of units in the layer to which they connect.

然后，使用[`tf.zeros`](https://www.tensorflow.org/api_docs/python/tf/zeros)初始化偏移，以确保它们以所有零值开始，并且它们的形状只是它们连接到的图层中的单位数。

The graph's three primary ops -- two @{tf.nn.relu}
ops wrapping @{tf.matmul}
for the hidden layers and one extra `tf.matmul` for the logits -- are then
created, each in turn, with separate `tf.Variable` instances connected to each
of the input placeholders or the output tensors of the previous layer.

图形的三个主要操作 - 两个[`tf.nn.relu`](https://www.tensorflow.org/api_docs/python/tf/nn/relu)操作包含[`tf.matmul`](https://www.tensorflow.org/api_docs/python/tf/matmul)的隐藏层和一个额外的`tf.matmul`用于逻辑 - 然后，依次创建，单独的`tf.Variable`实例连接到每个的输入占位符或上一层的输出张量。

```python
hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
```

```python
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
```

```python
logits = tf.matmul(hidden2, weights) + biases
```

Finally, the `logits` tensor that will contain the output is returned.

最后，包含输出的`logits` tensor将被返回。

### Loss

The `loss()` function further builds the graph by adding the required loss
ops.

`loss()`函数通过添加所需的损失操作来进一步构建图形。

First, the values from the `labels_placeholder` are converted to 64-bit integers. Then, a @{tf.nn.sparse_softmax_cross_entropy_with_logits} op is added to automatically produce 1-hot labels from the `labels_placeholder` and compare the output logits from the `inference()` function with those 1-hot labels.

首先，将`labels_placeholder`的值转换为64位整数。 然后，添加一个[`tf.nn.sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits)操作，以自动从`labels_placeholder`产生one-hot标签，并将来自`inference()`函数的输出逻辑与这些one-hot标签进行比较。

```python
labels = tf.to_int64(labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels, logits=logits, name='xentropy')
```

It then uses @{tf.reduce_mean}
to average the cross entropy values across the batch dimension (the first
dimension) as the total loss.

然后使用[`tf.reduce_mean'](https://www.tensorflow.org/api_docs/python/tf/reduce_mean)将批量维度（第一维）的交叉熵值作为总损耗进行平均。

```python
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
```

And the tensor that will then contain the loss value is returned.

包含损失值的tensor将被返回。

> Note: Cross-entropy is an idea from information theory that allows us
> to describe how bad it is to believe the predictions of the neural network,
> given what is actually true. For more information, read the blog post Visual
> Information Theory (http://colah.github.io/posts/2015-09-Visual-Information/)

> 注意：交叉熵是信息理论的一个想法，可以让我们描述相信神经网络的预测是有多糟糕的，因为实际上是真的。 有关更多信息，请阅读博客文章Visual Information Theory(http://colah.github.io/posts/2015-09-Visual-Information/)

### Training

The `training()` function adds the operations needed to minimize the loss via
[Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent).

`training()`函数通过[`Gradient Descent'](https://en.wikipedia.org/wiki/Gradient_descent)添加最小化损失所需的操作。

Firstly, it takes the loss tensor from the `loss()` function and hands it to a
@{tf.summary.scalar},
an op for generating summary values into the events file when used with a
@{tf.summary.FileWriter} (see below).  In this case, it will emit the snapshot value of
the loss every time the summaries are written out.

首先，它从`loss()`)函数中获取损失tensor，并将其传递给[`tf.summary.scalar'](https://www.tensorflow.org/api_docs/python/tf/summary/scalar)， 该函数用于在与[`tf.summary.FileWriter'](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter)一起使用时将事件文件中的汇总值生成（见下文）。 在这种情况下，每当生成摘要时，它将发出损失的快照值。

```python
tf.summary.scalar('loss', loss)
```

Next, we instantiate a @{tf.train.GradientDescentOptimizer}
responsible for applying gradients with the requested learning rate.

接下来，我们实例化一个[`tf.train.GradientDescentOptimizer`](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)，负责应用所需学习率的渐变。

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
```

We then generate a single variable to contain a counter for the global
training step and the @{tf.train.Optimizer.minimize}
op is used to both update the trainable weights in the system and increment the
global step.  This op is, by convention, known as the `train_op` and is what must
be run by a TensorFlow session in order to induce one full step of training
(see below).

然后，我们生成单个变量以包含全局训练步骤的计数器，并且[`tf.train.Optimizer.minimize`](https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#minimize)操作用于更新系统中的可训练权重并增加全局步长。 按惯例，这个操作被称为`train_op`，是由TensorFlow会话运行的，以便引导一个完整的训练步骤（见下文）。

```python
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
```

## Train the Model

Once the graph is built, it can be iteratively trained and evaluated in a loop
controlled by the user code in `fully_connected_feed.py`.

构建图形后，可以在`full_connected_feed.py`中由用户代码控制的循环中进行的迭代训练和评估。

### The Graph

At the top of the `run_training()` function is a python `with` command that
indicates all of the built ops are to be associated with the default
global @{tf.Graph}
instance.

在`run_training（）`函数的顶部是一个python命令，指示所有构建的ops都与默认的全局`tf.Graph`实例相关联。

```python
with tf.Graph().as_default():
```

A `tf.Graph` is a collection of ops that may be executed together as a group.
Most TensorFlow uses will only need to rely on the single default graph.

`tf.Graph`是可以作为一组一起执行的操作的集合。 大多数TensorFlow用途只需要依赖于单个默认图形。

More complicated uses with multiple graphs are possible, but beyond the scope of
this simple tutorial.

更复杂的使用多个图形是可能的，但超出了这个简单教程的范围。

### The Session

Once all of the build preparation has been completed and all of the necessary
ops generated, a @{tf.Session}
is created for running the graph.

一旦所有的构建准备已经完成并且生成了所有必要的操作，则创建一个`tf.Session`来运行图形。

```python
sess = tf.Session()
```

Alternately, a `Session` may be generated into a `with` block for scoping:

或者，可以将会话生成到具有块的范围：

```python
with tf.Session() as sess:
```

The empty parameter to session indicates that this code will attach to
(or create if not yet created) the default local session.

会话的空参数表示此代码将附加到默认本地会话（或创建尚未创建）。

Immediately after creating the session, all of the `tf.Variable`
instances are initialized by calling @{tf.Session.run}
on their initialization op.

在创建会话之后，所有的`tf.Variable`实例都通过在初始化操作中调用`tf.Session.run`来初始化。

```python
init = tf.global_variables_initializer()
sess.run(init)
```

The @{tf.Session.run}
method will run the complete subset of the graph that
corresponds to the op(s) passed as parameters.  In this first call, the `init`
op is a @{tf.group}
that contains only the initializers for the variables.  None of the rest of the
graph is run here; that happens in the training loop below.

[`tf.Session.run`](https://www.tensorflow.org/api_docs/python/tf/Session#run)方法将运行与作为参数传递的操作对应的图形的完整子集。 在这第一个调用中，`init` op是一个[`tf.group`](https://www.tensorflow.org/api_docs/python/tf/group)，它只包含变量的初始化器。 图的其余部分都不在这里运行; 这在下面的训练循环中发生。

### Train Loop

After initializing the variables with the session, training may begin.

在会话初始化变量后，可以开始训练。

The user code controls the training per step, and the simplest loop that
can do useful training is:

用户代码控制每一步的训练，最简单的循环可以做有用的训练是：

```python
for step in xrange(FLAGS.max_steps):
    sess.run(train_op)
```

However, this tutorial is slightly more complicated in that it must also slice
up the input data for each step to match the previously generated placeholders.

但是，本教程稍微复杂一些，因为它还必须分割每个步骤的输入数据，以匹配先前生成的占位符。

#### Feed the Graph

For each step, the code will generate a feed dictionary that will contain the
set of examples on which to train for the step, keyed by the placeholder
ops they represent.

对于每个步骤，代码将生成一个feed词典，其中将包含一组示例，用于训练该步骤，由其所代表的占位符操作键入。

In the `fill_feed_dict()` function, the given `DataSet` is queried for its next
`batch_size` set of images and labels, and tensors matching the placeholders are
filled containing the next images and labels.

在`fill_feed_dict（）`函数中，查询给定的DataSet用于其下一个batch_size图像和标签集，填充与占位符匹配的tensor，其中包含下一个图像和标签。

```python
images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                               FLAGS.fake_data)
```

A python dictionary object is then generated with the placeholders as keys and
the representative feed tensors as values.

然后生成一个python字典对象，占位符作为键，代表性的Feed张量作为值。

```python
feed_dict = {
    images_placeholder: images_feed,
    labels_placeholder: labels_feed,
}
```

This is passed into the `sess.run()` function's `feed_dict` parameter to provide
the input examples for this step of training.

这被传递给`sess.run（）`函数feed_dict参数，以提供此步骤的培训的输入示例。

#### Check the Status

The code specifies two values to fetch in its run call: `[train_op, loss]`.

该代码指定在运行调用中获取的两个值：`[train_op，loss]`。

```python
for step in xrange(FLAGS.max_steps):
    feed_dict = fill_feed_dict(data_sets.train,
                               images_placeholder,
                               labels_placeholder)
    _, loss_value = sess.run([train_op, loss],
                             feed_dict=feed_dict)
```

Because there are two values to fetch, `sess.run()` returns a tuple with two
items.  Each `Tensor` in the list of values to fetch corresponds to a numpy
array in the returned tuple, filled with the value of that tensor during this
step of training. Since `train_op` is an `Operation` with no output value, the
corresponding element in the returned tuple is `None` and, thus,
discarded. However, the value of the `loss` tensor may become NaN if the model
diverges during training, so we capture this value for logging.

因为要获取两个值，所以`sess.run（）`返回一个包含两个项的元组tuple。 在提取的值列表中的每个Tensor对应于返回的元组中的numpy数组，在该训练步骤中填充该张量的值。 由于`train_op`是没有输出值的操作，返回的元组中的相应元素为None，因此被丢弃。 然而，如果模型在训练期间发生分歧，则损失张量的值可能变为`NaN`，因此我们捕获该值用于记录。

Assuming that the training runs fine without NaNs, the training loop also
prints a simple status text every 100 steps to let the user know the state of
training.

假设没有`NaNs`的训练运行良好，训练循环也会每100个步骤打印简单的状态文本，让用户知道训练状态。

```python
if step % 100 == 0:
    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
```

#### Visualize the Status

In order to emit the events files used by @{$summaries_and_tensorboard$TensorBoard},
all of the summaries (in this case, only one) are collected into a single Tensor
during the graph building phase.

为了发布[`TensorBoard`](https://www.tensorflow.org/get_started/summaries_and_tensorboard)使用的事件文件，在图形构建阶段，所有的摘要（在这种情况下只有一个）被收集到一个Tensor中。

```python
summary = tf.summary.merge_all()
```

And then after the session is created, a @{tf.summary.FileWriter}
may be instantiated to write the events files, which
contain both the graph itself and the values of the summaries.

然后在创建会话之后，可以将[`tf.summary.FileWriter`](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter)实例化为写入事件文件，其中包含图形本身和摘要的值。

```python
summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
```

Lastly, the events file will be updated with new summary values every time the
`summary` is evaluated and the output passed to the writer's `add_summary()`
function.

最后，每当`summary`被评估时，事件文件将被新的摘要值更新，同时输出值将通过`add_summary()`函数进行写入。

```python
summary_str = sess.run(summary, feed_dict=feed_dict)
summary_writer.add_summary(summary_str, step)
```

When the events files are written, TensorBoard may be run against the training
folder to display the values from the summaries.

当写入事件文件时，可以针对训练文件夹运行TensorBoard，以显示摘要中的值。

![MNIST TensorBoard](https://www.tensorflow.org/images/mnist_tensorboard.png "MNIST TensorBoard")

**NOTE**: For more info about how to build and run Tensorboard, please see the accompanying tutorial @{$summaries_and_tensorboard$Tensorboard: Visualizing Learning}.

**注意**：有关如何构建和运行Tensorboard的更多信息，请参阅随附的教程Tensorboard：[可视化学习](https://www.tensorflow.org/get_started/summaries_and_tensorboard)。

#### Save a Checkpoint

In order to emit a checkpoint file that may be used to later restore a model
for further training or evaluation, we instantiate a
@{tf.train.Saver}.

为了发出可能用于稍后恢复模型以进行进一步训练或评估的检查点文件，我们实例化了一个`tf.train.Saver`。

```python
saver = tf.train.Saver()
```

In the training loop, the @{tf.train.Saver.save}
method will periodically be called to write a checkpoint file to the training
directory with the current values of all the trainable variables.

在训练循环中，将定期调用`tf.train.Saver.save`方法，将训练目录中的一个检查点文件写入所有可训练变量的当前值。

```python
saver.save(sess, FLAGS.train_dir, global_step=step)
```

At some later point in the future, training might be resumed by using the
@{tf.train.Saver.restore}
method to reload the model parameters.

在将来的某些稍后的一点，可以通过使用`tf.train.Saver.restore`方法来重新加载模型参数来恢复训练。

```python
saver.restore(sess, FLAGS.train_dir)
```

## Evaluate the Model

Every thousand steps, the code will attempt to evaluate the model against both
the training and test datasets.  The `do_eval()` function is called thrice, for
the training, validation, and test datasets.

每一个步骤，代码将尝试对训练和测试数据集进行评估。 `do_eval（）`函数被调用三次，用于训练，验证和测试数据集。

```python
print('Training Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.train)
print('Validation Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.validation)
print('Test Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.test)
```

> Note that more complicated usage would usually sequester the `data_sets.test`
> to only be checked after significant amounts of hyperparameter tuning.  For
> the sake of a simple little MNIST problem, however, we evaluate against all of
> the data.

> 请注意，更复杂的使用通常会将`data_sets.test`隔离，以便在大量超参数调整后才能进行检查。 然而，为了简单的小MNIST问题，我们对所有数据进行评估。

### Build the Eval Graph

Before entering the training loop, the Eval op should have been built
by calling the `evaluation()` function from `mnist.py` with the same
logits/labels parameters as the `loss()` function.

在进入训练循环之前，Eval op应该是通过调用来自`mnist.py`的`evaluate（）`函数，使用与`loss（）`函数相同的logits / labels参数构建的。

```python
eval_correct = mnist.evaluation(logits, labels_placeholder)
```

The `evaluation()` function simply generates a @{tf.nn.in_top_k}
op that can automatically score each model output as correct if the true label
can be found in the K most-likely predictions.  In this case, we set the value
of K to 1 to only consider a prediction correct if it is for the true label.

`evaluate（）`函数只是生成一个`tf.nn.in_top_k`操作，如果真正的标签可以在K个最可能的预测中找到，那么可以自动对每个模型输出进行评分。 在这种情况下，我们将K的值设置为1，以便仅对真实标签考虑预测是否正确。

```python
eval_correct = tf.nn.in_top_k(logits, labels, 1)
```

### Eval Output

One can then create a loop for filling a `feed_dict` and calling `sess.run()`
against the `eval_correct` op to evaluate the model on the given dataset.

然后可以创建一个填充`feed_dict`的循环，并针对eval_correct op调用`sess.run（）`来评估给定数据集上的模型。

```python
for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
```

The `true_count` variable simply accumulates all of the predictions that the
`in_top_k` op has determined to be correct.  From there, the precision may be
calculated from simply dividing by the total number of examples.

`true_count`变量简单地累积了`in_top_k` op已经确定为正确的所有预测。 从那里可以从简单地除以实例的总数来计算精度。

```python
precision = true_count / num_examples
print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
      (num_examples, true_count, precision))
```
