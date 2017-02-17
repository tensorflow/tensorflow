# specs -- simple specifications for TensorFlow networks

This library implements a simple domain-specific language for specifying
deep neural networks in TensorFlow.

From a high level, there are a set of standard operators and ways of
combining them:

 - operator `|` takes the output from one layer and "pipes" it into the next
 - operator `**` repeats a layer multiple times

Naming conventions:

 - single character names are reserved to users
 - built-in layers are capitalized, not CamelCase (Relu, Fs, etc.)
 - built-in layers that are common are usually two letters (Cr, Fs, etc.)
 - less common operations are longer (Relu, Conc, etc.)
 - temporary names should end in _

Common layers:

Common layers are defined by short, capitalized abbreviations. For layers
that take an activation function (fully_connected, conv2d), the acronym
is a conjunction of a base layer and the activation. For example, `Fs`
represents a fully connected layer followed by a sigmoid, whereas `Ft`
represents a fully connected layer followed by a Tanh.

 - `Fx` = tf.contrib.layers.fully_connected; x = activation function, one of s/t/r/l/m
 - `Cx` = tf.contrib.layers.conv2d; x = activation function, one of s/t/r/l/m
 - `Mp` = tf.contrib.layers.max_pool2d
 - `Ap` = tf.contrib.layers.avg_pool2d
 - `Bn` = tf.contrib.layers.batch_norm

Nonlinearities (suffixes for C/F, so Cs = convolutional layer + sigmoid):

 - `s` = sigmoid
 - `t` = tanh
 - `r` = relu
 - `l` = linear (i.e., None)
 - `m` = softmax

Positional and keyword arguments are the same as for the underlying
slim and TensorFlow functions. Therefore, common usage patterns are:

    Cr(64, [5, 5]) # conv2d with a 5x5 footprint and 64 outputs
    Mp([2, 2])     # max pooling using [2, 2] steps

Explicit nonlinearities:

 - `Relu` = tf.nn.relu
 - `Sig` = tf.nn.sigmoid
 - `Tanh` = tf.nn.tanh
 - `Smax` = tf.nn.softmax

Reshaping:

 - `Flat` = slim.flatten
 - `Reshape` = tf.reshape
 - `Squeeze` = tf.squeeze
 - `Expand` = tf.expand_dims

Multidimensional LSTM:

These are intended as alternatives to 2D convolutions.  For sequence models,
there will be other modeling primitives.

 - `Lstm2` = Fun(lstm2d.separable_lstm)  # 2D-to-2D
 - `Lstm2to1` = Fun(lstm2d.reduce_to_sequence)  # 2D-to-1D
 - `Lstm2to0` = Fun(lstm2d.reduce_to_final)  # 2D-to-vector
 - `Clstm2(n, m)` is a `Cl(n, [3,3])` followed by `Lstm2(m)`
 - `Dws(n)` is a depthwise convolution `Cs(n, [1, 1])`

Other:

 - `Id` = identity
 - `Do` = tf.contrib.layers.dropout
 - `Lrn` = tf.nn.local_response_normalization
 - `Unit` = tf.contrib.layers.unit_norm
 - `Conc` is roughly tf.nn.concat

Binding external functions:

 - `External` - import an external function using module path
 - `Import` - import an external function using statements

A network specification is a sequence of `name = expression` Python statements,
with the `net` variable holding the network that is being defined. That is,
your specification must have a statement of the form `net = ...` as its
last statement.

So, a simple MNIST network might look like:

    net = Cr(64, [5, 5]) | Fs(10)

More complicated:

    net = (Cr(64, [5, 5]) | Mp([2, 2])) ** 3 | Fs(10)

With temporary names:

    cmp_ = Cr(64, [5, 5]) | Mp([2, 2])
    net = cmp_ ** 3 | Fs(10)

(You can also separate statements with `;` instead of `\n`)

General model structure:

 - Models are sequences of `name = expression` statements
   in Python syntax.
 - Other kinds of statements are not allowed (with a few
   exceptions, like calling `debug()`)
 - Names should be assigned only once.

These constraints are only partially enforced by the library right
now, but may be strictly enforced in the future.

# More Details

The spec language is intended for rapid experimentation with common
layer types; it's not a replacement for the standard TensorFlow or
slim APIs. If you have some complex layer type or construct that's
difficult to represent in `spec`, you can implement it directly in
Python and then easily make it available as a `spec` operator.

Since partial application with positional arguments can be a little
confusing, you can also specify positional arguments with keywords like
`_1`:

    cr5_ = Cr(_1=[5, 5]); net = cr5_(64) ** 3 | Fs(10)

You can enable debugging by putting `debug()` at the beginning of your network
definition:

    debug(); net = Cr(64, [5, 5]) | Fs(10)

The module is a "combinator library". To make the syntax work nicely
with Python, the `__call__` operator is overloaded to perform partial
application.

To create a network from Python, you just call the following:

      inputs = tf.placeholder(...)
      spec = "net = (Cr(64, [5, 5]) | Mp([2, 2])) ** 3 | Fs(10)"
      outputs = specs.create_net(spec, inputs)

You can pass variable bindings into `create_net`:

      inputs = tf.placeholder(...)
      spec = "net = (Cr(64, [5, 5]) | Mp([2, 2])) ** depth | Fs(10)"
      outputs = specs.create_net(spec, inputs, dict(depth=3))

# Using `specs` in Code

The specs operators are defined in the module `specs_ops`. To facilitate
using the `specs` DSL in your code without namespace pollution, you can
use the `specs.ops` context manager, which will temporarily make the
`specs` operators available in your code:

    import tensorflow as tf
    import numpy.random as npr
    specs = tf.contrib.specs.python

    with specs.ops:
      net = (Cr(64, [2, 2]) | Mp([2, 2])) ** 3 | Flat | Fs(10)
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
    outputs = net.funcall(inputs)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    sess.run([outputs], feed_dict={inputs: npr.uniform(size=(17, 28, 28, 1))})

# Sharing and Variables

You can share variables among subnets by wrapping them with `Shared`:

    f = Shared(Fr(100))
    g = f | f | f | f

This will stack four fully connected ReLU layers, sharing the same
weights and biases.

You can also create variables explicitly:

    v = Var("v")

You can use this to write expressions like this:

    net = Cl(100, 3) + Var("b", shape=[128, 128, 100]))

Note that, under the covers, both the `Cl` operator and the `Var` operator
generate functions that are eventually applied via `funcall` to an input
tensor; the function generated by the `Var` operator ignores its argument
and calls `tf.get_variable` with the supplied arguments.

# Pulling in New Primitives

If you need some special function in your spec language, you can make
it available using `External` or `Import`. The following two statements
are equivalent:

    Sig = External("some_module", "some_op")
    Sig = Import("import tensorflow as tf; f = tf.nn.sigmoid")

You probably will want to use `Import` because TensorFlow contains a
number of imports that look like they are in modules, but they are
actually just values placed in the namespace somehow. The `Import`
function takes an arbitrary Python statement that eventually needs to
assign a value to the variable `f` that is then wrapped up as a function.

# Summaries

There are a number of functions that give you information about the structure
of specs (and other, similarly structured, TensorFlow graphs); the first
number is the number of parameters, followed by the op, and the shape.

    >>> summaries.tf_spec_summary("net = Cr(100, [3,3]) | Flat | Fs(10)",
                                  input_shape=(17, 28, 28, 1))
         0 Placeholder          [17, 28, 28, 1]
      1000 Conv                 [17, 28, 28, 100]
         0 Flatten              [17, 78400]
    784010 fully_connected      [17, 10]
    >>>

# ToDo

More documentation, comments.

The following features are intended to be added soon (names subject to change):

 - add sequence processing layers
 - add named point cuts
 - Seq(a, b, c).add(name=layer).add(name=layer) for explicit seq. structures
 - S2d, D2s (space/depth operators)
 - `Shared(...)` -- variable sharing
 - `Mix(...)` -- weighted convex combination of layer outputs
 - `Lincom(...)` -- weighted linear combination of layer outputs
 - `SameDepth(A)` -- makes output depth same as input
 - summary ops
 - slim's `arg_scope`
 - automatic wrapping of long-name slim layers
 - depth-to-space, etc.

Eventually, there may be a similar spec language for
input layers and pipelines.
