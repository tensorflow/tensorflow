TF Immediate design (draft)

[[TOC]]

# Goal

Be able to have Torch-like/numpy-like execution which runs commands immediately.

Ie. 

a = tf.constant([[1,1],[1,1]])

b = tf.constant([[2,2],[2,2]])

c = tf.matmul(a, b)    # executes the result

print c                       # prints the result

# Design

## Overview

The idea is to be able to replace

import tensorflow as tf

a = tf.ones((3,3))

<TensorFlow graph construction>

with

import tensorflow.immediate as immediate

tf = immediate.Environment("")

a = tf.ones((3, 3))

<TensorFlow graph construction>

Where <TensorFlow graph construction> is unchanged between the two examples, except that operation in the second example are executed immediately and no "session.run" calls are needed.

Mirroring "tf.Tensor" objects, our framework has “immediate.Tensor” objects which have same semantics as “tf.Tensor” but are associated with concrete data in TensorFlow runtime. This means that in addition to standard operations you can do on tf.Tensor objects, immediate.Tensor objects can be printed, they can be used in Python control flow, and they can be converted to numpy arrays.

For instance you can do the following

tf = immediate.Environment("")     # initialize TensorFlow session with default values

a = tf.ones(())*3

counter = tf.numpy_to_tensor(np.zeros(()))    # Python -> TensorFlow transfer 

while (a>0):    # TensorFlow -> Python transfer of boolean value

  print a       # TensorFlow -> Python transfer of int32 value

  a-=1

  counter+=1

print counter    # TensorFlow -> Python transfer of int32 value

Note that "a>0" gets translated into tf.greater(a, 0). This is evaluated to a boolean in TensorFlow runtime, and then transferred back to Python runtime to control the loop.

## immediate.Environment

We implement it as an object to encapsulate "immediate environment". An immediate mode environment keeps track of the following

1. Session in which computations are run

2. Options used to launch "session.run" calls such as tf.RunOptions(tf.FullTracing)

3. Graph used for the computations

4. Cache which maps operations to nodes in the graph that has already been created

It also implements dispatch logic using "__getattr__" operator that mirrors operations available in TensorFlow namespace. For instance the following two will run identical computations in the TensorFlow graph, with the difference that the second line will perform execution immediately.

tf_env = immediate.Environment()

tf.nn.conv2d(<args>)

tf_env.nn.conv2d(<args>)

### immediate.Environment dispatch logic

Environment object must support all operations that are available in "tf" and “tf.nn” namespace. It does this by means of overriding “__getattr__” operator.

Because of the way __getattr__ works, simple rewriting logic can only handle flat level functions like "tf_env.matmul" and not “tf_env.nn.conv2d”. To support the later use-case, we implement a helper class which redirects “tf.nn.conv2d” to “tf.nn__dot__conv2d” so that regular rewriting logic can be applied.

tf_env.op(a, b, attr1=val1, attr2=val2) executes TensorFlow operation "op" with concrete values taken from a, b which are of immediate.Tensor type, and val1, val2, are non-Tensor types that determine op attributes. For reasons of simpler implementation (see OpFactory cache), we require that inputs (ie, immediate.Tensor objects) are always positional arguments, and non-inputs are always keyword arguments. This is already a convention used in regular TensorFlow, however, it is not enforced by the regular TensorFlow wrapper. immediatemmediate wrapper enforces it.

The following happens when "tf_env.op(a, b, attr1=val1, attr2=val2)" is called:

1. We obtain immediate.Operation object op by calling OpFactory with the original parameters

    1. Ie, op = self.op_factory(a, b, attr1=val1, attr2=val2)

2. Run op on inputs a,b to obtain immediate.Tensor or immediate.Tensors and return result

    2. Ie, return  op(a, b)

### Handling Python-only ops

There’s a large number of Python-only ops (172 in tf.* namespace). Some of them are widely used. For instance, "tf.random_uniform" was 6th most frequently used function among one sample of TensorFlow’s interactive uses. Another example is tf.reduce_.* ops. These functions fill the gap in TensorFlow API by introducing special logic to do the operation in terms of TF-native ops.

For instance, tf.clip() is a Python-only op that expresses "clip" logic in terms of “tf.minimum” and “tf.maximum” ops which are TF-native ops.

Because these ops can encompass arbitrary Python logic, it doesn’t make sense to implement the same caching strategy for them as for TF-native ops, and instead we monkey-patch them to use functions from our immediatemmediate environment, instead of deferring to original "tf." namespace.

More specifically for Python-only op "reduce_sum" the flow looks like this:

1. User initializes environment, tf=immediate.Environment()

2. User calls tf.reduce_sum

3. The environment recognizes "reduce_sum" is a Python-only op, and retrieves its function object fun from original “tf” namespace

4. Environment substitutes self itself into fun.__globals__[‘tf’] so that "tf" calls initiated by fun are dispatched back to the environment

5. Environment calls "fun"

## immediate.Tensor

Data object in this model has type immediate.Tensor. An immediate tensor is associated with concrete array of data and is similar in spirit to numpy ndarray or Torch array. It refers to the underlying data by means of a TensorHandle which points to data stored in TensorFlow runtime. It overloads __del__ operator to release the memory in TensorFlow runtime when the object is garbage collected.

An Immediate Tensor object keeps track of the immediate environment it was created in, in order to be able to dispatch overloaded operators to calls in the correct environment.

## immediate.Operation

This is a Python class that simplifies running part of graph corresponding to a particular operation.

immediate.Operation is tied to an instance of immediate.Environment and represents a specific tensorflow operation instantiated in the graph in that environment. It is created by immediate.OpFactory. For instance first time you call "nn.conv2d(<arguments>)"

# create the op

op = immediate.OpFactory(‘matmul’, <arguments>)

# run the op

result = op(<arguments>)

Internally, each Operation creates a piece of graph which accepts TF tensor handles, runs the operation, and returns resulting tensor handles. Inputs into op are "immediate.Tensor" objects and the output is “immediate.Tensor” object as well

## immediate.OpWrapper

This is an object that is an analogous in semantics to TensorFlow wrapper function, such as "tf.sum". When called, it calls the corresponding operation on TensorFlow side using immediate execution.

It is returned by immediate.Environment and is a simple wrapper that creates an op using environment’s OpFactory and calls it.

## immediate.OpFactory

OpFactory is used to create/reuse operations in current environment.

 

For efficiency we want to modify graph only when necessary and reuse parts of graph when possible. For instance, consider following code

 immediate.ones((3,3)) + immediate.ones((3,3))

 immediate.ones((4,4)) + immediate.ones((4,4))

 immediate.ones((3,3), dtype=tf.float32) + immediate.ones((3,3), dtype=tf.float32)

The first addition would modify graph to create an addition OpDef, whereas second addition would reuse it. Third addition would modify the graph again because "dtype" is part of OpDef attributes, hence we need to create two separate OpDef’s to handle int32 and float32 addition.

Consider following code

op = op_factory(‘matmul’, <arguments>)

It examines <arguments> list to figure out what kind of OpDef would be created during this call and checks cache to see if the necessary OpDef was already created in the current environment’s graph:

1. If the necessary OpDef has been created, it returns immediate.Operation object encapsulating this OpDef

2. Otherwise is runs original tensorflow function to create the necessary piece of graph (tf.nn.conv2d(<arguments>) and encapsulates the newly created piece of graph into immediate.Operation object, saving result in the cache

### OpFactory cache

The goal is to reuse previously created OpDef instances to avoid duplicating new ones. Each TensorFlow OpDef instance is keyed by attributes, such as dtype of inputs, so we need a new OpDef for each new set of attributes. When attributes are created in the regular TensorFlow (op_def_factory.py), there are two kinds of attributes -- regular attributes, specified as keyword arguments in the original Python wrapper invocation, and "inferred" attributes which are not explicitly specified. The most common inferred attribute kind is “dtype” which is inferred from input types. Consider following invocation

tf.nn.conv2d(a, b, name=’hi’, use_cudnn_on_gpu=True)

* "a" and “b” are Tensor inputs are are not part of OpDef

* Values for "name" and “use_cudnn_on_gpu” are part of OpDef

* Type of a and b is inferred and is used to initialize "T" attribute of OpDef

To figure out whether a particular OpDef has been created by a previous invocation of a wrapped tf.* call, we check <arguments> to construct a key and retrieve previously constructed op from cache using this key, or construct a new one on cache miss.

To simplify the lookup logic, we don’t reimplement attribute inference logic from op_def_lib.py, but compute the key using surrogate information, with sufficient precision to find correct OpDef in cache. In particular, we don’t distinguish between cases when:

1. all inputs must be the same type, and hence producing only one type attribute in OpDef (T)

2. Inputs may be different types (Scatter op), resulting in multiple type attributes like Tdata, Tidx

In particular we do the following

1. Extract dtypes dtype1,dtype2,...from positional arguments in <arguments> (inferred attributes)

2. Take values val1,val2 of of keyword arguments in <arguments> (explicit attributes)

3. Convert values val1, val2 into hashable type (ie, list to tuple)

4. The key is then (opname, dtype1, dtype2,...., val1, val2, …)

