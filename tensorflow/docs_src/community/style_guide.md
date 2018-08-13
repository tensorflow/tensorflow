# TensorFlow Style Guide

This page contains style decisions that both developers and users of TensorFlow
should follow to increase the readability of their code, reduce the number of
errors, and promote consistency.

[TOC]

## Python style

Generally follow
[PEP8 Python style guide](https://www.python.org/dev/peps/pep-0008/),
except for using 2 spaces.


## Python 2 and 3 compatible

* All code needs to be compatible with Python 2 and 3.

* Next lines should be present in all Python files:

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
```

* Use `six` to write compatible code (for example `six.moves.range`).


## Bazel BUILD rules

TensorFlow uses Bazel build system and enforces next requirements:

* Every BUILD file should contain next header:

```
# Description:
#   <...>

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])
```



* For all Python BUILD targets (libraries and tests) add next line:

```
srcs_version = "PY2AND3",
```


## Tensor

* Operations that deal with batches may assume that the first dimension of a Tensor is the batch dimension.

* In most models the *last dimension* is the number of channels.

* Dimensions excluding the first and last usually make up the "space" dimensions: Sequence-length or Image-size.

## Python operations

A *Python operation* is a function that, given input tensors and parameters,
creates a part of the graph and returns output tensors.

* The first arguments should be tensors, followed by basic python parameters.
 The last argument is `name` with a default value of `None`.
 If operation needs to save some `Tensor`s to Graph collections,
 put the arguments with names of the collections right before `name` argument.

* Tensor arguments should be either a single tensor or an iterable of tensors.
 E.g. a "Tensor or list of Tensors" is too broad. See `assert_proper_iterable`.

* Operations that take tensors as arguments should call `convert_to_tensor`
 to convert non-tensor inputs into tensors if they are using C++ operations.
 Note that the arguments are still described as a `Tensor` object
 of a specific dtype in the documentation.

* Each Python operation should have a `name_scope` like below. Pass as
 arguments `name`, a default name of the op, and a list of the input tensors.

* Operations should contain an extensive Python comment with Args and Returns
 declarations that explain both the type and meaning of each value. Possible
 shapes, dtypes, or ranks should be specified in the description.
 @{$documentation$See documentation details}

* For increased usability include an example of usage with inputs / outputs
 of the op in Example section.

Example:

    def my_op(tensor_in, other_tensor_in, my_param, other_param=0.5,
              output_collections=(), name=None):
      """My operation that adds two tensors with given coefficients.

      Args:
        tensor_in: `Tensor`, input tensor.
        other_tensor_in: `Tensor`, same shape as `tensor_in`, other input tensor.
        my_param: `float`, coefficient for `tensor_in`.
        other_param: `float`, coefficient for `other_tensor_in`.
        output_collections: `tuple` of `string`s, name of the collection to
                            collect result of this op.
        name: `string`, name of the operation.

      Returns:
        `Tensor` of same shape as `tensor_in`, sum of input values with coefficients.

      Example:
        >>> my_op([1., 2.], [3., 4.], my_param=0.5, other_param=0.6,
                  output_collections=['MY_OPS'], name='add_t1t2')
        [2.3, 3.4]
      """
      with tf.name_scope(name, "my_op", [tensor_in, other_tensor_in]):
        tensor_in = tf.convert_to_tensor(tensor_in)
        other_tensor_in = tf.convert_to_tensor(other_tensor_in)
        result = my_param * tensor_in + other_param * other_tensor_in
        tf.add_to_collection(output_collections, result)
        return result

Usage:

    output = my_op(t1, t2, my_param=0.5, other_param=0.6,
                   output_collections=['MY_OPS'], name='add_t1t2')


## Layers

Use `tf.keras.layers`, not `tf.layers`.

See `tf.keras.layers` and [the Keras guide](../guide/keras.md#custom_layers) for details on how to sub-class layers.
