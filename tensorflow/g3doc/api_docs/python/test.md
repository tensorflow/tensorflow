<!-- This file is machine generated: DO NOT EDIT! -->

# Testing
[TOC]

## Unit tests

TensorFlow provides a convenience class inheriting from `unittest.TestCase`
which adds methods relevant to TensorFlow tests.  Here is an example:

    import tensorflow as tf


    class SquareTest(tf.test.TestCase):

      def testSquare(self):
        with self.test_session():
          x = tf.square([2, 3])
          self.assertAllEqual(x.eval(), [4, 9])


    if __name__ == '__main__':
      tf.test.main()


`tf.test.TestCase` inherits from `unittest.TestCase` but adds a few additional
methods.  We will document these methods soon.

- - -

### `tf.test.main()` {#main}

Runs all unit tests.



## Utilities

- - -

### `tf.test.assert_equal_graph_def(actual, expected)` {#assert_equal_graph_def}

Asserts that two `GraphDef`s are (mostly) the same.

Compares two `GraphDef` protos for equality, ignoring versions and ordering of
nodes, attrs, and control inputs.  Node names are used to match up nodes
between the graphs, so the naming of nodes must be consistent.

##### Args:


*  <b>`actual`</b>: The `GraphDef` we have.
*  <b>`expected`</b>: The `GraphDef` we expected.

##### Raises:


*  <b>`AssertionError`</b>: If the `GraphDef`s do not match.
*  <b>`TypeError`</b>: If either argument is not a `GraphDef`.


- - -

### `tf.test.get_temp_dir()` {#get_temp_dir}

Returns a temporary directory for use during tests.

There is no need to delete the directory after the test.

##### Returns:

  The temporary directory.


- - -

### `tf.test.is_built_with_cuda()` {#is_built_with_cuda}

Returns whether TensorFlow was built with CUDA (GPU) support.



## Gradient checking

[`compute_gradient`](#compute_gradient) and
[`compute_gradient_error`](#compute_gradient_error) perform numerical
differentiation of graphs for comparison against registered analytic gradients.

- - -

### `tf.test.compute_gradient(x, x_shape, y, y_shape, x_init_value=None, delta=0.001, init_targets=None)` {#compute_gradient}

Computes and returns the theoretical and numerical Jacobian.

If `x` or `y` is complex, the Jacobian will still be real but the
corresponding Jacobian dimension(s) will be twice as large.  This is required
even if both input and output is complex since TensorFlow graphs are not
necessarily holomorphic, and may have gradients not expressible as complex
numbers.  For example, if `x` is complex with shape `[m]` and `y` is complex
with shape `[n]`, each Jacobian `J` will have shape `[m * 2, n * 2]` with

    J[:m, :n] = d(Re y)/d(Re x)
    J[:m, n:] = d(Im y)/d(Re x)
    J[m:, :n] = d(Re y)/d(Im x)
    J[m:, n:] = d(Im y)/d(Im x)

##### Args:


*  <b>`x`</b>: a tensor or list of tensors
*  <b>`x_shape`</b>: the dimensions of x as a tuple or an array of ints. If x is a list,
  then this is the list of shapes.

*  <b>`y`</b>: a tensor
*  <b>`y_shape`</b>: the dimensions of y as a tuple or an array of ints.
*  <b>`x_init_value`</b>: (optional) a numpy array of the same shape as "x"
    representing the initial value of x. If x is a list, this should be a list
    of numpy arrays.  If this is none, the function will pick a random tensor
    as the initial value.
*  <b>`delta`</b>: (optional) the amount of perturbation.
*  <b>`init_targets`</b>: list of targets to run to initialize model params.
    TODO(mrry): remove this argument.

##### Returns:

  Two 2-d numpy arrays representing the theoretical and numerical
  Jacobian for dy/dx. Each has "x_size" rows and "y_size" columns
  where "x_size" is the number of elements in x and "y_size" is the
  number of elements in y. If x is a list, returns a list of two numpy arrays.


- - -

### `tf.test.compute_gradient_error(x, x_shape, y, y_shape, x_init_value=None, delta=0.001, init_targets=None)` {#compute_gradient_error}

Computes the gradient error.

Computes the maximum error for dy/dx between the computed Jacobian and the
numerically estimated Jacobian.

This function will modify the tensors passed in as it adds more operations
and hence changing the consumers of the operations of the input tensors.

This function adds operations to the current session. To compute the error
using a particular device, such as a GPU, use the standard methods for
setting a device (e.g. using with sess.graph.device() or setting a device
function in the session constructor).

##### Args:


*  <b>`x`</b>: a tensor or list of tensors
*  <b>`x_shape`</b>: the dimensions of x as a tuple or an array of ints. If x is a list,
  then this is the list of shapes.

*  <b>`y`</b>: a tensor
*  <b>`y_shape`</b>: the dimensions of y as a tuple or an array of ints.
*  <b>`x_init_value`</b>: (optional) a numpy array of the same shape as "x"
    representing the initial value of x. If x is a list, this should be a list
    of numpy arrays.  If this is none, the function will pick a random tensor
    as the initial value.
*  <b>`delta`</b>: (optional) the amount of perturbation.
*  <b>`init_targets`</b>: list of targets to run to initialize model params.
    TODO(mrry): Remove this argument.

##### Returns:

  The maximum error in between the two Jacobians.


