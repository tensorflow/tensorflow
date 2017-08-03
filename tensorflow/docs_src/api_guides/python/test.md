# Testing
[TOC]

## Unit tests

TensorFlow provides a convenience class inheriting from `unittest.TestCase`
which adds methods relevant to TensorFlow tests.  Here is an example:

```python
    import tensorflow as tf


    class SquareTest(tf.test.TestCase):

      def testSquare(self):
        with self.test_session():
          x = tf.square([2, 3])
          self.assertAllEqual(x.eval(), [4, 9])


    if __name__ == '__main__':
      tf.test.main()
```

`tf.test.TestCase` inherits from `unittest.TestCase` but adds a few additional
methods.  See @{tf.test.TestCase} for details.

*   @{tf.test.main}
*   @{tf.test.TestCase}
*   @{tf.test.test_src_dir_path}

## Utilities

Note: `tf.test.mock` is an alias to the python `mock` or `unittest.mock`
depending on the python version.

*   @{tf.test.assert_equal_graph_def}
*   @{tf.test.get_temp_dir}
*   @{tf.test.is_built_with_cuda}
*   @{tf.test.is_gpu_available}
*   @{tf.test.gpu_device_name}

## Gradient checking

@{tf.test.compute_gradient} and @{tf.test.compute_gradient_error} perform
numerical differentiation of graphs for comparison against registered analytic
gradients.
