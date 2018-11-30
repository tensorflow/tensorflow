# TF 2.0 Showcase

The code here shows idiomatic ways to write TensorFlow 2.0 code. It doubles as
an integration test.

## General guidelines for showcase code:

- Code should minimize dependencies and be self-contained in one file. A user
  should be able to copy-paste the example code into their project and have it
  just work.
- Code should emphasize simplicity over performance, as long as it performs
  within a factor of 2-3x of the optimized implementation.
- Code should work on CPU and single GPU.
- Code should run in Python 3.
- Code should conform to the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)


- Code should follow these guidelines:
  - Prefer Keras.
  - Split code into separate input pipeline and model code segments.
  - Don't use tf.cond or tf.while_loop; instead, make use of AutoGraph's
    functionality to compile Python `for`, `while`, and `if` statements.
  - Prefer a simple training loop over Estimator
  - Save and restore a SavedModel.
  - Write basic TensorBoard metrics - loss, accuracy,
