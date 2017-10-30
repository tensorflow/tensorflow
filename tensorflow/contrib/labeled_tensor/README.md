# Labels for TensorFlow

LabeledTensor is a library for adding semantically meaningful dimension and
coordinate labels to tensors in Tensorflow.

LabeledTensor was inspired by [xarray](http://xarray.pydata.org) and
[pandas](http://pandas.pydata.org), projects that adds labels to NumPy array.

## Data model

`LabeledTensor` is an immutable object consisting of two components:

- `tensor`: the `tf.Tensor` object containing the labeled tensor's data.
- `axes`: an OrderedDict-like object with keys given by axis names (e.g.,
  ``"channel"``) and values given by `Axis` objects.

`Axis` objects keep track of the size of a dimension and, optionally, coordinate
labels along that axis (e.g., `("red", "green", "blue")`) in the form of a
tuple stored in `Axis.labels`.

Operations on `LabeledTensors` use, preserve and transform axis names and
labels.

## Quick start

Try out the following snippet in a script or Jupyter notebook:

    import tensorflow as tf

    lt = tf.contrib.labeled_tensor

    # Create two LabeledTensors:
    raw_image = tf.ones((299, 299, 3))
    axes = ['row', 'column', ('channel', ['red', 'green', 'blue'])]
    image = lt.LabeledTensor(raw_image, axes)
    assert image.tensor is raw_image
    weights = lt.LabeledTensor(tf.constant([0.1, 0.3, 0.6]),
                               [image.axes['channel']])

    # Examples of valid operations:
    lt.transpose(image, ['column', 'row', 'channel'])
    lt.reshape(image, ['row', 'column'], ['pixel'])
    lt.concat([image, image], 'row')
    lt.reduce_sum(image, ['channel'])
    lt.select(image, {'channel': 'red'})
    lt.cast(image / 256.0, tf.uint8)
    image * weights
    lt.matmul(image[0, :, :], weights)
    tf.cos(image)  # automatically converts to tf.Tensor

## Adding a custom op

LabeledTensor has wrappers for [quite a
few](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/labeled_tensor/__init__.py)
TensorFlow ops.

To easily add your own, you can use the `define_unary_op`, `define_binary_op`
and `define_reduce_op` functions, e.g.,

    log = lt.define_unary_op('log', tf.log)

## Questions

Please reach out to the authors:

- Stephan Hoyer (shoyer@google.com, github.com/shoyer)
- Eric Christiansen (ericmc@google.com, github.com/emchristiansen)
