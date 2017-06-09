# Using the `Dataset` API for TensorFlow Input Pipelines

The `Dataset` API is designed to let you build complex input pipelines from
simple, reusable pieces. For example, the pipeline for an image model might
aggregate data from files in a distributed file system, apply random
perturbations to each image, and merge randomly selected images into a batch
for training. The pipeline for a text model might involve extracting symbols
from raw text data, converting them to embedding identifiers with a lookup
table, and batching together sequences of different lengths. The `Dataset` API
makes it easy to deal with large amounts of data, different data formats, and
complicated transformations.

The `Dataset` API introduces two new abstractions to TensorFlow:

* A `tf.contrib.data.Dataset` represents a sequence of elements, in which
  each element contains one or more `Tensor` objects. For example, in an image
  pipeline, an element might be a single training example, with a pair of
  tensors representing the image data and a label. A `Dataset` can either be a
  *source* (e.g. `Dataset.from_tensor_slices()` constructs a dataset from one
  or more `tf.Tensor` objects), or a *transformation* (e.g. `Dataset.batch()`
  constructs a dataset by stacking consecutive elements of another dataset into
  a single element).

* A `tf.contrib.data.Iterator` provides the main way to extract elements from a
  dataset. The `Iterator.get_next()` operation yields the next element of a
  `Dataset`, and typically acts as the interface between input pipeline code and
  your model. The simplest iterator is a "one-shot iterator", which is
  associated with a particular `Dataset` and iterates through it once. For more
  sophisticated uses, the `Iterator.initializer` operation enables you to
  reinitialize and parameterize an iterator with different datasets, so that
  you can, for example, iterate over training and validation data multiple times
  in the same program.

## Tutorial

This programmers' guide includes step-by-step instructions for a variety of
input data use cases. Also see the `Dataset` and `Iterator` class references
for more detailed information about the API.

### Basic mechanics

This section of the guide describes the fundamentals of creating different kinds
of `Dataset` and `Iterator` objects, and how to extract data from them.

#### Defining a source dataset

You can build a `Dataset` using one of the following *source* dataset
constructors:

* From in-memory data:
  * `tf.contrib.data.Dataset.from_tensors()`
  * `tf.contrib.data.Dataset.from_tensor_slices()`

* From on-disk data:
  * `tf.contrib.data.FixedLengthRecordDataset()`
  * `tf.contrib.data.TextLineDataset()`
  * `tf.contrib.data.TFRecordDataset()`

* From parameters:
  * `tf.contrib.data.Dataset.range()`

#### Transforming a dataset

The `tf.contrib.data.Dataset` class has many methods that can be chained
together to *transform* one dataset into another:

* Per-element transformations:
  * `Dataset.filter()`
  * `Dataset.flat_map()`
  * `Dataset.map()`
  * `Dataset.zip()`

* Multi-element transformations:
  * `Dataset.batch()`
  * `Dataset.dense_to_sparse_batch()`
  * `Dataset.group_by_window()`
  * `Dataset.padded_batch()`
  * `Dataset.repeat()`
  * `Dataset.shuffle()`
  * `Dataset.skip()`
  * `Dataset.take()`

The following sections contain examples of how to use these transformations to
solve common problems.

#### Dataset structure

A dataset comprises elements that each have the same structure. An element
contains one or more `tf.Tensor` objects, called *components*. Each component
has a `tf.DType` representing the type of elements in the tensor, and a
`tf.TensorShape` representing the (possibly partially specified) static shape of
each element. The `Dataset.output_types` and `Dataset.output_shapes` properties
allow you to inspect the inferred types and shapes of each component of a
dataset element. The *nested structure* of these properties map to the structure
of an element, which may be a single tensor, a tuple of tensors, or a nested
tuple of tensors. For example:

```python
dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.contrib.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.contrib.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
```

The `Dataset` transformations support datasets of any structure. When using the
`Dataset.map()`, `Dataset.flat_map()` and `Dataset.filter()` transformations,
which apply a function to each element, the element structure determines the
arguments of the function:

```python
dataset1 = dataset1.map(lambda x: ...)

dataset2 = dataset2.flat_map(lambda x, y: ...)

# *N.B.* Lambda argument destructuring is not available in Python 3.
dataset3 = dataset3.filter(lambda x, (y, z): ...)
```

#### Creating an iterator

One you have built a `Dataset` to represent your input data, the next step is to
create an `Iterator` to access elements from that dataset.  The `Dataset` API
currently supports three kinds of iterator, in increasing level of
sophistication:

A *one-shot* iterator is the simplest form of iterator, which only supports
iterating once through a dataset, with no need for explicit initialization.
One-shot iterators handle almost all of the cases that the existing queue-based
input pipelines support, but they do not support parameterization. Using the
example of `Dataset.range()`:

```python
dataset = tf.contrib.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

An *initializable* iterator requires you to run an explicit
`iterator.initializer` operation before using it. In exchange for this
inconvenience, it enables you to *parameterize* the definition of the dataset,
using one or more `tf.placeholder()` tensors that can be fed when you
initialize the iterator. Continuing the `Dataset.range()` example:

```python
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.contrib.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  assert i == value

# Initialize the same iterator over a dataset with 100 elements.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

A *reinitializable* iterator can be initialized from multiple different
`Dataset` objects. For example, you might have a training input pipeline that
uses random perturbations to the input images to improve generalization, and
a validation input pipeline that evaluates predictions on unmodified data. These
pipelines will typically use different `Dataset` objects that have the same
structure (i.e. the same types and compatible shapes for each component). 

```python
training_dataset = tf.contrib.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.contrib.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = Iterator.from_structure(training_dataset.output_types,
                                   training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # Initialize an iterator over the validation dataset.
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)
```

#### Consuming values from an iterator

The `Iterator.get_next()` method returns one or more `tf.Tensor` objects that
correspond to the symbolic next element of an iterator. Each time these tensors
are evaluated, they take the value of the next element in the underlying
dataset. (Note that, like other stateful objects in TensorFlow, calling
`Iterator.get_next()` does not immediately advance the iterator. Instead you
must use the returned `tf.Tensor` objects in a TensorFlow expression, and pass
the result of that expression to `tf.Session.run()` to get the next elements and
advance the iterator.)

If the iterator reaches the end of the dataset, executing
the `Iterator.get_next()` operation will raise a `tf.errors.OutOfRangeError`.
After this point the iterator will be in an unusable state, and you must
initialize it again if you want to use it further.

```python
dataset = tf.contrib.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer's
# training operation.
result = tf.add(next_element, next_element)

sess.run(iterator.initializer)
print(sess.run(result))  # ==> "0"
print(sess.run(result))  # ==> "2"
print(sess.run(result))  # ==> "4"
print(sess.run(result))  # ==> "6"
print(sess.run(result))  # ==> "8"
try:
  sess.run(result)
except tf.errors.OutOfRangeError:
  print("End of dataset")  # ==> "End of dataset"
```

A common pattern is to wrap the "training loop" in a `try`-`except` block:

```python
sess.run(iterator.initializer)
while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    break
```

If each element of the dataset has a nested structure, the return value of
`Iterator.get_next()` will be one or more `tf.Tensor` objects in the same
nested structure:

```python
dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.contrib.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.contrib.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

sess.run(iterator.initializer)
next1, (next2, next3) = iterator.get_next()
```

Note that evaluating *any* of `next1`, `next2`, or `next3` will advance the
iterator for all components. A typical consumer of an iterator will include all
components in a single expression.

### Reading input data

#### Consuming NumPy arrays

If all of your input data fit in memory, the simplest way to create a `Dataset`
from them is to convert them to `tf.Tensor` objects and use
`Dataset.from_tensor_slices()`.

```python
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.contrib.data.Dataset.from_tensor_slices((features, labels))
```

Note that the above code snippet will embed the `features` and `labels` arrays
in your TensorFlow graph as constants. This works well for a small dataset, but
wastes memory, and can run into the 2GB limit for the `tf.GraphDef` protocol
buffer.

As an alternative, you can define the `Dataset` in terms of `tf.placeholder()`
tensors, and *feed* the NumPy arrays when you initialize an `Iterator` over the
dataset.

```python
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
```

#### Consuming TFRecord data

The `Dataset` API supports a variety of file formats so that you can process
large datasets that do not fit in memory. The TFRecord file format is a
simple record-oriented binary format that many TensorFlow applications use for
training data. The `tf.contrib.data.TFRecordDataset` class enables you to
stream over the contents of one or more TFRecord files as part of an input
pipeline.

```python
# Creates a dataset that reads all of the examples from two files.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
```

The `filenames` argument to the `TFRecordDataset` initializer can be a
`tf.Tensor` of strings. Therefore if you have two sets of files for training
and validation purposes, you can use a `tf.placeholder(tf.string)` to represent
the filenames, and initialize an iterator from the appropriate filenames:

```python
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.TFRecordDataset(filenames)
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.

# Initialize `iterator` with training data.
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# Initialize `iterator` with validation data.
validation_filenames = ["/var/data/validation1.tfrecord", ...]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
```

#### Consuming text data

Many datasets are distributed as one or more text files. The
`tf.contrib.data.TextLineDataset` provides an easy way to extract lines from
one or more text files. Given one or more filenames, a `TextLineDataset` will
produce one string-valued element per line of those files. Like a
`TFRecordDataset`, `TextLineDataset` accepts `filenames` as a `tf.Tensor`, so
you can parameterize it by passing a `tf.placeholder(tf.string)`.

```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.contrib.data.TextLineDataset(filenames)
```

By default, a `TextLineDataset` yields *every* line of each file, which may
not be desirable, for example if the file starts with a header line, or contains
comments. These lines can be removed using the `Dataset.skip()` and
`Dataset.filter()` transformations. To apply these transformations to each
file separately, we use `Dataset.flat_map()` to create a nested `Dataset` for
each file.

```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames)

# Use `Dataset.flat_map()` to transform each file separately.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset = dataset.flat_map(
    lambda filename: (
        tf.contrib.data.Dataset.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))
```

<!--
TODO(mrry): Add these sections.

#### Consuming from a Python generator
#### Consuming from an index file and images
-->

### Preprocessing data with `Dataset.map()`

The `Dataset.map(f)` transformation produces a new dataset by applying a given
function `f` to each element of the input dataset. It is based on
the
[`map()` function](https://en.wikipedia.org/wiki/Map_(higher-order_function))
that is commonly applied to lists (and other structures) in functional
programming languages.  The function `f` takes the `tf.Tensor` objects that
represent a single element in the input, and returns the `tf.Tensor` objects
that will represent a single element in the new dataset. Its implementation uses
standard TensorFlow operations to transform one element into another.

This section covers common examples of how to use `Dataset.map()`.

#### Parsing `tf.Example` protocol buffer messages

Many input pipelines extract `tf.train.Example` protocol buffer messages from a
TFRecord-format file (written, for example, using
`tf.python_io.TFRecordWriter`). Each `tf.train.Example` record contains one or
more "features", and the input pipeline typically converts these features into
tensors.

```python
# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["image"], parsed_features["label"]

# Creates a dataset that reads all of the examples from two files, and extracts
# the image and label features.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
```

#### Decoding image data and resizing it

When training a neural network on real-world image data, it is often necessary
to convert images of different sizes to a common size, so that they may be
batched into a fixed size.

```python
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
labels = tf.constant([0, 37, 29, 1, ...])

dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
```

#### Applying arbitrary Python logic with `tf.py_func()`

For performance reasons, we encourage you to use TensorFlow operations for
preprocessing your data whenever possible. However, it is sometimes useful to
be able to call upon external Python libraries when parsing your input data,
and you can do this by invoking the `tf.py_func()` operation in a
`Dataset.map()` transformation.

```python
import cv2

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(image_string, cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
labels = [0, 37, 29, 1, ...]

dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype]))
dataset = dataset.map(_resize_function)
```

<!--
TODO(mrry): Add this section.

#### Handling text data with unusual sizes
-->

### Batching dataset elements

#### Simple batching

The simplest form of batching stacks `n` consecutive elements of a dataset into
a single element. The `Dataset.batch()` transformation does exactly this, with
the same constraints as the `tf.stack()` operator, applied to each component
of the elements: i.e. for each component *i*, all elements must have a tensor
of the exact same shape.

```python
inc_dataset = tf.contrib.data.Dataset.range(100)
dec_dataset = tf.contrib.data.Dataset.range(0, -100, -1)
dataset = tf.contrib.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
```

#### Batching tensors with padding

The above recipe works for tensors that all have the same size. However, many
models (e.g. sequence models) work with input data that can have varying size
(e.g. sequences of different lengths). To handle this case, the
`Dataset.padded_batch()` transformation enables you to batch tensors of
different shape by specifying one or more dimensions in which they may be
padded.

```python
dataset = tf.contrib.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]
```

The `Dataset.padded_batch()` transformation allows you to set different padding
for each dimension of each component, and it may be variable-length (signified
by `None` in the example above) or constant-length. It is also possible to
override the padding value, which defaults to 0.

<!--
TODO(mrry): Add this section.

#### Dense ragged -> tf.SparseTensor
-->

### Training workflows

#### Processing multiple epochs

The `Dataset` API offers two main ways to process multiple epochs of the same
data.

The simplest way to iterate over a dataset in multiple epochs is to use the
`Dataset.repeat()` transformation. For example, to create a dataset that repeats
its input for 10 epochs:

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)
```

Applying the `Dataset.repeat()` transformation with no arguments will repeat
the input indefinitely. The `Dataset.repeat()` transformation concatenates its
arguments without signaling the end of one epoch and the beginning of the next
epoch.

If you want to receive a signal at the end of each epoch, you can write a
training loop that catches the `tf.errors.OutOfRangeError` at the end of a
dataset. At that point you might collect some statistics (e.g. the validation
error) for the epoch.

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 100 epochs.
for _ in range(100):
  sess.run(iterator.initializer)
  while True:
    try:
      sess.run(next_element)
    except tf.errors.OutOfRangeError:
      break

  # [Perform end-of-epoch calculations here.]
```

#### Randomly shuffling input data

The `Dataset.shuffle()` transformation randomly shuffles the input dataset
using a similar algorithm to `tf.RandomShuffleQueue`: it maintains a fixed-size
buffer and chooses the next element uniformly at random from that buffer.

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
```
