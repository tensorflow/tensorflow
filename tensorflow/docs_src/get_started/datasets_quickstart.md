# Datasets Quick Start

The @{tf.data} module contains a collection of classes that allows you to
easily load data, manipulate it, and pipe it into your model. This document
introduces the API by walking through two simple examples:

* Reading in-memory data from numpy arrays.
* Reading lines from a csv file.

<!-- TODO(markdaoust): Add links to an example reading from multiple-files
(image_retraining), and a from_generator example. -->

## Basic input

Taking slices from an array is the simplest way to get started with `tf.data`.

The @{$get_started/premade_estimators$Premade Estimators} chapter describes
the following `train_input_fn`, from
[`iris_data.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py),
to pipe the data into the Estimator:

``` python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
```

Let's look at this more closely.

### Arguments

This function expects three arguments. Arguments expecting an "array" can
accept nearly anything that can be converted to an array with `numpy.array`.
One exception is
[`tuple`](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences)
which has special meaning for `Datasets`.

* `features`: A `{'feature_name':array}` dictionary (or
  [`DataFrame`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html))
  containing the raw input features.
* `labels` : An array containing the
  [label](https://developers.google.com/machine-learning/glossary/#label)
  for each example.
* `batch_size` : An integer indicating the desired batch size.

In [`premade_estimator.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py)
we retrieved the Iris data using the `iris_data.load_data()` function.
You can run it, and unpack the results as follows:

``` python
import iris_data

# Fetch the data
train, test = iris_data.load_data()
features, labels = train
```

Then we passed this data to the input function, with a line similar to this:

``` python
batch_size=100
iris_data.train_input_fn(features, labels, batch_size)
```

Let's walk through the `train_input_fn()`.

### Slices

In the simplest cases, @{tf.data.Dataset.from_tensor_slices} function takes an
array and returns a @{tf.data.Dataset} representing slices of the array. For
example, an array containing the @{$tutorials/layers$mnist training data}
has a shape of `(60000, 28, 28)`. Passing this to `from_tensor_slices` returns
a `Dataset` object containing 60000 slices, each one a 28x28 image.

The code that returns this `Dataset` is as follows:

``` python
train, test = tf.keras.datasets.mnist.load_data()
mnist_x, mnist_y = train

mnist_ds = tf.data.Dataset.from_tensor_slices(mnist_x)
print(mnist_ds)
```

This will print the following line, showing the @{$programmers_guide/tensors#shapes$shapes} and @{$programmers_guide/tensors#data_types$types} of the items in
the dataset. Note that the dataset does not know how many items it contains.

``` None
<TensorSliceDataset shapes: (28,28), types: tf.uint8>
```

The dataset above represents a collection of simple arrays, but datasets are
much more powerful than this. Datasets transparently handle any nested
combination of dictionaries or tuples. For example, ensuring that `features`
is a standard dictionary, you can then convert the dictionary of arrays to
a `Dataset` of dictionaries as follows:

``` python
dataset = tf.data.Dataset.from_tensor_slices(dict(features))
print(dataset)
```
``` None
<TensorSliceDataset

  shapes: {
    SepalLength: (), PetalWidth: (),
    PetalLength: (), SepalWidth: ()},

  types: {
      SepalLength: tf.float64, PetalWidth: tf.float64,
      PetalLength: tf.float64, SepalWidth: tf.float64}
>
```

Here we see that when a `Dataset` contains structured elements, the `shapes`
and `types` of the `Dataset` take on the same structure. This dataset contains
dictionaries of @{$programmers_guide/tensors#rank$scalars}, all of type
`tf.float64`.

The first line of `train_input_fn` uses the same functionality, but adds
another level of structure. It creates a dataset containing
`(features, labels)` pairs.

The following code shows that the label is a scalar with type `int64`:

``` python
# Convert the inputs to a Dataset.
dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
print(dataset)
```
```
<TensorSliceDataset
    shapes: (
        {
          SepalLength: (), PetalWidth: (),
          PetalLength: (), SepalWidth: ()},
        ()),

    types: (
        {
          SepalLength: tf.float64, PetalWidth: tf.float64,
          PetalLength: tf.float64, SepalWidth: tf.float64},
        tf.int64)>
```

### Manipulation

Currently the `Dataset` would iterate over the data once, in a fixed order, and
only produce a single element at a time. It needs further processing before it
can be used for training. Fortunately, the `tf.data.Dataset` class provides
methods to better prepare the data for training. The next line of the input
function takes advantage of several of these methods:

``` python
# Shuffle, repeat, and batch the examples.
dataset = dataset.shuffle(1000).repeat().batch(batch_size)
```

The @{tf.data.Dataset.shuffle$`shuffle`} method uses a fixed-size buffer to
shuffle the items as they pass through. Setting a `buffer_size` greater than
the number of examples in the `Dataset` ensures that the data is completely
shuffled. The Iris data set only contains 150 examples.

The @{tf.data.Dataset.repeat$`repeat`} method has the `Dataset` restart when
it reaches the end. To limit the number of epochs, set the `count` argument.

The @{tf.data.Dataset.repeat$`batch`} method collects a number of examples and
stacks them, to create batches. This adds a dimension to their shape. The new
dimension is added as the first dimension. The following code uses
the `batch` method on the MNIST `Dataset`, from earlier. This results in a
`Dataset` containing 3D arrays representing stacks of `(28,28)` images:

``` python
print(mnist_ds.batch(100))
```

``` none
<BatchDataset
  shapes: (?, 28, 28),
  types: tf.uint8>
```
Note that the dataset has an unknown batch size because the last batch will
have fewer elements.

In `train_input_fn`, after batching the `Dataset` contains 1D vectors of
elements where each scalar was previously:

```python
print(dataset)
```
```
<TensorSliceDataset
    shapes: (
        {
          SepalLength: (?,), PetalWidth: (?,),
          PetalLength: (?,), SepalWidth: (?,)},
        (?,)),

    types: (
        {
          SepalLength: tf.float64, PetalWidth: tf.float64,
          PetalLength: tf.float64, SepalWidth: tf.float64},
        tf.int64)>
```


### Return

<!-- TODO(markdaoust) This line can be simplified to "return dataset" -->

The `train`, `evaluate`, and `predict` methods of every Estimator require
input functions to return a `(features, label)` pair containing
@{$programmers_guide/tensors$tensorflow tensors}. The `train_input_fn` uses
the following line to convert the Dataset into the expected format:

```python
# Build the Iterator, and return the read end of the pipeline.
features_result, labels_result = dataset.make_one_shot_iterator().get_next()
```

The result is a structure of @{$programmers_guide/tensors$TensorFlow tensors},
matching the layout of the items in the `Dataset`.
For an introduction to what these objects are and how to work with them,
see @{$programmers_guide/low_level_intro}.

``` python
print((features_result, labels_result))
```

```None
({
    'SepalLength': <tf.Tensor 'IteratorGetNext:2' shape=(?,) dtype=float64>,
    'PetalWidth': <tf.Tensor 'IteratorGetNext:1' shape=(?,) dtype=float64>,
    'PetalLength': <tf.Tensor 'IteratorGetNext:0' shape=(?,) dtype=float64>,
    'SepalWidth': <tf.Tensor 'IteratorGetNext:3' shape=(?,) dtype=float64>},
Tensor("IteratorGetNext_1:4", shape=(?,), dtype=int64))
```

## Reading a CSV File

The most common real-world use case for the `Dataset` class is to stream data
from files on disk. The @{tf.data} module includes a variety of
file readers. Let's see how parsing the Iris dataset from the csv file looks
using a `Dataset`.

The following call to the `iris_data.maybe_download` function downloads the
data if necessary, and returns the pathnames of the resulting files:

``` python
import iris_data
train_path, test_path = iris_data.maybe_download()
```

The [`iris_data.csv_input_fn`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py)
function contains an alternative implementation that parses the csv files using
a `Dataset`.

Let's look at how to build an Estimator-compatible input function that reads
from the local files.

### Build the `Dataset`

We start by building a @{tf.data.TextLineDataset$`TextLineDataset`} object to
read the file one line at a time. Then, we call the
@{tf.data.Dataset.skip$`skip`} method to skip over the first line of the file, which contains a header, not an example:

``` python
ds = tf.data.TextLineDataset(train_path).skip(1)
```

### Build a csv line parser

Ultimately we will need to parse each of the lines in the dataset, to
produce the necessary `(features, label)` pairs.

We will start by building a function to parse a single line.

The following `iris_data.parse_line` function accomplishes this task using the
@{tf.decode_csv} function, and some simple python code:

We must parse each of the lines in the dataset in order to generate the
necessary `(features, label)` pairs. The following `_parse_line` function
calls @{tf.decode_csv} to parse a single line into its features
and the label. Since Estimators require that features be represented as a
dictionary, we rely on Python's built-in `dict` and `zip` functions to build
that dictionary.  The feature names are the keys of that dictionary.
We then call the dictionary's `pop` method to remove the label field from
the features dictionary:

``` python
# Metadata describing the text columns
COLUMNS = ['SepalLength', 'SepalWidth',
           'PetalLength', 'PetalWidth',
           'label']
FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]
def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)

    # Pack the result into a dictionary
    features = dict(zip(COLUMNS,fields))

    # Separate the label from the features
    label = features.pop('label')

    return features, label
```

### Parse the lines

Datasets have many methods for manipulating the data while it is being piped
to a model. The most heavily-used method is @{tf.data.Dataset.map$`map`}, which
applies a transformation to each element of the `Dataset`.

The `map` method takes a `map_func` argument that describes how each item in the
`Dataset` should be transformed.

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/datasets/map.png">
</div>
<div style="text-align: center">
The @{tf.data.Dataset.map$`map`} method applies the `map_func` to
transform each item in the <code>Dataset</code>.
</div>

So to parse the lines as they are streamed out of the csv file, we pass our
`_parse_line` function to the `map` method:

``` python
ds = ds.map(_parse_line)
print(ds)
```
``` None
<MapDataset
shapes: (
    {SepalLength: (), PetalWidth: (), ...},
    ()),
types: (
    {SepalLength: tf.float32, PetalWidth: tf.float32, ...},
    tf.int32)>
```

Now instead of simple scalar strings, the dataset contains `(features, label)`
pairs.

the remainder of the `iris_data.csv_input_fn` function is identical
to `iris_data.train_input_fn` which was covered in the in the
[Basic input](#basic_input) section.

### Try it out

This function can be used as a replacement for
`iris_data.train_input_fn`. It can be used to feed an estimator as follows:

``` python
train_path, test_path = iris_data.maybe_download()

# All the inputs are numeric
feature_columns = [
    tf.feature_column.numeric_column(name)
    for name in iris_data.CSV_COLUMN_NAMES[:-1]]

# Build the estimator
est = tf.estimator.LinearClassifier(feature_columns,
                                    n_classes=3)
# Train the estimator
batch_size = 100
est.train(
    steps=1000,
    input_fn=lambda : iris_data.csv_input_fn(train_path, batch_size))
```

Estimators expect an `input_fn` to take no arguments. To work around this
restriction, we use `lambda` to capture the arguments and provide the expected
interface.

## Summary

The `tf.data` module provides a collection of classes and functions for easily
reading data from a variety of sources. Furthermore, `tf.data` has simple
powerful methods for applying a wide variety of standard and custom
transformations.

Now you have the basic idea of how to efficiently load data into an
Estimator. Consider the following documents next:


* @{$get_started/custom_estimators}, which demonstrates how to build your own
  custom `Estimator` model.
* The @{$low_level_intro#datasets$Low Level Introduction}, which demonstrates
  how to experiment directly with `tf.data.Datasets` using TensorFlow's low
  level APIs.
* @{$programmers_guide/datasets} which goes into great detail about additional
  functionality of `Datasets`.

