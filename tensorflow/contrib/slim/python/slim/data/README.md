# TensorFlow-Slim Data

TF-Slim provides a data loading library for facilitating the reading of data
from various formats. TF-Slim's data modules are composed of several layers of
abstraction to make it flexible enough to support multiple file storage types,
such as TFRecords or Text files, data encoding and features naming schemes.

# Overview

The task of loading data has two main components: (1) specification of how
a dataset is represented so it can be read and interpreted and (2) instruction
for providing the data to consumers of the dataset.

Secondly, one must specify instructions for how
the data is actually provided and housed in memory. For example, if the data is
sharded over many sources, should it be read in parallel from these sources?
Should it be read serially? Should the data be shuffled in memory?

# Dataset Specification

TF-Slim defines a dataset to be a set of files (that may or may not be encoded)
representing a finite set of samples, and which can be read to provide a
predefined set of entities or `items`. For example, a dataset might be stored
over thousands of files or a single file. The files might store the data in
clear text or some advanced encoding scheme. It might provide a single `item`,
like an image, or several `items`, like an image, a class label and a scene
label.

More concretely, TF-Slim's
[dataset](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/dataset.py)
is a tuple that encapsulates the following elements of a dataset specification:

* `data_sources`: A list of file paths that together make up the dataset
* `reader`: A TensorFlow
[Reader](https://www.tensorflow.org/api_docs/python/io_ops.html#ReaderBase)
appropriate for the file type in `data_sources`.
* `decoder`: A TF-Slim
[data_decoder](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/data_decoder.py)
class which is used to decode the content of the read dataset files.
* `num_samples`: The number of samples in the dataset.
* `items_to_descriptions`: A map from the items provided by the dataset to
descriptions of each.

In a nutshell, a dataset is read by (a) opening the files specified by
`data_sources` using the given `reader` class (b) decoding the files using
the given `decoder` and (c) allowing the user to request a list of `items` to
be returned as `Tensors`.

## Data Decoders

A
[data_decoder](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/data_decoder.py)
is a class which is given some (possibly serialized/encoded) data and returns a
list of `Tensors`. In particular, a given data decoder is able to decode a
predefined list of `items` and can return a subset or all of them, when
requested:

```python
# Load the data
my_encoded_data = ...
data_decoder = MyDataDecoder()

# Decode the inputs and labels:
decoded_input, decoded_labels = data_decoder.Decode(data, ['input', 'labels'])

# Decode just the inputs:
decoded_input = data_decoder.Decode(data, ['input'])

# Check which items a data decoder knows how to decode:
for item in data_decoder.list_items():
  print(item)
```

## Example: TFExampleDataDecoder

The
[tfexample_data_decoder.py](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/tfexample_data_decoder.py)
is a data decoder which decodes serialized `TFExample` protocol buffers. A
`TFExample` protocol buffer is a map from keys (strings) to either a
`tf.FixedLenFeature` or `tf.VarLenFeature`. Consequently, to decode a
`TFExample`, one must provide a mapping from one or more `TFExample` fields
to each of the `items` that the `tfexample_data_decoder` can provide. For
example, a dataset of `TFExamples` might store images in various formats and
each `TFExample` might contain an `encoding` key and a `format` key which can
be used to decode the image using the appropriate decoder (jpg, png, etc).

To make this possible, the `tfexample_data_decoder` is constructed by specifying
the a map of `TFExample` keys to either `tf.FixedLenFeature` or
`tf.VarLenFeature` as well as a set of `ItemHandlers`. An `ItemHandler`
provides a mapping from `TFExample` keys to the item being provided. Because a
`tfexample_data_decoder` might return multiple `items`, one often constructs a
`tfexample_data_decoder` using multiple `ItemHandlers`.

`tfexample_data_decoder` provides some predefined `ItemHandlers` which take care
of the common cases of mapping `TFExamples` to images, `Tensors` and
`SparseTensors`. For example, the following specification might be
used to decode a dataset of images:

```python
keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
    'image/class/label': tf.FixedLenFeature(
        [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
}

items_to_handlers = {
    'image': tfexample_decoder.Image(
      image_key = 'image/encoded',
      format_key = 'image/format',
      shape=[28, 28],
      channels=1),
    'label': tfexample_decoder.Tensor('image/class/label'),
}

decoder = tfexample_decoder.TFExampleDecoder(
    keys_to_features, items_to_handlers)
```

Notice that the TFExample is parsed using three keys: `image/encoded`,
`image/format` and `image/class/label`. Additionally, the first two keys are
mapped to a single `item` named 'image'. As defined, this `data_decoder`
provides two `items` named 'image' and 'label'.

# Data Provision

A
[data_provider](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/data_provider.py)
is a class which provides `Tensors` for each item requested:

```python
my_data_provider = ...
image, class_label, bounding_box = my_data_provider.get(
    ['image', 'label', 'bb'])
```

The
[dataset_data_provider](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/dataset_data_provider.py)
is a `data_provider` that provides data from a given `dataset` specification:

```python
dataset = GetDataset(...)
data_provider = dataset_data_provider.DatasetDataProvider(
    dataset, common_queue_capacity=32, common_queue_min=8)
```

The `dataset_data_provider` enables control over several elements of data
provision:

* How many concurrent readers are used.
* Whether the data is shuffled as its loaded into its queue
* Whether to take a single pass over the data or read data indefinitely.

