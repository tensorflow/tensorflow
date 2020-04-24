# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras text dataset generation utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.preprocessing.text_dataset_from_directory', v1=[])
def text_dataset_from_directory(directory,
                                labels='inferred',
                                label_mode='int',
                                class_names=None,
                                batch_size=32,
                                max_length=None,
                                shuffle=True,
                                seed=None,
                                validation_split=None,
                                subset=None,
                                follow_links=False):
  """Generates a `tf.data.Dataset` from text files in a directory.

  If your directory structure is:

  ```
  main_directory/
  ...class_a/
  ......a_text_1.txt
  ......a_text_2.txt
  ...class_b/
  ......b_text_1.txt
  ......b_text_2.txt
  ```

  Then calling `text_dataset_from_directory(main_directory, labels='inferred')`
  will return a `tf.data.Dataset` that yields batches of texts from
  the subdirectories `class_a` and `class_b`, together with labels
  0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

  Only `.txt` files are supported at this time.

  Arguments:
    directory: Directory where the data is located.
        If `labels` is "inferred", it should contain
        subdirectories, each containing text files for a class.
        Otherwise, the directory structure is ignored.
    labels: Either "inferred"
        (labels are generated from the directory structure),
        or a list/tuple of integer labels of the same size as the number of
        text files found in the directory. Labels should be sorted according
        to the alphanumeric order of the text file paths
        (obtained via `os.walk(directory)` in Python).
    label_mode:
        - 'int': means that the labels are encoded as integers
            (e.g. for `sparse_categorical_crossentropy` loss).
        - 'categorical' means that the labels are
            encoded as a categorical vector
            (e.g. for `categorical_crossentropy` loss).
        - 'binary' means that the labels (there can be only 2)
            are encoded as `float32` scalars with values 0 or 1
            (e.g. for `binary_crossentropy`).
        - None (no labels).
    class_names: Only valid if "labels" is "inferred". This is the explict
        list of class names (must match names of subdirectories). Used
        to control the order of the classes
        (otherwise alphanumerical order is used).
    batch_size: Size of the batches of data. Default: 32.
    max_length: Maximum size of a text string. Texts longer than this will
      be truncated to `max_length`.
    shuffle: Whether to shuffle the data. Default: True.
        If set to False, sorts the data in alphanumeric order.
    seed: Optional random seed for shuffling and transformations.
    validation_split: Optional float between 0 and 1,
        fraction of data to reserve for validation.
    subset: One of "training" or "validation".
        Only used if `validation_split` is set.
    follow_links: Whether to visits subdirectories pointed to by symlinks.
        Defaults to False.

  Returns:
    A `tf.data.Dataset` object.
      - If `label_mode` is None, it yields `string` tensors of shape
        `(batch_size,)`, containing the contents of a batch of text files.
      - Otherwise, it yields a tuple `(texts, labels)`, where `texts`
        has shape `(batch_size,)` and `labels` follows the format described
        below.

  Rules regarding labels format:
    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorial`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.
  """
  if labels != 'inferred':
    if not isinstance(labels, (list, tuple)):
      raise ValueError(
          '`labels` argument should be a list/tuple of integer labels, of '
          'the same size as the number of text files in the target '
          'directory. If you wish to infer the labels from the subdirectory '
          'names in the target directory, pass `labels="inferred"`. '
          'If you wish to get a dataset that only contains text samples '
          '(no labels), pass `labels=None`.')
    if class_names:
      raise ValueError('You can only pass `class_names` if the labels are '
                       'inferred from the subdirectory names in the target '
                       'directory (`labels="inferred"`).')
  if label_mode not in {'int', 'categorical', 'binary', None}:
    raise ValueError(
        '`label_mode` argument must be one of "int", "categorical", "binary", '
        'or None. Received: %s' % (label_mode,))

  if seed is None:
    seed = np.random.randint(1e6)
  file_paths, labels, class_names = dataset_utils.index_directory(
      directory,
      labels,
      formats=('.txt',),
      class_names=class_names,
      shuffle=shuffle,
      seed=seed,
      follow_links=follow_links)

  if label_mode == 'binary' and len(class_names) != 2:
    raise ValueError(
        'When passing `label_mode="binary", there must exactly 2 classes. '
        'Found the following classes: %s' % (class_names,))

  file_paths, labels = dataset_utils.get_training_or_validation_split(
      file_paths, labels, validation_split, subset)

  dataset = paths_and_labels_to_dataset(
      file_paths=file_paths,
      labels=labels,
      label_mode=label_mode,
      num_classes=len(class_names),
      max_length=max_length)
  if shuffle:
    # Shuffle locally at each iteration
    dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
  dataset = dataset.batch(batch_size)
  # Users may need to reference `class_names`.
  dataset.class_names = class_names
  return dataset


def paths_and_labels_to_dataset(file_paths,
                                labels,
                                label_mode,
                                num_classes,
                                max_length):
  """Constructs a dataset of text strings and labels."""
  path_ds = dataset_ops.Dataset.from_tensor_slices(file_paths)
  string_ds = path_ds.map(
      lambda x: path_to_string_content(x, max_length))
  if label_mode:
    label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
    string_ds = dataset_ops.Dataset.zip((string_ds, label_ds))
  return string_ds


def path_to_string_content(path, max_length):
  txt = io_ops.read_file(path)
  if max_length is not None:
    txt = string_ops.substr(txt, 0, max_length)
  return txt
