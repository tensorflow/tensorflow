# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

# Copied over from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/as_dataframe_test.py - all code belongs to original authors 

import pandas

import tensorflow as tf
from tensorflow_datasets import testing
from tensorflow_datasets.experimental import as_dataframe
from tensorflow_datasets.core import load

# Import for registration
# pylint: disable=unused-import,g-bad-import-order
from tensorflow_datasets.image_classification import mnist
from tensorflow_datasets.text import anli
# pylint: enable=unused-import,g-bad-import-order


def _as_df(ds_name: str, **kwargs) -> pandas.DataFrame:
  """Loads the dataset as `pandas.DataFrame`."""
  with testing.mock_data(num_examples=3):
    ds, ds_info = load.load(ds_name, split='train', with_info=True, **kwargs)
  df = as_dataframe.as_dataframe(ds, ds_info)
  return df


def test_as_dataframe():
  """Tests that as_dataframe works without the `tfds.core.DatasetInfo`."""
  ds = tf.data.Dataset.from_tensor_slices(
      {
          'some_key': [1, 2, 3],
          'nested': {
              'sub1': [1.0, 2.0, 3.0],
          },
      }
  )
  df = as_dataframe.as_dataframe(ds)
  assert isinstance(df, pandas.DataFrame)
  assert df._repr_html_().startswith('<style')
  assert list(df.columns) == ['nested/sub1', 'some_key']


def test_text_dataset():
  df = _as_df('anli')
  assert isinstance(df, pandas.DataFrame)
  assert isinstance(df._repr_html_(), str)
  assert list(df.columns) == ['context', 'hypothesis', 'label', 'uid']


def test_as_supervised():
  df = _as_df('mnist', as_supervised=True)
  assert isinstance(df, pandas.DataFrame)
  assert isinstance(df._repr_html_(), str)
  assert list(df.columns) == ['image', 'label']
