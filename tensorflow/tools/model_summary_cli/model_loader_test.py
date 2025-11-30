# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for model_loader module."""

import os
import tempfile

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.platform import test

from tensorflow.tools.model_summary_cli import model_loader


class DetectFormatTest(test.TestCase):

  def test_nonexistent_path(self):
    result = model_loader.detect_format('/nonexistent/path')
    self.assertEqual(result, 'not_found')

  def test_h5_extension(self):
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
      # Create a minimal valid model and save
      model = models.Sequential([layers.Dense(1, input_shape=(1,))])
      model.save(f.name)
      try:
        result = model_loader.detect_format(f.name)
        self.assertEqual(result, 'h5')
      finally:
        os.unlink(f.name)

  def test_keras_extension(self):
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
      model = models.Sequential([layers.Dense(1, input_shape=(1,))])
      model.save(f.name)
      try:
        result = model_loader.detect_format(f.name)
        self.assertEqual(result, 'keras')
      finally:
        os.unlink(f.name)

  def test_savedmodel_directory(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      model_path = os.path.join(tmpdir, 'saved_model')
      model = models.Sequential([layers.Dense(1, input_shape=(1,))])
      model.save(model_path, save_format='tf')
      result = model_loader.detect_format(model_path)
      self.assertEqual(result, 'savedmodel')

  def test_unknown_format(self):
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
      f.write(b'not a model')
      try:
        result = model_loader.detect_format(f.name)
        self.assertEqual(result, 'unknown')
      finally:
        os.unlink(f.name)


class LoadModelTest(test.TestCase):

  def test_load_nonexistent_raises(self):
    with self.assertRaises(model_loader.ModelLoadError) as ctx:
      model_loader.load_model('/nonexistent/path')
    self.assertIn('does not exist', str(ctx.exception))

  def test_load_unknown_format_raises(self):
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
      f.write(b'not a model')
      try:
        with self.assertRaises(model_loader.ModelLoadError) as ctx:
          model_loader.load_model(f.name)
        self.assertIn('could not determine', str(ctx.exception).lower())
      finally:
        os.unlink(f.name)

  def test_load_h5_model(self):
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
      original = models.Sequential([
          layers.Dense(10, input_shape=(5,), name='dense_1'),
          layers.Dense(1, name='dense_2')
      ])
      original.save(f.name)
      try:
        loaded = model_loader.load_model(f.name)
        self.assertEqual(len(loaded.layers), 2)
        self.assertEqual(loaded.layers[0].name, 'dense_1')
      finally:
        os.unlink(f.name)

  def test_load_savedmodel(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      model_path = os.path.join(tmpdir, 'saved_model')
      original = models.Sequential([
          layers.Dense(10, input_shape=(5,), name='dense_1'),
          layers.Dense(1, name='dense_2')
      ])
      original.save(model_path, save_format='tf')
      loaded = model_loader.load_model(model_path)
      self.assertEqual(len(loaded.layers), 2)


if __name__ == '__main__':
  test.main()
