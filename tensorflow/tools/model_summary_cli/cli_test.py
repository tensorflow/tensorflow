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
"""Tests for CLI module."""

import json
import os
import sys
import tempfile
from io import StringIO

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.platform import test

from tensorflow.tools.model_summary_cli import cli


class CreateParserTest(test.TestCase):

  def test_parser_creation(self):
    parser = cli.create_parser()
    self.assertIsNotNone(parser)

  def test_parse_basic_args(self):
    parser = cli.create_parser()
    args = parser.parse_args(['./model.h5'])
    self.assertEqual(args.model_path, './model.h5')
    self.assertIsNone(args.plot)
    self.assertFalse(args.json)

  def test_parse_with_plot(self):
    parser = cli.create_parser()
    args = parser.parse_args(['./model.h5', '--plot', 'out.png'])
    self.assertEqual(args.plot, 'out.png')

  def test_parse_with_json(self):
    parser = cli.create_parser()
    args = parser.parse_args(['./model.h5', '--json'])
    self.assertTrue(args.json)

  def test_parse_with_line_length(self):
    parser = cli.create_parser()
    args = parser.parse_args(['./model.h5', '--line-length', '120'])
    self.assertEqual(args.line_length, 120)


class RunSummaryTest(test.TestCase):

  def test_run_summary_json_output(self):
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
      model = models.Sequential([
          layers.Dense(10, input_shape=(5,), name='dense_1')
      ])
      model.save(f.name)

      try:
        parser = cli.create_parser()
        args = parser.parse_args([f.name, '--json'])

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        result = cli.run_summary(args)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        self.assertEqual(result, 0)
        # Verify JSON is valid
        data = json.loads(output)
        self.assertEqual(data['model_name'], 'sequential')
        self.assertEqual(len(data['layers']), 1)
      finally:
        os.unlink(f.name)

  def test_run_summary_nonexistent_model(self):
    parser = cli.create_parser()
    args = parser.parse_args(['/nonexistent/model'])

    # Capture stderr
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    result = cli.run_summary(args)

    sys.stderr = old_stderr

    self.assertEqual(result, 1)


if __name__ == '__main__':
  test.main()
