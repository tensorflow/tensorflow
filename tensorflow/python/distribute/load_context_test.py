# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for load_context."""

import threading

from tensorflow.python.distribute import load_context
from tensorflow.python.eager import test


class LoadContextTest(test.TestCase):

  def test_default_state(self):
    self.assertFalse(load_context.in_load_context())
    self.assertIsNone(load_context.get_load_options())

  def test_single_context(self):
    options = object()
    with load_context.load_context(options):
      self.assertTrue(load_context.in_load_context())
      self.assertIs(load_context.get_load_options(), options)
    self.assertFalse(load_context.in_load_context())
    self.assertIsNone(load_context.get_load_options())

  def test_nested_context(self):
    outer_options = object()
    inner_options = object()
    with load_context.load_context(outer_options):
      self.assertTrue(load_context.in_load_context())
      self.assertIs(load_context.get_load_options(), outer_options)
      with load_context.load_context(inner_options):
        self.assertTrue(load_context.in_load_context())
        self.assertIs(load_context.get_load_options(), inner_options)
      self.assertTrue(load_context.in_load_context())
      self.assertIs(load_context.get_load_options(), outer_options)
    self.assertFalse(load_context.in_load_context())
    self.assertIsNone(load_context.get_load_options())

  def test_nested_context_with_none(self):
    outer_options = object()
    with load_context.load_context(outer_options):
      with load_context.load_context(None):
        self.assertTrue(load_context.in_load_context())
        self.assertIsNone(load_context.get_load_options())
      self.assertTrue(load_context.in_load_context())
      self.assertIs(load_context.get_load_options(), outer_options)
    self.assertFalse(load_context.in_load_context())
    self.assertIsNone(load_context.get_load_options())

  def test_clear_load_options_empty_stack_raises(self):
    with self.assertRaisesRegex(
        RuntimeError,
        r"clear_load_options\(\) called when load context stack is empty\.",
    ):
      load_context._load_context.clear_load_options()

  def test_thread_isolation(self):
    options = object()
    errors = []

    def thread_fn():
      try:
        if load_context.in_load_context():
          errors.append("Expected not in_load_context in background thread")
        if load_context.get_load_options() is not None:
          errors.append("Expected None load_options in background thread")
      except Exception as e:  # pylint: disable=broad-except
        errors.append(str(e))

    with load_context.load_context(options):
      t = threading.Thread(target=thread_fn)
      t.start()
      t.join()

    self.assertEmpty(errors)

  def test_exception_unwinds_stack(self):
    outer_options = object()
    inner_options = object()
    with load_context.load_context(outer_options):
      try:
        with load_context.load_context(inner_options):
          raise ValueError("Simulated failure")
      except ValueError:
        pass
      self.assertTrue(load_context.in_load_context())
      self.assertIs(load_context.get_load_options(), outer_options)
    self.assertFalse(load_context.in_load_context())
    self.assertIsNone(load_context.get_load_options())


if __name__ == "__main__":
  test.main()
