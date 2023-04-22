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
"""Test for SaveContext."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.eager import test
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options


class SaveContextTest(test.TestCase):

  def test_multi_thread(self):
    self.assertFalse(save_context.in_save_context())
    with self.assertRaisesRegex(ValueError, 'not in a SaveContext'):
      save_context.get_save_options()

    options = save_options.SaveOptions(save_debug_info=True)
    with save_context.save_context(options):
      self.assertTrue(save_context.in_save_context())
      self.assertTrue(save_context.get_save_options().save_debug_info)

      entered_context_in_thread = threading.Event()
      continue_thread = threading.Event()

      def thread_fn():
        self.assertFalse(save_context.in_save_context())
        with self.assertRaisesRegex(ValueError, 'not in a SaveContext'):
          save_context.get_save_options()

        options = save_options.SaveOptions(save_debug_info=False)
        with save_context.save_context(options):
          self.assertTrue(save_context.in_save_context())
          # save_debug_info has a different value in this thread.
          self.assertFalse(save_context.get_save_options().save_debug_info)
          entered_context_in_thread.set()
          continue_thread.wait()

        self.assertFalse(save_context.in_save_context())
        with self.assertRaisesRegex(ValueError, 'not in a SaveContext'):
          save_context.get_save_options()

      t = threading.Thread(target=thread_fn)
      t.start()

      entered_context_in_thread.wait()
      # Another thread shouldn't affect this thread.
      self.assertTrue(save_context.in_save_context())
      self.assertTrue(save_context.get_save_options().save_debug_info)

      continue_thread.set()
      t.join()
      # Another thread exiting SaveContext shouldn't affect this thread.
      self.assertTrue(save_context.in_save_context())
      self.assertTrue(save_context.get_save_options().save_debug_info)

    self.assertFalse(save_context.in_save_context())
    with self.assertRaisesRegex(ValueError, 'not in a SaveContext'):
      save_context.get_save_options()

  def test_enter_multiple(self):
    options = save_options.SaveOptions()
    with self.assertRaisesRegex(ValueError, 'already in a SaveContext'):
      with save_context.save_context(options):
        with save_context.save_context(options):
          pass


if __name__ == '__main__':
  test.main()
