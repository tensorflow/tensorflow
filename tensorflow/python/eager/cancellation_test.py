# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import threading
from tensorflow.python.eager import cancellation
from tensorflow.python.platform import test


class CancellationTest(test.TestCase):

  def test_start_cancel(self):
    manager = cancellation.CancellationManager()
    self.assertFalse(manager.is_cancelled)
    manager.start_cancel()
    self.assertTrue(manager.is_cancelled)

  def test_threading(self):
    manager1 = cancellation.CancellationManager()
    manager2 = cancellation.CancellationManager()
    thread_exceptions = []

    def thread_fn():
      try:
        with cancellation.CancellationManagerContext(manager2):
          self.assertEqual(cancellation.context(), manager2)
      except Exception as e:  # pylint: disable=broad-exception-caught
        thread_exceptions.append(e)

    with cancellation.CancellationManagerContext(manager1):
      self.assertEqual(cancellation.context(), manager1)
      t = threading.Thread(target=thread_fn)
      t.start()
      t.join()
      for exc in thread_exceptions:
        raise exc
      self.assertEqual(cancellation.context(), manager1)

  def test_nested_context(self):
    manager1 = cancellation.CancellationManager()
    manager2 = cancellation.CancellationManager()

    self.assertIsNone(cancellation.context())
    with cancellation.CancellationManagerContext(manager1):
      self.assertEqual(cancellation.context(), manager1)
      with cancellation.CancellationManagerContext(manager2):
        self.assertEqual(cancellation.context(), manager2)
      self.assertEqual(cancellation.context(), manager1)
    self.assertIsNone(cancellation.context())

  def test_get_cancelable_function(self):
    manager = cancellation.CancellationManager()

    def my_fn(x):
      """My docstring."""
      self.assertEqual(cancellation.context(), manager)
      return x + 1

    cancelable_fn = manager.get_cancelable_function(my_fn)
    self.assertEqual(cancelable_fn.__name__, "my_fn")
    self.assertEqual(cancelable_fn.__doc__, "My docstring.")
    self.assertEqual(cancelable_fn(5), 6)


if __name__ == "__main__":
  test.main()
