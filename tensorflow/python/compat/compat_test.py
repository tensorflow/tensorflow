# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for forward and backwards compatibility utilties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
from tensorflow.python.compat import compat
from tensorflow.python.platform import test


class CompatTest(test.TestCase):

  def _compatibility_date(self):
    date = compat._FORWARD_COMPATIBILITY_HORIZON  # pylint: disable=protected-access
    return (date.year, date.month, date.day)

  def _n_days_after(self, n):
    date = compat._FORWARD_COMPATIBILITY_HORIZON + datetime.timedelta(days=n)  # pylint: disable=protected-access
    return (date.year, date.month, date.day)

  def test_basic(self):
    compatibility_date = self._compatibility_date()
    one_day_before = self._n_days_after(-1)
    self.assertTrue(compat.forward_compatible(*one_day_before))
    self.assertFalse(compat.forward_compatible(*compatibility_date))

  def test_decorator(self):
    compatibility_date = self._compatibility_date()
    one_day_after = self._n_days_after(1)
    with compat.forward_compatibility_horizon(*one_day_after):
      self.assertTrue(compat.forward_compatible(*compatibility_date))
      self.assertFalse(compat.forward_compatible(*one_day_after))

    # After exiting context manager, value should be reset.
    self.assertFalse(compat.forward_compatible(*compatibility_date))

  def test_decorator_with_failure(self):
    compatibility_date = self._compatibility_date()
    one_day_after = self._n_days_after(1)

    class DummyError(Exception):
      pass

    try:
      with compat.forward_compatibility_horizon(*one_day_after):
        raise DummyError()
    except DummyError:
      pass  # silence DummyError

    # After exiting context manager, value should be reset.
    self.assertFalse(compat.forward_compatible(*compatibility_date))

  def test_environment_override(self):
    var_name = 'TF_FORWARD_COMPATIBILITY_DELTA_DAYS'

    def remove_os_environment_var():
      try:
        del os.environ[var_name]
      except KeyError:
        pass

    self.addCleanup(remove_os_environment_var)

    compatibility_date = self._compatibility_date()
    one_day_before = self._n_days_after(-1)
    one_day_after = self._n_days_after(1)
    ten_days_after = self._n_days_after(10)
    nine_days_after = self._n_days_after(9)

    self.assertTrue(compat.forward_compatible(*one_day_before))
    self.assertFalse(compat.forward_compatible(*compatibility_date))
    self.assertFalse(compat.forward_compatible(*one_day_after))
    self.assertFalse(compat.forward_compatible(*nine_days_after))
    self.assertFalse(compat.forward_compatible(*ten_days_after))

    os.environ[var_name] = '10'
    compat._update_forward_compatibility_date_number()
    self.assertTrue(compat.forward_compatible(*one_day_before))
    self.assertTrue(compat.forward_compatible(*compatibility_date))
    self.assertTrue(compat.forward_compatible(*one_day_after))
    self.assertTrue(compat.forward_compatible(*nine_days_after))
    self.assertFalse(compat.forward_compatible(*ten_days_after))

    del os.environ[var_name]
    compat._update_forward_compatibility_date_number()
    self.assertTrue(compat.forward_compatible(*one_day_before))
    self.assertFalse(compat.forward_compatible(*compatibility_date))
    self.assertFalse(compat.forward_compatible(*one_day_after))
    self.assertFalse(compat.forward_compatible(*nine_days_after))
    self.assertFalse(compat.forward_compatible(*ten_days_after))

    # Now test interaction between environment variable and context func.
    os.environ[var_name] = '10'
    compat._update_forward_compatibility_date_number()
    self.assertTrue(compat.forward_compatible(*one_day_after))
    with compat.forward_compatibility_horizon(*one_day_after):
      self.assertTrue(compat.forward_compatible(*one_day_before))
      self.assertTrue(compat.forward_compatible(*compatibility_date))
      self.assertFalse(compat.forward_compatible(*one_day_after))
      self.assertFalse(compat.forward_compatible(*nine_days_after))
      self.assertFalse(compat.forward_compatible(*ten_days_after))
    self.assertTrue(compat.forward_compatible(*one_day_after))


if __name__ == '__main__':
  test.main()
