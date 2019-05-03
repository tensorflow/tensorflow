# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Module with print function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# This import is useless, but serves to distinguish this module's future imports
# from the standard set of future imports used in TensorFlow.
from __future__ import with_statement


def f():
  print('foo')


lambda_f = lambda: None


class Foo(object):

  def f(self):
    print('foo')
