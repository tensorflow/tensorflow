# pylint: disable=g-bad-file-header
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A wrapper of Session API which runs hooks (deprecated).

These are deprecated aliases for classes and functions in `tf.train`. Please use
those directly.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training import monitored_session

# pylint: disable=invalid-name
Scaffold = monitored_session.Scaffold
SessionCreator = monitored_session.SessionCreator
ChiefSessionCreator = monitored_session.ChiefSessionCreator
WorkerSessionCreator = monitored_session.WorkerSessionCreator
MonitoredSession = monitored_session.MonitoredSession
# pylint: disable=invalid-name
