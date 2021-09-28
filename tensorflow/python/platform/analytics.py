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
"""Analytics helpers library."""

def track_usage(tool_id, tags):
  """No usage tracking for external library.

  Args:
    tool_id: A string identifier for tool to be tracked.
    tags: list of string tags that will be added to the tracking.
  """
  del tool_id, tags  # Unused externally.
