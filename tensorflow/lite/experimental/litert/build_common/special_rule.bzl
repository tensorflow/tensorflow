# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""External versions of LiteRT build rules that differ outside of Google."""

def lite_rt_friends():
    """Internal visibility for packages outside of LiteRT code location.

    Return the package group declaration for internal code locations that need
    visibility to LiteRT APIs"""

    return []

def gles_deps():
    """This is a no-op outside of Google."""
    return []

def gles_linkopts():
    """This is a no-op outside of Google."""
    return []
