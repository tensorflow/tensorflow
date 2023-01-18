# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Minimal setup.py that can build an sdist, but warns on install."""

import sys

import setuptools

TF_REMOVAL_WARNING = """

=========================================================
The "tf-nightly-gpu" package has been removed!

Please install "tf-nightly" instead.

Other than the name, the two packages have been identical
since tf-nightly 2.1, or roughly since Sep 2019. For more
information, see: pypi.org/project/tf-nightly-gpu
=========================================================

"""

# Cover all "pip install" situations
if "bdist_wheel" in sys.argv or "install" in sys.argv or "bdist_egg" in sys.argv:
  raise Exception(TF_REMOVAL_WARNING)

if __name__ == "__main__":
  setuptools.setup()
