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
# =============================================================================
"""Provides python test rules for Cloud TPU."""

load(
    "//tensorflow/python/tpu:tpu_test_wrapper.bzl",
    _get_kwargs_for_wrapping = "get_kwargs_for_wrapping",
)

def tpu_py_test(
        name,
        tags = None,
        disable_v2 = False,
        disable_v3 = False,
        disable_experimental = False,
        disable_mlir_bridge = True,
        disable_tfrt = None,
        args = [],
        **kwargs):
    """Generates identical unit test variants for various Cloud TPU versions.

    TODO(rsopher): actually generate v2 vs v3 tests.

    Args:
        name: Name of test. Will be prefixed by accelerator versions.
        tags: BUILD tags to apply to tests.
        disable_v2: If true, don't generate TPU v2 tests.
        disable_v3: If true, don't generate TPU v3 tests.
        disable_experimental: Unused.
        disable_mlir_bridge: Unused.
        disable_tfrt: Unused.
        args: Arguments to apply to tests.
        **kwargs: Additional named arguments to apply to tests.
    """

    native.py_test(
        **_get_kwargs_for_wrapping(
            name,
            tags,
            args,
            **kwargs
        )
    )
