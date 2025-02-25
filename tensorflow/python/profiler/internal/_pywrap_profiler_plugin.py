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
"""Alias from TF to profiler code as part of moving out of TF.
"""
from typing import Dict
from mlprofiler.pywrap import _pywrap_profiler_plugin


def monitor(arg0: str, arg1: int, arg2: int, arg3: bool) -> str:
  _pywrap_profiler_plugin.monitor(arg0, arg1, arg2, arg3)


def trace(
    arg0: str,
    arg1: str,
    arg2: str,
    arg3: bool,
    arg4: int,
    arg5: int,
    arg6: Dict,  # pylint: disable=g-bare-generic
) -> None:
  _pywrap_profiler_plugin.trace(arg0, arg1, arg2, arg3, arg4, arg5, arg6)


def xspace_to_tools_data(arg0: list, arg1: str, arg2: Dict = None) -> tuple:  # pylint: disable=g-bare-generic
  if not arg2:
    return _pywrap_profiler_plugin.xspace_to_tools_data(arg0, arg1)
  return _pywrap_profiler_plugin.xspace_to_tools_data(arg0, arg1, arg2)


def xspace_to_tools_data_from_byte_string(
    arg0: list, arg1: list, arg2: str, arg3: Dict  # pylint: disable=g-bare-generic
) -> tuple:  # pylint: disable=g-bare-generic
  return _pywrap_profiler_plugin.xspace_to_tools_data_from_byte_string(
      arg0, arg1, arg2, arg3
  )
