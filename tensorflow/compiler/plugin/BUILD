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

"""Configuration file for an XLA plugin.

  please don't check in changes to this file. to prevent changes appearing
  in git status, use:

  git update-index --assume-unchanged tensorflow/compiler/plugin/BUILD

  To add additional devices to the XLA subsystem, add targets to the
  dependency list in the 'plugin' target. For instance:

    deps = ["//tensorflow/compiler/plugin/example:plugin_lib"],

  ** Please don't remove this file - it is supporting some 3rd party plugins **
"""

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

# This target is for plugins/extensions to add their TF device, and XLA backend
# into the system. The depenency comes via the jit module.  The target should
# include the creation of a XlaDevice, and a factory registered with
# REGISTER_LOCAL_DEVICE_FACTORY.  It should also include the XLA compiler and
# associated platform, registered with xla::Compiler::RegisterCompilerFactory.
# See tensorflow/compiler/jit/xla_cpu_device.cc,
# tensorflow/compiler/xla/service/cpu/cpu_compiler.cc and
# tensorflow/stream_executor/host/host_platform.cc for examples.
cc_library(
    name = "plugin",
    deps = [
        "//tensorflow/compiler/plugin/poplar:poplar_lib",
        #"//tensorflow/compiler/plugin/example:example_lib",
    ],
)

# This target is for plugins/extensions to add their backend registration files.
# The dependency comes via the tf2xla module, and should register the backend
# using the REGISTER_XLA_BACKEND macro.  See tf2xla/xla_cpu_backend.cc for an
# example.
cc_library(
    name = "plugin_backend",
    deps = [
        "//tensorflow/compiler/plugin/poplar:poplar_backend",
        #"//tensorflow/compiler/plugin/example:example_backend",
    ],
)

# This target is added purely for the purpose of ensuring that `:xla_device` is
# always publicly visible to external XLA backend/plugin developers.
cc_library(
    name = "plugin_device",
    deps = [
        "//tensorflow/compiler/jit:xla_device",
    ],
)
