# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""The C API entry points a PluggableDevice needs must be defined, not only declared.

A plugin is a shared object that TensorFlow loads and that calls back into it.
It compiles against the public C headers and resolves those functions at load
time, so a function the headers declare and no shipped binary defines is not a
missing feature: it is a plugin that fails to load, or on a platform that binds
lazily, one that dies at the first call.

That is not hypothetical. Between 2.19.1 and 2.21.0 the fourteen entry points
of tensorflow/c/kernels_experimental.cc stopped being linked into any shipped
binary while their header went on declaring them, and nothing noticed for two
releases. They are the only way a plugin can reach a resource variable's
tensor, so every optimiser became unimplementable on a PluggableDevice.

This test is deliberately dumb: it asks the process whether each name resolves.
It does not call them, because calling them needs a kernel context.
"""

import ctypes

from tensorflow.python.platform import test


# Declared in tensorflow/c/kernels_experimental.h. A PluggableDevice needs
# these to implement any op that reads or writes a resource variable, which is
# every optimiser, the resource gather and scatter ops, and the reference
# assignments.
_KERNELS_EXPERIMENTAL = (
    "TF_AddNVariant",
    "TF_AssignRefVariable",
    "TF_AssignUpdateVariable",
    "TF_AssignVariable",
    "TF_DestroyTemporaryVariable",
    "TF_GetInputByName",
    "TF_GetInputTensorFromVariable",
    "TF_IsRefInput",
    "TF_MaybeLockVariableInputMutexesInOrder",
    "TF_OpKernelConstruction_GetAttrTensorShape",
    "TF_OpKernelContext_ForwardRefInputToRefOutput",
    "TF_ReleaseVariableInputLockHolder",
    "TF_TemporaryVariable",
    "TF_ZerosLikeVariant",
)

# Declared in tensorflow/c/kernels.h. These have never gone missing, because
# TensorFlow's own kernels call them, which is what pulls their object into the
# link. They are here as a control: if they fail too, the test is looking in
# the wrong place rather than finding a regression.
_KERNELS = (
    "TF_NewKernelBuilder",
    "TF_RegisterKernelBuilder",
    "TF_KernelBuilder_TypeConstraint",
    "TF_KernelBuilder_HostMemory",
    "TF_AllocateOutput",
    "TF_GetStream",
)


class PluggableDeviceCApiTest(test.TestCase):

  def setUp(self):
    super().setUp()
    try:
      # RTLD_DEFAULT: everything the process has loaded, which is where a
      # plugin's own references are resolved from.
      self._process = ctypes.CDLL(None)
    except OSError:
      self.skipTest("this platform has no flat symbol namespace to search")

  def _assert_defined(self, names, why):
    missing = []
    for name in names:
      try:
        getattr(self._process, name)
      except AttributeError:
        missing.append(name)
    self.assertEmpty(
        missing,
        f"{len(missing)} of {len(names)} entry points are declared by the "
        f"public C headers and defined by no binary in this build: "
        f"{', '.join(missing)}. {why}")

  def testKernelCApiIsDefined(self):
    self._assert_defined(
        _KERNELS,
        "These are called by TensorFlow's own kernels, so if they are missing "
        "the test is looking in the wrong place.")

  def testExperimentalKernelCApiIsDefined(self):
    self._assert_defined(
        _KERNELS_EXPERIMENTAL,
        "Nothing inside TensorFlow calls these; they exist for plugins, so "
        "nothing incidental pulls their object into the link. A "
        "PluggableDevice cannot reach a resource variable without them.")


if __name__ == "__main__":
  test.main()
