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

This test is deliberately dumb: it asks the shipped binaries whether each name
resolves. It does not call them, because calling them needs a kernel context.
"""

import ctypes
import glob
import os
import sys

# Imported for its side effect: it is what loads the extension module the
# symbols are looked up in.
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.platform import sysconfig
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

# The extension module pywrap_tensorflow loads. It is the primary place to
# look: it is what a plugin is linked against, and resolving a name through its
# handle searches it and the libraries it depends on, which is where the C API
# lands in a shared-object build.
#
# The process-wide namespace is not the place to look, and asking it is how
# this test first got the answer wrong. pywrap_tensorflow loads that extension
# with RTLD_LOCAL deliberately, so that TensorFlow's statically linked LLVM
# does not leak into a process that may later import its own; nothing of
# TensorFlow's is in RTLD_DEFAULT to find. On Windows ctypes cannot even be
# asked: CDLL(None) is a TypeError there rather than an OSError, which is what
# turned the first version of this test into an error instead of a failure.
_RUNTIME_MODULE = "tensorflow.python._pywrap_tensorflow_internal"


def _candidate_libraries():
  """Paths of the binaries a plugin's undefined references resolve against."""
  paths = []
  module = sys.modules.get(_RUNTIME_MODULE)
  path = getattr(module, "__file__", None)
  if path:
    paths.append(path)

  # A monolithic build keeps the C API in the extension module above. A
  # shared-object build puts it in libtensorflow_framework, which that module
  # depends on, so the handle above already reaches it; naming it as well
  # covers a layout where it does not.
  library_dir = sysconfig.get_lib()
  if sys.platform == "win32":
    patterns = ("*tensorflow*.dll",)
  elif sys.platform == "darwin":
    patterns = ("*tensorflow_framework*.dylib*",)
  else:
    patterns = ("*tensorflow_framework*.so*",)
  for pattern in patterns:
    paths.extend(sorted(glob.glob(os.path.join(library_dir, pattern))))
  return paths


class PluggableDeviceCApiTest(test.TestCase):

  def setUp(self):
    super().setUp()
    self._libraries = []
    for path in _candidate_libraries():
      try:
        self._libraries.append(ctypes.CDLL(path))
      except OSError:
        # A path that is not loadable on this platform tells us nothing; the
        # test fails below if no library at all defines the names.
        continue
    if not self._libraries:
      self.skipTest("no TensorFlow shared library to resolve symbols against")

  def _assert_defined(self, names, why):
    missing = [
        name for name in names
        if not any(getattr(library, name, None) for library in self._libraries)
    ]
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
