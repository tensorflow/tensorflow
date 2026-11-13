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
"""The Flex delegate entry point tf.lite.Interpreter relies on must be defined.

A model converted with tf.lite.OpsSet.SELECT_TF_OPS runs its TensorFlow ops
through the Flex delegate, and tf.lite.Interpreter never names that delegate:
InterpreterBuilder calls a weak tflite::AcquireFlexDelegate() whose strong
override, and the exported TF_AcquireFlexDelegate() next to it, come from the
delegate's own object file. When that object file is not linked into the
shipped binaries nothing fails to build or to import; every SELECT_TF_OPS model
instead fails in allocate_tensors() with "Select TensorFlow op(s) ... not
supported by this interpreter", which is what 2.20.0 shipped with.

This test asks the shipped binaries whether the entry point resolves. It does
not call it, because acquiring the delegate initializes a TensorFlow runtime.
"""

import ctypes
import glob
import os
import sys
import unittest

# Imported for its side effect: it is what loads the extension module the
# symbols are looked up in.
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.platform import sysconfig
from tensorflow.python.platform import test

# Defined in tensorflow/lite/delegates/flex/delegate_symbol.cc, which only
# makes it into a binary that links the Flex delegate.
_FLEX_DELEGATE = ("TF_AcquireFlexDelegate",)

# Declared in tensorflow/c/c_api.h and called by TensorFlow itself, so they
# have never gone missing. They are here as a control: if they fail too, the
# test is looking in the wrong place rather than finding a regression.
_C_API = (
    "TF_Version",
    "TF_NewStatus",
    "TF_DeleteStatus",
)

# The extension module pywrap_tensorflow loads. Resolving a name through its
# handle searches it and the libraries it depends on, which in a shared-object
# build includes libtensorflow_cc, where InterpreterBuilder lives and where the
# delegate has to land for the weak symbol to be overridden.
_RUNTIME_MODULE = (
    "tensorflow.python._pywrap_tensorflow_internal"
)


def _candidate_libraries():
  """Paths of the binaries tf.lite.Interpreter's references resolve against."""
  paths = []
  module = sys.modules.get(_RUNTIME_MODULE)
  path = getattr(module, "__file__", None)
  if path:
    paths.append(path)

  # A monolithic build keeps everything in the extension module above. A
  # shared-object build puts InterpreterBuilder and the delegate in
  # libtensorflow_cc and the C API in libtensorflow_framework, both of which
  # that module depends on, so its handle already reaches them; naming them as
  # well covers a layout where it does not.
  library_dir = sysconfig.get_lib()
  if sys.platform == "darwin":
    patterns = ("*tensorflow_cc*.dylib*", "*tensorflow_framework*.dylib*")
  else:
    patterns = ("*tensorflow_cc*.so*", "*tensorflow_framework*.so*")
  for pattern in patterns:
    paths.extend(sorted(glob.glob(os.path.join(library_dir, pattern))))
  return paths


# On Windows a name is reachable only if the module definition file exported
# it, so asking the loaded .pyd answers a different question (see
# pluggable_device_c_api_test.py in tensorflow/python/framework).
@unittest.skipIf(
    sys.platform == "win32",
    "symbol resolution on Windows goes through an export table, not the link",
)
class FlexDelegateSymbolTest(test.TestCase):

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
        name
        for name in names
        if not any(getattr(library, name, None) for library in self._libraries)
    ]
    self.assertEmpty(
        missing,
        f"{len(missing)} of {len(names)} entry points are defined by no "
        f"binary in this build: {', '.join(missing)}. {why}",
    )

  def testCApiIsDefined(self):
    self._assert_defined(
        _C_API,
        "These are called by TensorFlow itself, so if they are missing the "
        "test is looking in the wrong place.",
    )

  def testFlexDelegateIsDefined(self):
    self._assert_defined(
        _FLEX_DELEGATE,
        "Nothing inside TensorFlow refers to the Flex delegate by name, so "
        "nothing incidental pulls its object into the link, and without it "
        "tf.lite.Interpreter cannot run a SELECT_TF_OPS model.",
    )


if __name__ == "__main__":
  test.main()
