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
"""Compiles the embedded Metal shader source through the Metal runtime.

clang never sees this source: it is a string in metal_shader_library.mm that
the backend hands to newLibraryWithSource: at first use. A syntax error in it
is therefore invisible to every check except running on a machine with a GPU,
which is what this script is for.
"""

import ctypes
import ctypes.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "src/tensorflow/core/common_runtime/metal/kernels" /
          "metal_shader_library.mm")


def extract_shader_source(text: str) -> str:
  """Pulls the raw string literal holding the Metal source."""
  match = re.search(r'R"(\w*)\((.*?)\)\1"', text, re.S)
  if match is None:
    sys.exit("no raw string literal found in metal_shader_library.mm")
  return match.group(2)


def main() -> int:
  shader = extract_shader_source(SOURCE.read_text())
  print(f"shader source: {len(shader)} bytes, "
        f"{shader.count('kernel void')} kernels")

  objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
  ctypes.cdll.LoadLibrary(ctypes.util.find_library("Metal"))
  objc.objc_getClass.restype = ctypes.c_void_p
  objc.sel_registerName.restype = ctypes.c_void_p
  objc.objc_msgSend.restype = ctypes.c_void_p
  objc.objc_msgSend.argtypes = [ctypes.c_void_p] * 2

  create = ctypes.CDLL(None).MTLCreateSystemDefaultDevice
  create.restype = ctypes.c_void_p
  device = create()
  if not device:
    sys.exit("no Metal device on this machine")

  # NSString from the shader source, then newLibraryWithSource:options:error:.
  nsstring = ctypes.c_void_p(objc.objc_getClass(b"NSString"))
  sel = ctypes.c_void_p(objc.sel_registerName(b"stringWithUTF8String:"))
  msg = objc.objc_msgSend
  msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
  source = ctypes.c_void_p(msg(nsstring, sel, shader.encode()))

  # The same compile options the backend uses, or this checks something else.
  # MTLLanguageVersion3_0 is (3 << 16) | 0; the float overload of
  # atomic_fetch_add_explicit only exists from Metal 3.0, and the default
  # version follows the toolchain rather than the source.
  options_class = ctypes.c_void_p(objc.objc_getClass(b"MTLCompileOptions"))
  msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  options = ctypes.c_void_p(
      msg(ctypes.c_void_p(msg(options_class,
                              ctypes.c_void_p(
                                  objc.sel_registerName(b"alloc")))),
          ctypes.c_void_p(objc.sel_registerName(b"init"))))
  set_version = ctypes.c_void_p(
      objc.sel_registerName(b"setLanguageVersion:"))
  msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
  msg(options, set_version, (3 << 16) | 0)

  error = ctypes.c_void_p(0)
  sel = ctypes.c_void_p(
      objc.sel_registerName(b"newLibraryWithSource:options:error:"))
  msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
  library = msg(ctypes.c_void_p(device), sel, source, options,
                ctypes.byref(error))
  if not library:
    description = ctypes.c_void_p(0)
    if error:
      sel_desc = ctypes.c_void_p(objc.sel_registerName(b"localizedDescription"))
      msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
      description = ctypes.c_void_p(msg(error, sel_desc))
      sel_utf8 = ctypes.c_void_p(objc.sel_registerName(b"UTF8String"))
      msg.restype = ctypes.c_char_p
      print(msg(description, sel_utf8).decode(), file=sys.stderr)
    sys.exit("the Metal shader library failed to compile")
  print("the Metal shader library compiles")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
