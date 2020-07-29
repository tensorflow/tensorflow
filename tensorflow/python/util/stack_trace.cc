/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/python/util/stack_trace.h"

namespace {

// Returns C string from a Python string object. Handles Python2/3 strings.
// TODO(kkb): This is a generic Python utility function. factor out as a
// utility.
const char* GetPythonString(PyObject* o) {
#if PY_MAJOR_VERSION >= 3
  if (PyBytes_Check(o)) {
    return PyBytes_AsString(o);
  } else {
    return PyUnicode_AsUTF8(o);
  }
#else
  return PyBytes_AsString(o);
#endif
}
}  // namespace

namespace tensorflow {
std::string StackTrace::ToString() const {
  DCheckPyGilState();

  std::ostringstream result;
  for (int i = size_ - 1; i >= 0; --i) {
    result << "  File \"" << PyUnicode_AsUTF8(code_objs_[i]->co_filename)
           << "\", line "
           << PyCode_Addr2Line(code_objs_[i], last_instructions_[i]) << ", in "
           << GetPythonString(code_objs_[i]->co_name)
           << "\n    <source line unimplemented>\n";
    // TODO(kkb): Add source code line.  See tf_stack.cc's
    // FrameSummary::line() function.
  }
  return result.str();
}
}  // namespace tensorflow
