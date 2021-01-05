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

#include <limits>

#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/stringpiece.h"

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

std::vector<StackFrame> StackTrace::ToStackFrames(
    const StackTraceMap& mapper, const StackTraceFilter& filtered,
    bool reverse_traversal, int limit) const {
  DCheckPyGilStateForStackTrace();
  std::vector<StackFrame> result;
  result.reserve(code_objs_.size());

  if (limit == -1) limit = std::numeric_limits<int>::max();

  for (int i = 0; i < code_objs_.size(); i++) {
    int idx = reverse_traversal ? i : code_objs_.size() - 1 - i;

    const std::pair<PyCodeObject*, int>& code_obj = code_objs_[idx];
    const char* file_name = GetPythonString(code_obj.first->co_filename);
    const int line_number = PyCode_Addr2Line(code_obj.first, code_obj.second);

    if (filtered && filtered(file_name)) {
      continue;
    }

    absl::optional<StackFrame> mapped =
        mapper ? mapper(std::make_pair(file_name, line_number)) : absl::nullopt;

    if (mapped) {
      result.push_back(*mapped);
    } else {
      result.emplace_back(StackFrame{file_name, line_number,
                                     GetPythonString(code_obj.first->co_name)});
    }

    if (result.size() == limit) {
      break;
    }
  }

  return result;
}

StackTrace* StackTraceManager::Get(int id) {
  DCheckPyGilStateForStackTrace();
  if (next_id_ - id > kStackTraceCircularBufferSize) return nullptr;

  return &stack_traces_[id & (kStackTraceCircularBufferSize - 1)];
}

StackTraceManager* const stack_trace_manager = new StackTraceManager();

}  // namespace tensorflow
