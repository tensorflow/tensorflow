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

std::vector<StackFrame> PythonStackTrace::ToStackFrames(
    const SourceMap& source_map, const StackTraceFilter& filtered,
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

    const auto it = source_map.find(SourceLoc{file_name, line_number});
    if (it != source_map.end()) {
      result.push_back(it->second);
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

PythonStackTrace* PythonStackTraceManager::Get(int id) {
  DCheckPyGilStateForStackTrace();
  if (next_id_ - id > kStackTraceCircularBufferSize) return nullptr;

  return &stack_traces_[id & (kStackTraceCircularBufferSize - 1)];
}

PythonStackTraceManager* const python_stack_trace_manager =
    new PythonStackTraceManager();

std::string GetPythonStackTraceString() {
  DCHECK(Py_IsInitialized());
  PyGILState_STATE py_threadstate;
  py_threadstate = PyGILState_Ensure();
  PyObject* m_dict = PyDict_New();
  PyDict_SetItemString(m_dict, "__builtins__", PyEval_GetBuiltins());
  PyMapping_SetItemString(m_dict, "traceback",
                          PyImport_ImportModule("traceback"));
  PyObject* trace = PyRun_String("''.join(traceback.format_stack())",
                                 Py_eval_input, m_dict, m_dict);
  std::string trace_string(GetPythonString(trace));
  PyGILState_Release(py_threadstate);
  Py_DECREF(m_dict);
  Py_DECREF(trace);
  return trace_string;
}

}  // namespace tensorflow
