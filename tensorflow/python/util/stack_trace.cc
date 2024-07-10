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

#include <Python.h>

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "tensorflow/core/platform/stack_frame.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace {

// Returns C string from a Python string object. Handles Python2/3 strings.
// TODO(kkb): This is a generic Python utility function. factor out as a
// utility.
const char* GetPythonString(PyObject* o) {
  if (PyBytes_Check(o)) {
    return PyBytes_AsString(o);
  } else {
    return PyUnicode_AsUTF8(o);
  }
}

}  // namespace

namespace tensorflow {

ABSL_MUST_USE_RESULT
ABSL_ATTRIBUTE_HOT
std::shared_ptr<StackTrace> StackTrace::Capture(int limit) {
  DCHECK(PyGILState_Check());
  if (limit == -1) limit = std::numeric_limits<int>::max();

  StackTrace result;
#if PY_VERSION_HEX >= 0x030B0000
  PyFrameObject* oldframe;
  PyFrameObject* frame = PyThreadState_GetFrame(PyThreadState_GET());
  for (int i = 0; i < limit && frame != nullptr; oldframe = frame,
           frame = PyFrame_GetBack(frame), Py_DECREF(oldframe), ++i) {
    PyCodeObject* code_obj = PyFrame_GetCode(frame);
    DCHECK(code_obj != nullptr);

    int line_number = PyFrame_GetLineNumber(const_cast<PyFrameObject*>(frame));
    result.code_objs_.push_back(std::make_pair(code_obj, line_number));
  }
  Py_XDECREF(frame);
#else
  const PyFrameObject* frame = PyThreadState_GET()->frame;
  for (int i = 0; i < limit && frame != nullptr; frame = frame->f_back, ++i) {
    PyCodeObject* code_obj = frame->f_code;
    Py_XINCREF(code_obj);
    DCHECK(code_obj != nullptr);

    int line_number = PyFrame_GetLineNumber(const_cast<PyFrameObject*>(frame));
    result.code_objs_.push_back(std::make_pair(code_obj, line_number));
  }
#endif

  static absl::flat_hash_map<uint64_t, std::shared_ptr<StackTrace>>* cache =
      new absl::flat_hash_map<uint64_t, std::shared_ptr<StackTrace>>();

  uint64_t hash_code = result.hash();
  if (!cache->contains(hash_code)) {
    cache->insert(std::make_pair(
        hash_code, std::make_shared<StackTrace>(std::move(result))));
  }

  return cache->at(hash_code);
}

std::vector<StackFrame> StackTrace::ToStackFrames(
    const SourceMap& source_map, const StackTraceFilter& filtered,
    bool reverse_traversal, int limit) const {
  auto gil_state = PyGILState_Ensure();
  std::vector<StackFrame> result;
  result.reserve(code_objs_.size());

  if (limit == -1) limit = std::numeric_limits<int>::max();

  for (int i = 0; i < code_objs_.size(); i++) {
    int idx = reverse_traversal ? i : code_objs_.size() - 1 - i;

    const std::pair<PyCodeObject*, int>& code_obj = code_objs_[idx];
    const char* file_name = GetPythonString(code_obj.first->co_filename);
    const int line_number = code_obj.second;

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

  PyGILState_Release(gil_state);
  return result;
}

}  // namespace tensorflow
