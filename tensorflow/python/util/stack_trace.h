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

#ifndef TENSORFLOW_PYTHON_UTIL_STACK_TRACE_H_
#define TENSORFLOW_PYTHON_UTIL_STACK_TRACE_H_

#include <Python.h>
#include <frameobject.h>

#include <array>
#include <sstream>
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/types/optional.h"
#include "tensorflow/core/util/abstract_stack_trace.h"
#include "tensorflow/python/lib/core/py_util.h"

namespace tensorflow {

// A class for capturing Python stack trace.
class StackTrace final {
 public:
  static constexpr int kMaxDepth = 10;

  StackTrace() : size_(0) {}

  // Returns `StackTrace` object that captures the current Python stack trace.
  // Python GIL must be acquired beforehand.
  ABSL_MUST_USE_RESULT
  ABSL_ATTRIBUTE_HOT
  static StackTrace Capture() {
    DCheckPyGilState();

    StackTrace result;
    const PyFrameObject* frame = PyThreadState_GET()->frame;
    int i = 0;
    for (; i < kMaxDepth && frame != nullptr; frame = frame->f_back, ++i) {
      PyCodeObject* code_obj = frame->f_code;
      DCHECK(code_obj != nullptr);

      Py_INCREF(code_obj);
      result.code_objs_[i] = code_obj;
      result.last_instructions_[i] = frame->f_lasti;
    }
    result.size_ = i;
    return result;
  }

  // Python GIL must be acquired beforehand.
  ABSL_ATTRIBUTE_HOT
  ~StackTrace() { Clear(); }

  StackTrace(StackTrace&& other) {
    code_objs_ = other.code_objs_;
    last_instructions_ = other.last_instructions_;
    size_ = other.size_;
    other.size_ = 0;
  }

  // Python GIL must be acquired beforehand.
  ABSL_ATTRIBUTE_HOT
  StackTrace& operator=(StackTrace&& other) {
    Clear();

    code_objs_ = other.code_objs_;
    last_instructions_ = other.last_instructions_;
    size_ = other.size_;
    other.size_ = 0;
    return *this;
  }

  // Returns a structured representation of the captured stack trace.
  std::vector<StackFrame> ToStackFrames() const;

 private:
  std::array<PyCodeObject*, kMaxDepth> code_objs_;
  std::array<int, kMaxDepth> last_instructions_;
  int size_;

  // Python GIL must be acquired beforehand.
  ABSL_ATTRIBUTE_HOT
  void Clear() {
    DCheckPyGilState();
    for (int i = 0; i < size_; ++i) Py_DECREF(code_objs_[i]);
  }

  StackTrace(const StackTrace&) = delete;
  StackTrace& operator=(const StackTrace&) = delete;
};

// A class that manages Python stack traces in a circular buffer. Users can
// insert stack trace entries and retrive them by ids.
class StackTraceManager {
 public:
  static constexpr int kStackTraceCircularBufferSize = 1024;

  // Captures the current Python stack trace and returns an id.
  // Python GIL must be acquired beforehand.
  ABSL_MUST_USE_RESULT
  ABSL_ATTRIBUTE_HOT
  int Capture() {
    DCheckPyGilState();
    const int id = next_id_++;
    const int index = id & (kStackTraceCircularBufferSize - 1);
    stack_traces_[index] = StackTrace::Capture();
    return id;
  }

  // Retrieve captured Python stack trace by id. Returns `nullptr` if the
  // requested stack trace is evicted from the circular buffer.
  // Python GIL must be acquired beforehand.
  ABSL_MUST_USE_RESULT
  StackTrace* Get(int id);

 private:
  int next_id_ = 0;
  std::array<StackTrace, kStackTraceCircularBufferSize> stack_traces_;
};

// Singleton StackTraceManager.
extern StackTraceManager* const stack_trace_manager;

// Returns Python stack trace object that can be converted to string.
// Note that the actual stack trace is kept in a circular buffer for string
// conversion could fail if it's evicted before.
// Python GIL must be acquired beforehand.
inline AbstractStackTrace GetStackTrace() {
  DCheckPyGilState();
  return AbstractStackTrace(stack_trace_manager->Capture(), [](int id) {
    PyGILState_STATE gstate = PyGILState_Ensure();
    std::vector<StackFrame> result =
        stack_trace_manager->Get(id)->ToStackFrames();
    PyGILState_Release(gstate);
    return result;
  });
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_STACK_TRACE_H_
