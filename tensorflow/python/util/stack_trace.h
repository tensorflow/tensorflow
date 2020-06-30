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
      DCHECK(frame->f_trace == nullptr);
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

  // Returns string representation of the captured stack trace.
  std::string ToString() const;

  // TODO(kkb): Implement structured stack trace object getter.

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

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_STACK_TRACE_H_
