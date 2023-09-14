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

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tsl/platform/fingerprint.h"

namespace tensorflow {

// A class for capturing Python stack trace.
class StackTrace : public CapturedStackTrace {
 public:
  static constexpr int kStackTraceInitialSize = 30;

  StackTrace() = default;
  StackTrace(StackTrace&& other) = default;

  // Returns `StackTrace` object that captures the current Python stack trace.
  // `limit` determines how many stack frames at most are returned: set to -1
  // for "no limit".
  // Python GIL must be acquired beforehand.
  ABSL_MUST_USE_RESULT
  ABSL_ATTRIBUTE_HOT
  static std::shared_ptr<StackTrace> Capture(int limit);

  uint64_t hash() const {
    uint64_t hash = 0;
    for (const auto& p : code_objs_) {
      hash = tsl::FingerprintCat64(hash,
                                   reinterpret_cast<const uint64_t>(p.first));
      hash = tsl::FingerprintCat64(hash, p.second);
    }
    return hash;
  }

  ~StackTrace() override { Clear(); }

  // Returns a structured representation of the captured stack trace.
  // `source_map` provides a custom mapping for translating stack frames,
  // `filter` returns `true` for the stack frames which should be omitted.
  //
  // `reverse_traversal` changes the traversal order of the stack trace, and
  // `limit` bounds the number of returned frames (after filtering).
  std::vector<StackFrame> ToStackFrames(const SourceMap& source_map,
                                        const StackTraceFilter& filtered,
                                        bool reverse_traversal,
                                        int limit) const override;

 private:
  void Clear() {
    auto gil_state = PyGILState_Ensure();
    for (const auto& p : code_objs_) Py_DECREF(p.first);
    code_objs_.clear();
    PyGILState_Release(gil_state);
  }

  absl::InlinedVector<std::pair<PyCodeObject*, int>, 16> code_objs_;

  StackTrace(const StackTrace&) = delete;
  StackTrace& operator=(const StackTrace&) = delete;
};

inline std::shared_ptr<StackTrace> GetStackTrace(int limit = -1) {
  return StackTrace::Capture(limit);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_STACK_TRACE_H_
