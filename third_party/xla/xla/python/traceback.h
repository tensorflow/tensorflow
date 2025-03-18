/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_TRACEBACK_H_
#define XLA_PYTHON_TRACEBACK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// placeholder for index annotation headers
#include "absl/container/inlined_vector.h"
#include "nanobind/nanobind.h"
#include "xla/python/nb_class_ptr.h"

namespace xla {

// Represents a Python traceback. This object is designed to be allocated on
// the Python heap; creating or destroying a traceback requires the GIL.
class Traceback {
 public:
  // Requires GIL. Creates a Traceback object that requires destructor to be
  // invoked with GIL held as well.
  static std::optional<nb_class_ptr<Traceback>> Get();

  // Requires GIL.
  static bool enabled() { return enabled_; }
  // Requires GIL.
  static void SetEnabled(bool enabled);

  // Requires GIL. Don't call this directly, you're looking for Get().
  Traceback();
  // Requires GIL.
  ~Traceback();

  Traceback(const Traceback&) = delete;
  Traceback(Traceback&& other) noexcept;
  Traceback& operator=(const Traceback&) = delete;
  Traceback& operator=(Traceback&&) = delete;

  // Requires the GIL be held.
  std::string ToString() const;

  struct Frame {
    nanobind::str file_name;
    nanobind::str function_name;
    int function_start_line;
    int line_num;

    std::string ToString() const;
  };
  std::vector<Frame> Frames() const;

  const absl::InlinedVector<std::pair<PyCodeObject*, int>, 32>& raw_frames()
      const {
    return frames_;
  }

  // Returns the traceback as a fake Python Traceback object, suitable for
  // using as an exception traceback.
  nanobind::object AsPythonTraceback() const;

  bool operator==(const Traceback& other) const {
    return frames_ == other.frames_;
  }
  bool operator!=(const Traceback& other) const {
    return frames_ != other.frames_;
  }

 private:
  // Each frame is a pair of a code object and a "lasti" instruction location
  // in bytes. The size of _Py_CODEUNIT has changed across different Python
  // versions; the lasti value here has already been multiplied by
  // sizeof(_Py_CODEUNIT) if needed and is suitable for passing to functions
  // like PyCode_Addr2Line().
  absl::InlinedVector<std::pair<PyCodeObject*, int>, 32> frames_;

  // Protected by GIL.
  static bool enabled_;
};

using nb_traceback = nb_class_ptr<Traceback>;

template <typename H>
H AbslHashValue(H h, const Traceback& traceback) {
  h = H::combine(std::move(h), traceback.raw_frames());
  return h;
}

void BuildTracebackSubmodule(nanobind::module_& m);

// TODO(b/318709106): This is a temporary solution to propagate a hint to
// backends that the current traceback does not change within the scope.
// This should be removed once context propagation from IFRT API is
// implemented.

// A scope that caches the current traceback cache id in thread_local storage.
// This is used as a hint for the backend that the current Traceback does not
// change within the scope.
class TracebackCacheScope {
 public:
  TracebackCacheScope();
  ~TracebackCacheScope();

 private:
  uint64_t id_;
  uint64_t previous_id_;
};

// Returns the current traceback cache id.
uint64_t CurrentTracebackCacheId();

}  // namespace xla

#endif  // XLA_PYTHON_TRACEBACK_H_
