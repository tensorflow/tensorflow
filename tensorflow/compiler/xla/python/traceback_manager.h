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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TRACEBACK_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TRACEBACK_MANAGER_H_

#include "absl/container/node_hash_map.h"
#include "absl/types/optional.h"
#include "pybind11/pybind11.h"

namespace xla {

struct TracebackImpl;

template <typename H>
H AbslHashValue(H h, const TracebackImpl& tb);

// Traceback manager class that deduplicates traceback objects to save memory.
// It probably does not save time to deduplicate tracebacks, but we expect to
// see many copies of the same tracebacks and hence we deduplicate in an attempt
// to save memory.
class TracebackManager {
 public:
  static TracebackManager* Get();

  ~TracebackManager();

  TracebackManager(const TracebackManager&) = delete;
  TracebackManager(TracebackManager&&) = delete;
  TracebackManager& operator=(const TracebackManager&) = delete;
  TracebackManager& operator=(TracebackManager&&) = delete;

  struct Frame {
    pybind11::str file_name;
    pybind11::str function_name;
    unsigned int line_num;
    int function_start_line;
  };

  // RAII class that holds a reference to a traceback.
  class Traceback {
   public:
    Traceback() = default;
    ~Traceback();

    Traceback(const Traceback&);
    Traceback(Traceback&&);
    Traceback& operator=(const Traceback&);
    Traceback& operator=(Traceback&&);

    // Requires the GIL be held.
    std::string ToString() const;

    // Returns the stack frame objects, in order from innermost to outermost.
    const std::vector<Frame>* Frames() const;

   private:
    friend class TracebackManager;

    Traceback(TracebackManager* manager,
              std::pair<TracebackImpl const, int>* impl);

    // nullptr for a default-constructed Traceback, non-null otherwise.
    TracebackManager* manager_ = nullptr;
    // Points to an entry in tracebacks_. Not owned.
    std::pair<TracebackImpl const, int>* impl_ = nullptr;
  };

  // Returns a Traceback for the current thread. Returns nullopt if tracebacks
  // aren't enabled,
  absl::optional<Traceback> GetTraceback();

  // Enables or disables traceback collection.
  void SetEnabled(bool enabled);
  bool enabled() const { return enabled_; }

 private:
  TracebackManager();

  bool enabled_ = false;

  // Deduplicated tracebacks. Map from traceback to reference count.
  // The map and its contents are protected by the GIL, which is why we do not
  // need an atomic integer for the reference count.
  absl::node_hash_map<TracebackImpl, int> tracebacks_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TRACEBACK_MANAGER_H_
