/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_EXECUTABLE_REGISTRY_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_EXECUTABLE_REGISTRY_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"

namespace tensorflow {
namespace ifrt_serving {

// Maintains a process-wide map from program ids to executables. Used by the
// `IfrtCall` TensorFlow op to look up executables and invoke them.
//
// Invoking a TPU program inside a `IfrtCall` TF op requires being
// able to retrieve an executable for the given program. Since there's no easy
// way to pass non-serializable attributes to TF ops, we encode a program id
// (that is unique within a process) as an attribute of a `IfrtCall` op and
// use this registry class to let the `IfrtCall` op look up an executable
// during TF op execution.
class ServingExecutableRegistry {
 public:
  // RAII handle for registered executables.
  class Handle {
   public:
    Handle();  // Constructs an empty handle.

    // Move only.
    Handle(Handle&& other);
    Handle& operator=(Handle&& other);
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    ~Handle();

    // Returns the program id that the handle represents, or `std::nullopt` if
    // the handle is empty.
    std::optional<int64_t> program_id() const { return program_id_; }

    // Unregisters the owned executable, if any, early (before the destructor).
    // Calling this method multiple times is a no-op.
    void Release();

   private:
    friend class ServingExecutableRegistry;

    // Can only be constructed by `ServingExecutableRegistry::Register()`.
    explicit Handle(int64_t program_id);

    // Program id. `std::nullopt` if the handle is already released.
    std::optional<int64_t> program_id_;
  };

  // Registers an executable under the given program id. Returns an RAII handle
  // that unregisters the program at its destruction.
  static absl::StatusOr<Handle> Register(
      int64_t program_id, std::unique_ptr<IfrtServingExecutable> executable);

  // Looks up an executable registered under the given program id, or returns
  // nullptr if there's no such program.
  static IfrtServingExecutable* Lookup(int64_t program_id);

 private:
  friend class Handle;

  static absl::Mutex mu_;

  // Mapping from program ids to executables.
  static absl::flat_hash_map<int64_t,
                             std::unique_ptr<IfrtServingExecutable>>* const
      executables_ ABSL_GUARDED_BY(&mu_);
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_EXECUTABLE_REGISTRY_H_
