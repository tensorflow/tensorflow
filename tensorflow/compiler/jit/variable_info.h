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

#ifndef TENSORFLOW_COMPILER_JIT_VARIABLE_INFO_H_
#define TENSORFLOW_COMPILER_JIT_VARIABLE_INFO_H_

#include <map>
#include <optional>
#include <set>
#include <string>

#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Information about the state of a variable passed as input to the _XlaCompile
// and _XlaRun operators.  Unlocks the resource variable and decrements its
// refcount on destruction.
class VariableInfo {
 public:
  explicit VariableInfo(int index, absl::string_view name, Var* var,
                        const std::optional<ManagedStackTrace>&
                            definition_stack_trace = std::nullopt);
  VariableInfo(VariableInfo&& other);

  VariableInfo& operator=(VariableInfo&& other);

  VariableInfo(const VariableInfo&) = delete;
  VariableInfo& operator=(const VariableInfo&) = delete;

  // The index of the DT_RESOURCE input to the _XlaCompile/_XlaRun operator.
  // Note that the indices can be different between _XlaCompile and _XlaRun.
  int index() const { return index_; }

  // A pointer to the resource variable.  May be null if this VariableInfo is
  // "empty", i.e. it does not track a resource variable.
  Var* var() const { return var_; }

  // Returns the variable name.
  absl::string_view name() const { return name_; }

  // Returns true if the resource variable lock was successfully acquired by
  // this thread.
  bool lock_held() const { return lock_held_; }
  void set_lock_held() { lock_held_ = true; }

  // Returns true if the resource variable reader lock was successfully acquired
  // by this thread.
  bool shared_lock_held() const { return shared_lock_held_; }
  void set_shared_lock_held() { shared_lock_held_ = true; }

  bool read_only() const { return read_only_; }
  void set_read_only() { read_only_ = true; }

  const std::optional<ManagedStackTrace>& definition_stack_trace() const {
    return definition_stack_trace_;
  }

  ~VariableInfo();

 private:
  int index_;
  std::string name_;
  Var* var_;
  std::optional<ManagedStackTrace> definition_stack_trace_;

  // We can't use a optional<mutex_lock> here because it confuses the compiler's
  // thread safety analysis. Instead we use a boolean flag and release the lock
  // in the VariableInfo destructor.
  bool lock_held_ = false;
  bool shared_lock_held_ = false;

  // Whether this variable is going to be mutated. Left false if the caller
  // doesn't provide this information.
  bool read_only_ = false;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_VARIABLE_INFO_H_
