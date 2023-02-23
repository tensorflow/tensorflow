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

#include "tensorflow/compiler/jit/variable_info.h"

#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {

VariableInfo::VariableInfo(
    int index, absl::string_view name, Var* var,
    const std::optional<ManagedStackTrace>& definition_stack_trace)
    : index_(index),
      name_(name),
      var_(var),
      definition_stack_trace_(definition_stack_trace) {}

VariableInfo::VariableInfo(VariableInfo&& other)
    : index_(other.index_),
      var_(other.var_),
      definition_stack_trace_(other.definition_stack_trace_),
      lock_held_(other.lock_held_) {
  other.index_ = -1;
  other.var_ = nullptr;
}

VariableInfo& VariableInfo::operator=(VariableInfo&& other) {
  index_ = other.index_;
  var_ = other.var_;
  lock_held_ = other.lock_held_;
  definition_stack_trace_ = other.definition_stack_trace_;

  other.index_ = -1;
  other.var_ = nullptr;

  return *this;
}

VariableInfo::~VariableInfo() {
  // Release the variable's lock if we hold it. Ensures that the lock is
  // released even on error.  It does not matter in what order we release the
  // locks.
  if (var()) {
    if (lock_held()) {
      var()->mu()->unlock();
    }
    if (shared_lock_held()) {
      var()->mu()->unlock_shared();
    }

    // Unref the variable so it can be released by ResourceManager.
    var()->Unref();
  }
}

}  // namespace tensorflow
