/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_LITE_AOT_XLA_AOT_FUNCTION_H_
#define XLA_BACKENDS_CPU_LITE_AOT_XLA_AOT_FUNCTION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/literal.h"
#include "xla/service/executable.h"

namespace xla::cpu {

// A wrapper around NanoRtExecutable that allows users to execute compiled
// XLA:CPU functions in a blocking manner.
// The user is responsible for allocating the arguments to the function.

// The expected workflow is:
// 1. Create an XlaAotFunction object by calling Create() with the
// CompilationResultProto.
// 2. Allocate memory for the arguments and results.
// 3. Set the argument data by calling set_arg_data() for each argument.
// 4. Execute the function by calling Execute().
// 5. Get the result data by calling result_data() for each result.

class XlaAotFunction {
 public:
  static absl::StatusOr<std::unique_ptr<XlaAotFunction>> Create(
      const CompilationResultProto& compilation_result);

  // Not thread safe.
  absl::Status Execute();

  void set_arg_data(size_t index, const void* data) {
    DCHECK_LT(index, arguments_.size())
        << "Index " << index
        << " is out of bounds. Arguments size: " << arguments_.size();

    arguments_[index] =
        NanoRtExecutable::Argument(data, argument_sizes_[index]);
  }

  int64_t arg_size(size_t index) const { return argument_sizes_[index]; }
  int64_t result_size(size_t index) const {
    return results_[index].data().size();
  }

  void* result_data(size_t index) { return results_[index].data().data(); }
  const void* result_data(size_t index) const {
    return results_[index].data().data();
  }

 protected:
  explicit XlaAotFunction(std::unique_ptr<NanoRtExecutable> executable,
                          std::vector<Literal> results_literals,
                          Literal temp_literal);

 private:
  std::unique_ptr<NanoRtExecutable> executable_;

  // Used for the blocking invocation to Execute.
  // TODO(basioli): Consider allocating Literals for users.
  std::vector<NanoRtExecutable::Argument> arguments_;
  // Memorize argument sizes
  std::vector<int64_t> argument_sizes_;

  std::vector<NanoRtExecutable::Result> results_;
  std::vector<Literal> results_literals_;

  NanoRtExecutable::PreallocatedTemp temp_;
  Literal temp_literal_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_LITE_AOT_XLA_AOT_FUNCTION_H_
