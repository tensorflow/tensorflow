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

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/literal.h"
#include "xla/service/cpu/executable.pb.h"
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
  // Creates an XlaAotFunction object from a CompilationResultProto.
  // It infers the argument and result names from the HLO module in the
  // produced executable.
  static absl::StatusOr<std::unique_ptr<XlaAotFunction>> Create(
      const CompilationResultProto& compilation_result);

  // Same as above but allows users to specify custom argument and result names.
  static absl::StatusOr<std::unique_ptr<XlaAotFunction>> Create(
      const CompilationResultProto& compilation_result,
      std::vector<std::string> arg_names,
      std::vector<std::string> result_names);

  // Not thread safe.
  absl::Status Execute();

  void set_arg_data(absl::string_view arg_name, const void* data) {
    auto it = name_to_argument_index_.find(arg_name);
    CHECK(it != name_to_argument_index_.end())
        << "Argument " << arg_name << " not found.";
    set_arg_data(it->second, data);
  }

  void set_arg_data(size_t index, const void* data) {
    DCHECK_LT(index, arguments_.size())
        << "Index " << index
        << " is out of bounds. Arguments size: " << arguments_.size();

    arguments_[index] =
        NanoRtExecutable::Argument(data, argument_sizes_[index]);
  }

  int64_t arg_size(size_t index) const {
    DCHECK_LT(index, argument_sizes_.size());
    return argument_sizes_[index];
  }

  int64_t arg_size(absl::string_view arg_name) const {
    auto it = name_to_argument_index_.find(arg_name);
    CHECK(it != name_to_argument_index_.end())
        << "Argument " << arg_name << " not found.";
    return arg_size(it->second);
  }

  int64_t result_size(size_t index) const {
    DCHECK_LT(index, results_.size());
    return results_[index].data().size();
  }
  int64_t result_size(absl::string_view result_name) const {
    auto it = name_to_result_index_.find(result_name);
    CHECK(it != name_to_result_index_.end())
        << "Result " << result_name << " not found.";
    return result_size(it->second);
  }

  void* result_data(size_t index) const {
    DCHECK_LT(index, results_.size());
    return results_[index].data().data();
  }

  void* result_data(absl::string_view result_name) const {
    auto it = name_to_result_index_.find(result_name);
    CHECK(it != name_to_result_index_.end())
        << "Result " << result_name << " not found.";
    return result_data(it->second);
  }

  const NanoRtExecutable* executable() { return executable_.get(); }

 protected:
  explicit XlaAotFunction(std::unique_ptr<NanoRtExecutable> executable,
                          std::vector<Literal> results_literals,
                          Literal temp_literal,
                          std::vector<std::string> argument_names,
                          std::vector<std::string> result_names);

 private:
  std::unique_ptr<NanoRtExecutable> executable_;

  // Used for the blocking invocation to Execute.
  // TODO(basioli): Consider allocating Literals for users.
  absl::flat_hash_map<std::string, int64_t> name_to_argument_index_;
  std::vector<NanoRtExecutable::Argument> arguments_;
  // Memorize argument sizes
  std::vector<int64_t> argument_sizes_;

  absl::flat_hash_map<std::string, int64_t> name_to_result_index_;
  std::vector<NanoRtExecutable::Result> results_;
  std::vector<Literal> results_literals_;

  NanoRtExecutable::PreallocatedTemp temp_;
  Literal temp_literal_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_LITE_AOT_XLA_AOT_FUNCTION_H_
