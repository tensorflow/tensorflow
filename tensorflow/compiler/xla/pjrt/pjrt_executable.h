/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

// Static device memory usage for a compiled program.
// The on-device memory needed to run an executable is at least
//   generated_code_size_in_bytes
//   + argument_size_in_bytes + output_size_in_bytes - alias_size_in_bytes
//   + temp_size_in_bytes.
struct CompiledMemoryStats {
  int64_t generated_code_size_in_bytes = 0;
  int64_t argument_size_in_bytes = 0;
  int64_t output_size_in_bytes = 0;
  // How much argument is reused for output.
  int64_t alias_size_in_bytes = 0;
  int64_t temp_size_in_bytes = 0;

  std::string DebugString() const;
};

class PjRtExecutable {
 public:
  virtual ~PjRtExecutable() = default;

  virtual int num_replicas() const = 0;

  virtual int num_partitions() const = 0;

  virtual int64_t SizeOfGeneratedCodeInBytes() const = 0;

  // Unique name for this executable, e.g., HloModule name.
  virtual absl::string_view name() const = 0;

  // Return an HloModule (optimized) per partition.
  virtual StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const = 0;

  // Return memory stats that allow callers to estimate device memory usage
  // when running this executable.
  virtual StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const {
    return Unimplemented("Retrieving CompiledMemoryStats is not supported.");
  }

  // Serialize this executable into a string and return the value.
  virtual StatusOr<std::string> SerializeExecutable() const {
    return Unimplemented("Serializing executable is not supported.");
  }

  // Return a fingerprint of this executable.
  virtual StatusOr<std::string> FingerprintExecutable() const {
    return Unimplemented("Fingerprinting executable is not supported.");
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_EXECUTABLE_H_
