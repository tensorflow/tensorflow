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
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

// Provides configuration for implementations that support compile and execute
// spanning multiple slices. A slice is a set of devices connected by dedicated
// high speed interconnect. Connectivity between slices is typically over data
// center networks. Concrete implementations of MultiSliceConfig contain
// environment specific information to enable communication between devices on
// different slices. Passed as options during compile and execute.
// Implementations that do not support this are allowed to pass nullptr.
class MultiSliceConfig {
 public:
  virtual ~MultiSliceConfig();

  // Returns the total number of slices.
  virtual int32_t NumSlices() const = 0;

  // Returns the SliceID at this host - an integer in [0, NumSlices)
  virtual int32_t SliceId() const = 0;

  // Returns the number of devices on each slice indexed by SliceId.
  virtual absl::flat_hash_map<int32_t, int32_t> NumDevicesPerSlice() const = 0;
};

struct CompileOptions {
  // The layouts of the arguments that the computation should expect.
  std::optional<std::vector<Shape>> argument_layouts;

  // If true, the supplied computation expects its arguments to be wrapped in a
  // tuple and passed as a single parameter.
  bool parameter_is_tupled_arguments = false;

  // XLA's compilation time options.
  ExecutableBuildOptions executable_build_options;

  // If true, the executable can be run on any device. May only be true if
  // !executable_build_options.has_device_assignment(), so only applies to
  // single-device executables. Beware: on GPUs, sometimes an executable
  // compiled for one device doesn't run on another.
  bool compile_portable_executable = false;

  // XLA compilation profile version.
  int64_t profile_version = 0;

  // Set multi_slice_config to trigger compilation for DCN connected multi
  // slice operation.
  const MultiSliceConfig* multi_slice_config = nullptr;

  // Serialize the CompileOptions into a CompileOptionsProto.
  StatusOr<CompileOptionsProto> ToProto() const;
};

StatusOr<CompileOptions> CompileOptionsFromProto(
    const CompileOptionsProto& input);

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
