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

#ifndef XLA_SERVICE_HLO_RUNNER_INTERFACE_H_
#define XLA_SERVICE_HLO_RUNNER_INTERFACE_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class BufferAssignmentProto;

// A base class for running an HloModule. This executes the given HloModule on a
// certain backend directly without using the client interface. HloModule can be
// explicitly built, or loaded from a serialization file (e.g., hlo proto
// file), or parsed from a hlo textual IR string.
class HloRunnerInterface {
 public:
  // The options used to configure an ExecuteReplicated() call.
  struct ReplicatedExecuteOptions {
    // The number of devices the HLO module should be replicated onto.
    int64_t num_replicas = 1;

    // The arguments to be fed to each replica. Since this is used for a
    // replicated execution, all the arguments are the same for all replicas.
    std::vector<const Literal*> arguments;

    // If the HLO module being run has an infeed instruction, this will be the
    // data which will be fed to it, for as many as infeed_steps steps.
    std::vector<const Literal*> infeed_values;

    // The number of times the infeed literal should be fed to the HLO module.
    // For a clean exit, this should match the iterations-per-loop parameter
    // used when generating the HLO module proto (that is usually the main
    // while boundary counter). A value higher then iterations-per-loop would
    // lead to infeed threads feeding to a gone computation, while a lower
    // value would trigger a stuck ExecuteReplicated() call (the computation
    // will be trying to infeed data which will never come).
    int64_t infeed_steps = -1;

    // The shape of the outfeed operation. If empty, the HLO module does not
    // generate any outfeed.
    Shape outfeed_shape;

    // A pointer to a vector where the outfeed values will be stored. If
    // nullptr, the values will be read and discarded.
    std::vector<Literal>* outfeed_values = nullptr;

    // Whether the HLO passes should be run on the input module. Usually
    // saved modules are coming from after the HLO pass pipeline, so triggering
    // another run will likely cause errors.
    bool run_hlo_passes = false;

    // If true, executes on multiple threads using se::Stream::ExecuteOnStream.
    // Otherwise, executes using xla::Executable::ExecuteOnStreams.
    bool use_threads = false;
  };

  HloRunnerInterface() = default;

  virtual ~HloRunnerInterface() = default;

  // Converts an HloModule from the given hlo textual IR string (in
  // HloModule::ToString format).
  static absl::StatusOr<std::unique_ptr<HloModule>> CreateModuleFromString(
      const absl::string_view hlo_string, const DebugOptions& debug_options);

  // Reads the proto file in xla.HloProto format, creates and returns the
  // HloModule.
  static absl::StatusOr<std::unique_ptr<HloModule>>
  ReadModuleFromBinaryProtoFile(const std::string& filename,
                                const DebugOptions& debug_options);
  static absl::StatusOr<std::unique_ptr<HloModule>> ReadModuleFromTextProtoFile(
      const std::string& filename, const DebugOptions& debug_options);

  // Reads the proto file in xla.HloModule format, creates and returns the
  // HloModule.
  static absl::StatusOr<std::unique_ptr<HloModule>>
  ReadModuleFromModuleBinaryProtofile(const std::string& filename,
                                      const DebugOptions& debug_options);

  // Reads the hlo text dump file in HloModule::ToString format, creates and
  // returns the HloModule.
  static absl::StatusOr<std::unique_ptr<HloModule>> ReadModuleFromHloTextFile(
      const std::string& filename, const DebugOptions& debug_options);

  // Creates an executable object given an HLO module. If run_hlo_passes is
  // true, the HLO passes will be run as part of compilation.
  virtual absl::StatusOr<std::unique_ptr<Executable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) = 0;

  // Same as above, except it takes buffer assignment as input.
  // Note: The default implementation of the API here does not utilize the given
  // buffer assignment. A derived runner interface is expected to override the
  // following method to achieve this functionality.
  virtual absl::StatusOr<std::unique_ptr<Executable>>
  CreateExecutableWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* /*buffer_assignment_proto*/,
      bool run_hlo_passes) {
    LOG(WARNING) << "Ignoring the buffer assignment proto provided.";
    return CreateExecutable(std::move(module), run_hlo_passes);
  }

  // Executes the given module with given literals as input and returns the
  // result as a Literal.
  //
  // If run_hlo_passes is false, the module will be executed without Hlo
  // optimization
  absl::StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                                  absl::Span<const Literal* const> arguments,
                                  bool run_hlo_passes = true) {
    return Execute(std::move(module), arguments, run_hlo_passes, nullptr);
  }

  absl::StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                                  absl::Span<const Literal> arguments,
                                  bool run_hlo_passes = true,
                                  ExecutionProfile* profile = nullptr);

  virtual absl::StatusOr<Literal> Execute(
      std::unique_ptr<HloModule> module,
      absl::Span<const Literal* const> arguments, bool run_hlo_passes,
      ExecutionProfile* profile) = 0;

  // Same as above 3 methods, but with buffer assignment specified.
  absl::StatusOr<Literal> ExecuteWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* buffer_assignment_proto,
      absl::Span<const Literal* const> arguments, bool run_hlo_passes = true) {
    return ExecuteWithBufferAssignment(std::move(module),
                                       buffer_assignment_proto, arguments,
                                       run_hlo_passes, nullptr);
  }

  absl::StatusOr<Literal> ExecuteWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* buffer_assignment_proto,
      absl::Span<const Literal> arguments, bool run_hlo_passes = true,
      ExecutionProfile* profile = nullptr);

  // Note: The default implementation of the API here does not utilize the given
  // buffer assignment. A derived runner interface is expected to override the
  // following method to achieve this functionality.
  virtual absl::StatusOr<Literal> ExecuteWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* /*buffer_assignment_proto*/,
      absl::Span<const Literal* const> arguments, bool run_hlo_passes,
      ExecutionProfile* profile) {
    LOG(WARNING) << "Ignoring the buffer assignment proto provided.";
    return Execute(std::move(module), arguments, run_hlo_passes, profile);
  }

  // Same as 3 Execute methods above, but with Executable as input.
  absl::StatusOr<Literal> ExecuteWithExecutable(
      Executable* executable, absl::Span<const Literal> arguments,
      ExecutionProfile* profile = nullptr);

  absl::StatusOr<Literal> ExecuteWithExecutable(
      Executable* executable, absl::Span<const Literal* const> arguments) {
    return ExecuteWithExecutable(executable, arguments, nullptr);
  }

  virtual absl::StatusOr<Literal> ExecuteWithExecutable(
      Executable* executable, absl::Span<const Literal* const> arguments,
      ExecutionProfile* profile) = 0;

  // Executes a given HLO module into a set of replicas, and returns a map
  // with the replica number as key, and the corresponding returned literal as
  // value.
  // TODO(b/172931928): change to non-virtual function.
  virtual absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options) = 0;

  // Same as above, but with specified device assignment.
  virtual absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) = 0;

  virtual absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::function<Executable*(int64_t)> executable_provider,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) = 0;

  // Returns the name of this runner.
  virtual absl::string_view Name() const = 0;

  typedef std::function<Shape(const Shape&)> DeviceShapeRepresentationFn;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_RUNNER_INTERFACE_H_
