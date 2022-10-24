/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_RUNNER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_RUNNER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_runner_interface.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// A base class for running an HloModule. This executes the given HloModule on a
// certain backend directly without using the client interface. HloModule can be
// explicitly built, or loaded from a serialization file (e.g., hlo proto
// file), or parsed from a hlo textual IR string.
class HloRunner : public HloRunnerInterface {
 public:
  // intra_op_parallelism_threads: For the CPU backend only. It is the thread
  // pool size for parallel execution of an individual operator. The default
  // value of -1 will result in initializing the thread pool with the number of
  // threads equal to the number of
  // cores in the system.
  explicit HloRunner(se::Platform* platform,
                     int intra_op_parallelism_threads = -1);

  ~HloRunner() override;

  // Transfers data between the host and device.
  StatusOr<ScopedShapedBuffer> TransferLiteralToDevice(const Literal& literal);
  StatusOr<std::vector<ScopedShapedBuffer>> TransferLiteralsToDevice(
      absl::Span<const Literal* const> literals);
  StatusOr<std::vector<ScopedShapedBuffer>> TransferLiteralsToDevice(
      absl::Span<const Literal> literals);
  StatusOr<Literal> TransferLiteralFromDevice(const ShapedBuffer& buffer);

  // Executes the given module with given literals as input and returns the
  // result as a Literal.
  //
  // If run_hlo_passes is false, the module will be executed without Hlo
  // optimization.

  using HloRunnerInterface::Execute;

  StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                            absl::Span<const Literal* const> arguments,
                            bool run_hlo_passes,
                            ExecutionProfile* profile) override;

  using HloRunnerInterface::ExecuteWithExecutable;

  StatusOr<Literal> ExecuteWithExecutable(
      Executable* executable, absl::Span<const Literal* const> arguments,
      ExecutionProfile* profile) override;

  // As Execute(), but accepts and returns device buffers instead of host
  // buffers.
  StatusOr<ExecutionOutput> ExecuteWithDeviceBuffers(
      std::unique_ptr<HloModule> module,
      absl::Span<ScopedShapedBuffer const> arguments,
      bool run_hlo_passes = true, ExecutionProfile* profile = nullptr);

  StatusOr<ExecutionOutput> ExecuteWithDeviceBuffers(
      Executable* executable, absl::Span<ScopedShapedBuffer const> arguments,
      ExecutionProfile* profile = nullptr);

  // Creates an executable object given an HLO module. If run_hlo_passes is
  // true, the HLO passes will be run as part of compilation.
  StatusOr<std::unique_ptr<Executable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) override;

  // Executes a given HLO module into a set of replicas, and returns a map
  // with the replica number as key, and the corresponding returned literal as
  // value.
  StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options) override;

  // Same as above, but with specified device assignment.
  StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override;

  // Same as above, but with a reusable Executable.  This may update the profile
  // information in *executable.
  //
  // Note that this call ignores ReplicatedExecutionOptions::run_hlo_passes,
  // since we've already compiled the Executable.
  StatusOr<std::vector<Literal>> ExecuteReplicated(
      Executable* executable, const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment, ExecutionProfile* profile = nullptr);

  // Same as above, but with different reusable Executables. This may update the
  // profile information in *executables.
  //
  // Note that this call ignores ReplicatedExecutionOptions::run_hlo_passes,
  // since we've already compiled the Executable.
  StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::function<Executable*(int64_t)> executable_provider,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment = nullptr);

  // If backend is not created in the constructor, creates and returns the
  // default backend. If creation fails, crashes the program.
  //
  // This creates the backend lazily so it's possible to instantiate an
  // HloRunner in a program without any backends linked in.
  Backend& backend();
  const Backend& backend() const;

  absl::string_view Name() const override;

  DeviceShapeRepresentationFn device_shape_representation_fn() {
    return device_shape_representation_fn_;
  }

 private:
  // Creates a ServiceExecutableRunOptions object to configure a run on device,
  // using the provided stream object. If device_assignment is not nullptr, it
  // will be used to configure the replication parameters. Replicated executions
  // should pass the device_assignment parameter.
  ServiceExecutableRunOptions GetServiceRunOptionsForDevice(
      int64_t device, se::Stream* stream, DeviceAssignment* device_assignment,
      RunId run_id);

  // Common implementation code for ExecuteReplicated() above.
  StatusOr<std::vector<Literal>> ExecuteReplicatedImpl(
      std::function<StatusOr<std::vector<ScopedShapedBuffer>>(
          const std::vector<ServiceExecutableRunOptions>&,
          const std::vector<absl::Span<const ShapedBuffer* const>>&)>
          execution_helper,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment);

  std::unique_ptr<Backend> backend_;

  DeviceShapeRepresentationFn device_shape_representation_fn_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_RUNNER_H_
