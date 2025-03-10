/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_RUNNER_H_
#define XLA_SERVICE_HLO_RUNNER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

class BufferAssignmentProto;

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
  absl::StatusOr<ScopedShapedBuffer> TransferLiteralToDevice(
      const Literal& literal, int64_t param_no);
  absl::StatusOr<std::vector<ScopedShapedBuffer>> TransferLiteralsToDevice(
      absl::Span<const Literal* const> literals);
  absl::StatusOr<std::vector<ScopedShapedBuffer>> TransferLiteralsToDevice(
      absl::Span<const Literal> literals);
  absl::StatusOr<Literal> TransferLiteralFromDevice(const ShapedBuffer& buffer);

  // Executes the given module with given literals as input and returns the
  // result as a Literal.
  //
  // If run_hlo_passes is false, the module will be executed without Hlo
  // optimization.

  using HloRunnerInterface::Execute;

  absl::StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                                  absl::Span<const Literal* const> arguments,
                                  bool run_hlo_passes,
                                  ExecutionProfile* profile) override;

  using HloRunnerInterface::ExecuteWithBufferAssignment;

  absl::StatusOr<Literal> ExecuteWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* buffer_assignment_proto,
      absl::Span<const Literal* const> arguments, bool run_hlo_passes,
      ExecutionProfile* profile) override;

  using HloRunnerInterface::ExecuteWithExecutable;

  absl::StatusOr<Literal> ExecuteWithExecutable(
      OpaqueExecutable* executable, absl::Span<const Literal* const> arguments,
      ExecutionProfile* profile) override;

  // As Execute(), but accepts and returns device buffers instead of host
  // buffers.
  //
  // ExecuteWithMovedDeviceBuffers is more memory-safe, but it consumes the
  // arguments. Please consider using that.
  //
  // This may overwrite the values of the arguments if the the module has
  // aliasing.
  absl::StatusOr<ExecutionOutput> ExecuteWithDeviceBuffers(
      std::unique_ptr<HloModule> module,
      absl::Span<ScopedShapedBuffer const> arguments,
      bool run_hlo_passes = true, ExecutionProfile* profile = nullptr);

  absl::StatusOr<ExecutionOutput> ExecuteWithDeviceBuffers(
      OpaqueExecutable* executable,
      absl::Span<ScopedShapedBuffer const> arguments,
      ExecutionProfile* profile = nullptr);

  // As Execute(), but accepts and returns device buffers instead of host
  // buffers.
  //
  // This is a memory-safer version of ExecuteWithDeviceBuffers, but it consumes
  // the arguments.
  absl::StatusOr<ExecutionOutput> ExecuteWithMovedDeviceBuffers(
      std::unique_ptr<HloModule> module,
      std::vector<ScopedShapedBuffer> arguments, bool run_hlo_passes = true,
      ExecutionProfile* profile = nullptr);

  absl::StatusOr<ExecutionOutput>
  ExecuteWithMovedDeviceBuffersAndBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* buffer_assignment_proto,
      std::vector<ScopedShapedBuffer> arguments, bool run_hlo_passes = true,
      ExecutionProfile* profile = nullptr);

  absl::StatusOr<ExecutionOutput> ExecuteWithMovedDeviceBuffers(
      Executable* executable, std::vector<ScopedShapedBuffer> arguments,
      ExecutionProfile* profile = nullptr);

  // Creates an executable object given an HLO module. If run_hlo_passes is
  // true, the HLO passes will be run as part of compilation.
  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) override;

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
  CreateExecutableWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* /*buffer_assignment_proto*/,
      bool run_hlo_passes) override;

  // Creates a runner-internal executable object given a runner and
  // platform-specific serialized executable representation. The serialized
  // representation must have been produced by a compiler of the same platform
  // and version as this one.
  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> DeserializeExecutable(
      absl::Nonnull<const tsl::protobuf::Message*> serialized) const override;

  // Executes a given HLO module into a set of replicas, and returns a map
  // with the replica number as key, and the corresponding returned literal as
  // value.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options) override;

  // Same as above, but with specified device assignment.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override;

  // Same as above, but with a reusable Executable.  This may update the profile
  // information in *executable.
  //
  // Note that this call ignores ReplicatedExecutionOptions::run_hlo_passes,
  // since we've already compiled the Executable.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      OpaqueExecutable* executable, const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment, ExecutionProfile* profile = nullptr);

  // Same as above, but with different reusable Executables. This may update the
  // profile information in *executables.
  //
  // Note that this call ignores ReplicatedExecutionOptions::run_hlo_passes,
  // since we've already compiled the Executable.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::function<OpaqueExecutable*(int64_t)> executable_provider,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override;

  // If backend is not created in the constructor, creates and returns the
  // default backend. If creation fails, crashes the program.
  //
  // This creates the backend lazily so it's possible to instantiate an
  // HloRunner in a program without any backends linked in.
  Backend& backend();
  const Backend& backend() const;

  absl::string_view Name() const override;

  DeviceShapeRepresentationFn device_shape_representation_fn() const override {
    return device_shape_representation_fn_;
  }

  DeviceShapeSizeFn device_shape_size_fn() const override {
    return backend().compiler()->ShapeSizeBytesFunction();
  }

  int device_count() const override { return backend().device_count(); }

  bool HasProperty(HloRunnerPropertyTag::Type tag) const override;

  // Helpers to interact with OpaqueExecutable before all users are migrated.
  absl::StatusOr<Executable*> ExecutableFromWrapped(
      const OpaqueExecutable* wrapped) const;
  absl::StatusOr<std::unique_ptr<Executable>> ExecutableFromWrapped(
      std::unique_ptr<OpaqueExecutable> wrapped) const;
  std::unique_ptr<OpaqueExecutable> WrapExecutable(
      std::unique_ptr<Executable> executable) const;
  absl::StatusOr<absl::Nonnull<const HloModule*>> HloModuleFromWrapped(
      const OpaqueExecutable* wrapped) const override;
  // Returns the HloProto of the Executable wrapped by the given
  // OpaqueExecutable. This is a temporary API to help move to OpaqueExecutable.
  // We need to come up with a better way to obtain this information and
  // evaluate whether we need to do this at all. A drop-in migration to
  // HloRunnerPjRt (via HloRunnerInterface) won't be possible because this
  // information is not available from a PjRt(Loaded)Executable.
  //
  // TODO: b/393183864 - Remove this API.
  absl::StatusOr<absl::Nonnull<const HloProto*>> HloProtoFromWrapped(
      const OpaqueExecutable* wrapped) const;

 private:
  absl::StatusOr<ExecutionOutput> ExecuteWithExecutionInputs(
      Executable* executable, std::vector<ExecutionInput> arguments,
      ExecutionProfile* profile);

  // Creates a ServiceExecutableRunOptions object to configure a run on device,
  // using the provided stream object. If device_assignment is not nullptr, it
  // will be used to configure the replication parameters. Replicated executions
  // should pass the device_assignment parameter.
  ServiceExecutableRunOptions GetServiceRunOptionsForDevice(
      int64_t device, se::Stream* stream, DeviceAssignment* device_assignment,
      RunId run_id, int local_device_count);

  // Common implementation code for ExecuteReplicated() above.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicatedImpl(
      std::function<absl::StatusOr<std::vector<ScopedShapedBuffer>>(
          const std::vector<ServiceExecutableRunOptions>&,
          const std::vector<absl::Span<const ShapedBuffer* const>>&)>
          execution_helper,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment);

  // Gets or creates the DeviceMemoryAllocator.
  se::DeviceMemoryAllocator* GetAllocator();

  std::unique_ptr<Backend> backend_;

  std::unique_ptr<se::DeviceMemoryAllocator> allocator_;

  DeviceShapeRepresentationFn device_shape_representation_fn_;

  const ComputationLayout* entry_computation_layout_ = nullptr;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_RUNNER_H_
