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

#ifndef XLA_SERVICE_SERVICE_H_
#define XLA_SERVICE_SERVICE_H_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/client/xla_computation.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/allocation_tracker.h"
#include "xla/service/backend.h"
#include "xla/service/channel_tracker.h"
#include "xla/service/compilation_cache.h"
#include "xla/service/executable.h"
#include "xla/service/execution_tracker.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {

class Service;

// Options to configure the service when it is created.
class ServiceOptions {
 public:
  // Set the platform backing the service, or nullptr for the default platform.
  ServiceOptions& set_platform(se::Platform* platform);
  se::Platform* platform() const;

  // Set the default number of replicas to use when compiling replicated
  // programs.
  ServiceOptions& set_number_of_replicas(int number_of_replicas);
  int number_of_replicas() const;

  // Sets the thread pool size for parallel execution of an individual operator.
  ServiceOptions& set_intra_op_parallelism_threads(int num_threads);
  int intra_op_parallelism_threads() const;

  // Sets the allowed_devices set for selectively constructing stream executors
  // on the platform.
  ServiceOptions& set_allowed_devices(
      const std::optional<std::set<int>>& allowed_devices);
  const std::optional<std::set<int>>& allowed_devices() const;

 private:
  se::Platform* platform_ = nullptr;
  int number_of_replicas_ = 1;
  int intra_op_parallelism_threads_ = -1;
  std::optional<std::set<int>> allowed_devices_;
};

// A GlobalData object represents a globally-accessible allocation of
// data in the associated XLA service.
class GlobalData {
 public:
  // Gives ownership of the global data handle to this object.
  GlobalData(Service* parent, GlobalDataHandle handle);

  // Unregisters the wrapped handle, which causes the service to
  // deallocate the associated data.
  ~GlobalData();

  const GlobalDataHandle& handle() const { return handle_; }

  // Releases a set of GlobalData handles. A single RPC will be issued
  // per unique Service of the given GlobalData objects.
  static void Release(std::vector<std::unique_ptr<GlobalData>> instances);

 private:
  // Detaches the global data handle from the object, such that the destructor
  // will not try to release it.
  GlobalDataHandle Release() {
    parent_ = nullptr;
    return handle_;
  }

  GlobalDataHandle handle_;  // Handle being wrapped.
  Service* parent_;          // Service used to unregister handle_.

  GlobalData(const GlobalData&) = delete;
  GlobalData& operator=(const GlobalData&) = delete;
};

// A struct to represent a computation instance to be executed.
// * If execution_options.device_handles is not empty, the computation is
//   executed on the devices associated with the handles by partitioning the
//   computation based on the attached sharding attributes. Otherwise, a
//   device is chosen by the service.
struct XlaComputationInstance {
  const XlaComputation& computation;
  std::vector<GlobalData*> arguments;
  ExecutionOptions execution_options;
  ExecutionProfile* execution_profile;

  XlaComputationInstance(const XlaComputation& computation,
                         std::vector<GlobalData*> arguments,
                         ExecutionOptions execution_options,
                         ExecutionProfile* execution_profile)
      : computation(computation),
        arguments(std::move(arguments)),
        execution_options(execution_options),
        execution_profile(execution_profile) {}
};

// The XLA service object, which is the same across all platforms. It maintains
// the service state of computations and allocations, and delegates
// target-specific requests to the target-specific infrastructure
// (target-specific compiler, StreamExecutor).
class Service {
 public:
  // Unregisters a previously-allocated global handle.
  //
  // If the handle given is not currently allocated, a NOT_FOUND status is
  // returned.
  virtual absl::Status Unregister(const GlobalDataHandle& data);

  // Deconstructs a tuple. Returns a newly created GlobalDataHandle for each
  // element in the tuple.
  virtual absl::StatusOr<std::vector<std::unique_ptr<GlobalData>>>
  DeconstructTuple(const GlobalData& data);

  // Compiles a computation into an executable. The request contains the whole
  // computation graph. Returns the handle to the executable.
  virtual absl::StatusOr<ExecutionHandle> Compile(
      const XlaComputation& computation,
      absl::Span<const Shape> argument_shapes,
      const ExecutionOptions& execution_options);

  // Executes an executable with the provided global data passes as immutable
  // arguments. The request contains the handle to the executable. Returns
  // global data output and execution timing.
  virtual absl::StatusOr<std::unique_ptr<GlobalData>> Execute(
      const ExecutionHandle& handle, absl::Span<GlobalData* const> arguments,
      ExecutionProfile* execution_profile);

  // Executes one or more computations in parallel with the provided global data
  // passed as immutable arguments. Returns global data output for each
  // computation.
  absl::StatusOr<std::vector<std::unique_ptr<GlobalData>>> ExecuteGraphParallel(
      absl::Span<const XlaComputationInstance> computations);

  // Requests one or more device handles from the target.
  //
  // When N device handles are requested and the number of replicas is R, at
  // least N * R devices must be available. The devices are assigned based on
  // the device ordinals such that the first R available devices are assigned to
  // the first set of replicas, and the next R devices to the second set of
  // replicas, etc. Each returned device handle represents the device with the
  // replica id 0.
  virtual absl::StatusOr<std::vector<DeviceHandle>> GetDeviceHandles(
      int64_t device_count);

  // Requests that global data be transferred to the client in literal form.
  virtual absl::StatusOr<Literal> TransferToClient(
      const GlobalData& data, const Shape* shape_with_layout);

  // Transfers data from a literal provided by the client, into device memory.
  virtual absl::StatusOr<std::unique_ptr<GlobalData>> TransferToServer(
      const LiteralSlice& literal_slice, const DeviceHandle* device_handle);

  // Transfers data from a literal provided by the client, into the Infeed
  // buffer of the device.
  virtual absl::Status TransferToInfeed(const LiteralSlice& literal,
                                        int64_t replica_id,
                                        const DeviceHandle* device_handle);

  // Transfers data from the Outfeed othe device to the literal provided by the
  // client.
  virtual absl::StatusOr<Literal> TransferFromOutfeed(
      const Shape* shape_with_layout, int64_t replica_id,
      const DeviceHandle* device_handle);

  // Resets devices, clearing all existing state on all the devices associated
  // with this service (including memory allocated on the devices).
  //
  // ResetDevice may only be called where no previous Execution state on the
  // device is used by the next Execution.
  //
  // ResetDevice should be called before an Execution that expect the device to
  // be in the reset state. For example, if the prior Execution modifies device
  // state (e.g., architectural state) that the next Execution depends on.
  virtual absl::Status ResetDevice();

  virtual absl::StatusOr<Literal> ComputeConstantGraph(
      const XlaComputation& computation, const Layout* output_layout);

  // Returns the shape (with layout) of an array associated with a given data
  // handle.
  virtual absl::StatusOr<Shape> GetShape(const GlobalData& data);

  // Creates a unique channel handle that can be used for Send/Recv
  // instructions.
  virtual absl::StatusOr<ChannelHandle> CreateChannelHandle(
      ChannelHandle::ChannelType type);

  // Returns the backend used to execute computations.
  const Backend& backend() const { return *execute_backend_; }
  Backend* mutable_backend() { return execute_backend_.get(); }

  // Create a Hlo module config for the given program shape and arguments.
  // aot_options is optional; if not given a default is used.
  absl::StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
      const ProgramShape& program_shape,
      absl::Span<const Shape* const> argument_shapes,
      const ExecutionOptions* execution_options,
      const AotCompilationOptions* aot_options = nullptr);

  // Convenience function which checks whether the given client_shape
  // (presumably passed by the client to set the result layout) is valid for the
  // given computation result shape.
  static absl::Status ValidateResultShape(const Shape& client_shape,
                                          const Shape& result_shape);

  virtual ~Service() = default;

 private:
  // A private overload for Service itself, used by other methods within this
  // class.
  absl::StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
      const ProgramShape& program_shape,
      absl::Span<const ShapedBuffer* const> arguments,
      const ExecutionOptions& execution_options,
      const AotCompilationOptions* aot_options = nullptr);

  // Prepare the executors for executing parallel.
  absl::StatusOr<std::vector<se::StreamExecutor*>> GetExecutors(
      const ExecutionOptions& execution_options, int64_t requests_size,
      int64_t request_index) const;

  // Prepare the arguments for executing parallel.
  absl::StatusOr<std::vector<std::vector<const ShapedBuffer*>>> GetArguments(
      const ExecutionOptions& execution_options,
      absl::Span<const GlobalData* const> arguments) const;

 protected:
  friend class LocalExecutable;

  // The constructor is private. Use the NewService factory to create new
  // service objects.
  Service(const ServiceOptions& options,
          std::unique_ptr<Backend> execute_backend);

  // Resolves the given argument handles in the allocation tracker and returns
  // the corresponding allocations for every replica. The function also verifies
  // that each allocation matches the execution platform and device ordinal of
  // the corresponding replica.
  absl::StatusOr<std::vector<std::vector<const ShapedBuffer*>>>
  ResolveAndValidateArguments(
      absl::Span<const GlobalData* const> arguments,
      absl::Span<se::StreamExecutor* const> stream_executors) const;

 public:
  // Builds an Executable for the given parameters.
  //
  // If device_allocator is not null, the compiler may use it to allocate temp
  // buffers, which the compiler is responsible for freeing.  The allocator
  // given here need not match the allocator used when running the executable.
  absl::StatusOr<std::unique_ptr<Executable>> BuildExecutable(
      const HloModuleProto& module_proto,
      std::unique_ptr<HloModuleConfig> module_config, Backend* backend,
      se::StreamExecutor* executor, const Compiler::CompileOptions& options,
      bool run_backend_only = false);

  // Same as BuildExecutable() above, but builds a list of Executables for the
  // given computations that may interact with each other.
  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> BuildExecutables(
      const std::vector<const HloModuleProto*>& module_protos,
      std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
      Backend* backend, std::vector<std::vector<se::StreamExecutor*>> executors,
      const Compiler::CompileOptions& options, bool run_backend_only = false);

 protected:
  // Same as BuildExecutable() above, but builds a list of
  // AotCompilationResult(s), which can be persisted to later load Executable
  // objects.
  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  BuildAotResults(const std::vector<const HloModuleProto*>& module_protos,
                  std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
                  Backend* backend,
                  std::vector<std::vector<se::StreamExecutor*>> executors,
                  const Compiler::CompileOptions& options,
                  bool run_backend_only = false);

  // Runs the given executable with the given arguments and register the result
  // in the allocation tracker. The handle of the result from the tracker is
  // returned. If the parameter "profile" is not null, it points to an
  // ExecutionProfile object which will be filled in with profile data.
  absl::StatusOr<GlobalDataHandle> ExecuteAndRegisterResult(
      Executable* executable,
      absl::Span<const std::vector<const ShapedBuffer*>> arguments,
      Backend* backend, const DeviceHandle& device_handle,
      const std::string& result_tag, ExecutionProfile* profile);

  // Runs the given executables with the given arguments and register the result
  // from each executable in the allocation tracker. The handles of the result
  // from the tracker are returned.
  absl::StatusOr<std::vector<GlobalDataHandle>>
  ExecuteParallelAndRegisterResult(
      absl::Span<Executable* const> executables,
      absl::Span<const std::vector<std::vector<const ShapedBuffer*>>> arguments,
      Backend* backend, absl::Span<const DeviceHandle> device_handles,
      absl::Span<const std::string> result_tags, ExecutionProfile* profile);

  // Returns the stream executors assigned to the replicas represented by the
  // given device handle. Each device_handle is a virtual replicated device that
  // represents a set of physical devices for the replicas.
  absl::StatusOr<std::vector<se::StreamExecutor*>> Replicas(
      const Backend& backend, const DeviceHandle& device_handle) const;

  // Returns the device handle that represents the replicated device for a
  // single computation that is not model-parallelized.
  DeviceHandle SingleComputationDeviceHandle() const;

  ServiceOptions options_;

  // Cache containing previously built Executables.
  CompilationCache compilation_cache_;

  // Tracks channels created via the API.
  ChannelTracker channel_tracker_;

  // Tracks allocations made via the API and computation execution.
  AllocationTracker allocation_tracker_;

  // Tracks asynchronously launched executions via the API.
  ExecutionTracker execution_tracker_;

  // Backend to compile and execute computations on.
  std::unique_ptr<Backend> execute_backend_;

  Service(const Service&) = delete;
  Service& operator=(const Service&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_SERVICE_H_
