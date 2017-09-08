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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/allocation_tracker.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/channel_tracker.h"
#include "tensorflow/compiler/xla/service/compilation_cache.h"
#include "tensorflow/compiler/xla/service/computation_tracker.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/execution_tracker.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/service/user_computation.h"
#include "tensorflow/compiler/xla/service/versioned_computation_handle.h"
#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

// Options to configure the service when it is created.
class ServiceOptions {
 public:
  // Set the platform backing the service, or nullptr for the default platform.
  ServiceOptions& set_platform(perftools::gputools::Platform* platform);
  perftools::gputools::Platform* platform() const;

  // Set the number of replicas to use when compiling replicated
  // programs.
  ServiceOptions& set_number_of_replicas(int number_of_replicas);
  int number_of_replicas() const;

  // Sets the thread pool size for parallel execution of an individual operator.
  ServiceOptions& set_intra_op_parallelism_threads(int num_threads);
  int intra_op_parallelism_threads() const;

 private:
  perftools::gputools::Platform* platform_ = nullptr;
  int number_of_replicas_ = 1;
  int intra_op_parallelism_threads_ = -1;
};

// The XLA service object, which is the same across all platforms. It maintains
// the service state of computations and allocations, and delegates
// target-specific requests to the target-specific infrastructure
// (target-specific compiler, StreamExecutor).
class Service : public ServiceInterface {
 public:
  // Factory method for creating a new Service.
  static StatusOr<std::unique_ptr<Service>> NewService(
      perftools::gputools::Platform* platform = nullptr);
  static StatusOr<std::unique_ptr<Service>> NewService(
      const ServiceOptions& options);

  // Creates a new computation with the given name.
  // A unique ComputationHandle is returned.
  tensorflow::Status Computation(const ComputationRequest* arg,
                                 ComputationResponse* result) override;

  // Unregisters a previously-allocated global handle.
  //
  // If the handle given is not currently allocated, a NOT_FOUND status is
  // returned.
  tensorflow::Status Unregister(const UnregisterRequest* arg,
                                UnregisterResponse* result) override;

  // Deconstructs a tuple. Returns a newly created GlobalDataHandle for each
  // element in the tuple.
  tensorflow::Status DeconstructTuple(
      const DeconstructTupleRequest* arg,
      DeconstructTupleResponse* result) override;

  // Modifies the provided computation so that subsequent executions
  // will compute the provided ComputationDataHandle, rather than the
  // last expression enqueued on that Computation.
  tensorflow::Status SetReturnValue(const SetReturnValueRequest* arg,
                                    SetReturnValueResponse* results) override;

  // Executes a computation with the provided global data passed as
  // immutable arguments. Returns global data output and execution timing.
  tensorflow::Status Execute(const ExecuteRequest* arg,
                             ExecuteResponse* result) override;

  // Executes one or more computations in parallel with the provided global data
  // passed as immutable arguments. Returns global data output for each
  // computation.
  tensorflow::Status ExecuteParallel(const ExecuteParallelRequest* arg,
                                     ExecuteParallelResponse* result) override;

  // Requests one or more device handles from the target.
  //
  // When N device handles are requested and the number of replicas is R, at
  // least N * R devices must be available. The devices are assigned based on
  // the device ordinals such that the first R available devices are assigned to
  // the first set of replicas, and the next R devices to the second set of
  // replicas, etc. Each returned device handle represents the device with the
  // replica id 0.
  tensorflow::Status GetDeviceHandles(
      const GetDeviceHandlesRequest* arg,
      GetDeviceHandlesResponse* result) override;

  // Asynchronously executes a computation with provided arguments. Invokes
  // the provided computation with the provided global data passed as
  // immutable arguments. Returns a handle to the execution.
  //
  // (Note: The corresponding function in xla::Client was removed as part of
  // b/64116060, in an attempt to simplify our API.  We're keeping this around
  // for now in case we want to expose this to clients in a different way.)
  tensorflow::Status ExecuteAsync(const ExecuteAsyncRequest* arg,
                                  ExecuteAsyncResponse* result) override;

  // Waits until the specified execution is complete and returns the result.
  // Calling this API multiple times with the same execution handle returns the
  // method with an error since the execution handle is destroyed after the
  // first call.
  tensorflow::Status WaitForExecution(
      const WaitForExecutionRequest* arg,
      WaitForExecutionResponse* result) override;

  // Requests that global data be transferred to the client in literal form.
  tensorflow::Status TransferToClient(
      const TransferToClientRequest* arg,
      TransferToClientResponse* result) override;

  // Transfers data from a literal provided by the client, into device memory.
  tensorflow::Status TransferToServer(
      const TransferToServerRequest* arg,
      TransferToServerResponse* result) override;

  // Transfers data from a literal provided by the client, into the Infeed
  // buffer of the device.
  tensorflow::Status TransferToInfeed(
      const TransferToInfeedRequest* arg,
      TransferToInfeedResponse* result) override;

  // Transfers data from the Outfeed othe device to the literal provided by the
  // client.
  tensorflow::Status TransferFromOutfeed(
      const TransferFromOutfeedRequest* arg,
      TransferFromOutfeedResponse* result) override;

  // Resets devices, clearing all existing state on all the devices associated
  // with this service (including memory allocated on the devices).
  //
  // ResetDevice may only be called where no previous Execution state on the
  // device is used by the next Execution.
  //
  // ResetDevice should be called before an Execution that expect the device to
  // be in the reset state. For example, if the prior Execution modifies device
  // state (e.g., architectural state) that the next Execution depends on.
  tensorflow::Status ResetDevice(const ResetDeviceRequest* arg,
                                 ResetDeviceResponse* result) override;

  // Tests if an expression is a compile-time constant.
  tensorflow::Status IsConstant(const IsConstantRequest* arg,
                                IsConstantResponse* result) override;

  // Computes the value of a constant expression.
  tensorflow::Status ComputeConstant(const ComputeConstantRequest* arg,
                                     ComputeConstantResponse* result) override;

  // Returns the shape (with layout) of an array associated with a given data
  // handle.
  tensorflow::Status GetShape(const GetShapeRequest* arg,
                              GetShapeResponse* result) override;

  // Returns the program shape of the computation associated with the given
  // handle.
  tensorflow::Status GetComputationShape(
      const GetComputationShapeRequest* arg,
      GetComputationShapeResponse* result) override;

  /////
  // Computation-oriented methods.

  // Enqueues an Op on the computation.
  tensorflow::Status Op(const OpRequest* arg, OpResponse* result) override;

  // Retrieves the inferred shape for a value within a computation.
  tensorflow::Status GetLocalShape(const GetLocalShapeRequest* arg,
                                   GetLocalShapeResponse* result) override;

  // Retrieves the statistics of a computation.
  tensorflow::Status GetComputationStats(
      const ComputationStatsRequest* arg,
      ComputationStatsResponse* result) override;

  // Snapshots the current state of a computation handle into a serializable
  // protocol buffer form, so it can be loaded via
  // LoadComputationSnapshot.
  tensorflow::Status SnapshotComputation(
      const SnapshotComputationRequest* arg,
      SnapshotComputationResponse* result) override;

  // Loads a computation from a serialized protocol buffer created via
  // SnapshotComputation.
  tensorflow::Status LoadComputationSnapshot(
      const LoadComputationSnapshotRequest* arg,
      LoadComputationSnapshotResponse* result) override;

  // Creates a unique channel handle that can be used for Send/Recv
  // instructions.
  tensorflow::Status CreateChannelHandle(
      const CreateChannelHandleRequest* arg,
      CreateChannelHandleResponse* result) override;

  // Returns the ComputationTracker of the current service instance.
  // Only used in unit tests to access user computations from client.
  const ComputationTracker& computation_tracker() {
    return computation_tracker_;
  }

  // Returns the backend used to execute computations.
  const Backend& backend() const { return *execute_backend_; }
  Backend* mutable_backend() { return execute_backend_.get(); }

 private:
  // A private overload for Service itself, used by other methods within this
  // class.
  StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
      const ProgramShape& program_shape,
      tensorflow::gtl::ArraySlice<const Allocation*> arguments,
      const ExecutionOptions& execution_options);

 protected:
  friend class LocalExecutable;

  // The constructor is private. Use the NewService factory to create new
  // service objects.
  Service(const ServiceOptions& options,
          std::unique_ptr<Backend> execute_backend);

  static StatusOr<std::unique_ptr<Backend>> CreateComputeConstantBackend();

  // Resolves the given argument handles in the allocation tracker and returns
  // the corresponding allocations. The function also verifies that each
  // allocation matches the given backend and device ordinal.
  StatusOr<std::vector<const Allocation*>> ResolveAndValidateArguments(
      tensorflow::gtl::ArraySlice<const GlobalDataHandle*> arguments,
      const Backend* backend, int device_ordinal);

  // Create a Hlo module config for the given program shape and arguments.
  // execution_options is optional; if not given a default is used.
  // has_hybrid_result is used to initialize the same-named field in
  // HloModuleConfig -- see that class for documentation.
  StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
      const ProgramShape& program_shape,
      tensorflow::gtl::ArraySlice<const Shape*> argument_shapes,
      const ExecutionOptions* execution_options,
      bool has_hybrid_result = false);

  // Builds an Executable for the given parameters.
  StatusOr<std::unique_ptr<Executable>> BuildExecutable(
      const VersionedComputationHandle& versioned_handle,
      std::unique_ptr<HloModuleConfig> module_config,
      const tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments,
      Backend* backend, perftools::gputools::StreamExecutor* executor);

  // Same as BuildExecutable() above, but builds a list of Executables for the
  // given computations that may interact with each other.
  StatusOr<std::vector<std::unique_ptr<Executable>>> BuildExecutables(
      std::vector<VersionedComputationHandle> versioned_handles,
      std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
      Backend* backend,
      std::vector<perftools::gputools::StreamExecutor*> executors);

  // Similar to BuildExecutable, but look in the compilation cache for the
  // executable first. If the executable is not in the cache, it is built and
  // inserted into the cache.
  StatusOr<std::shared_ptr<Executable>> BuildAndCacheExecutable(
      const VersionedComputationHandle& versioned_handle,
      std::unique_ptr<HloModuleConfig> module_config,
      const tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments,
      Backend* backend, perftools::gputools::StreamExecutor* executor,
      ExecutionProfile* profile);

  // Runs the given executable with the given arguments and register the result
  // in the allocation tracker. The handle of the result from the tracker is
  // returned. If the parameter "profile" is not null, it points to an
  // ExecutionProfile object which will be filled in with profile data.
  StatusOr<GlobalDataHandle> ExecuteAndRegisterResult(
      Executable* executable,
      const tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments,
      Backend* backend, perftools::gputools::StreamExecutor* executor,
      const string& result_tag, ExecutionProfile* profile);

  // Runs the given executables with the given arguments and register the result
  // from each executable in the allocation tracker. The handles of the result
  // from the tracker are returned.
  StatusOr<std::vector<GlobalDataHandle>> ExecuteParallelAndRegisterResult(
      tensorflow::gtl::ArraySlice<Executable*> executables,
      tensorflow::gtl::ArraySlice<
          std::vector<perftools::gputools::DeviceMemoryBase>>
          arguments,
      Backend* backend,
      tensorflow::gtl::ArraySlice<DeviceHandle> device_handles,
      tensorflow::gtl::ArraySlice<string> result_tags);

  // Convenience function for adding a function to a user computation.
  template <typename RequestT, typename ResponseT>
  tensorflow::Status AddInstruction(
      const RequestT* arg, ResponseT* result,
      const std::function<StatusOr<ComputationDataHandle>(UserComputation*)>&
          adder);

  // Convenience function which checks whether the given shape_with_layout
  // (presumably passed by the client to set the result layout) is valid for the
  // given computation result shape.
  tensorflow::Status ValidateResultShapeWithLayout(
      const Shape& shape_with_layout, const Shape& result_shape) const;

  // Returns the stream executors assigned to the replicas represented by the
  // given device handle. Each device_handle is a virtual replicated device that
  // represents a set of physical devices for the replicas.
  StatusOr<std::vector<perftools::gputools::StreamExecutor*>> Replicas(
      const Backend& backend, const DeviceHandle& device_handle) const;

  // Returns the device handle that represents the replicated device for a
  // single computation that is not model-parallelized.
  DeviceHandle SingleComputationDeviceHandle() const;

  ServiceOptions options_;

  // Tracks computations built via the API.
  ComputationTracker computation_tracker_;

  // Tracks channels created via the API.
  ChannelTracker channel_tracker_;

  // Tracks allocations made via the API and computation execution.
  AllocationTracker allocation_tracker_;

  // Tracks asynchronously launched executions via the API.
  ExecutionTracker execution_tracker_;

  // Cache containing previously built Executables.
  CompilationCache compilation_cache_;

  // Backend to compile and execute computations on.
  //
  // TODO(b/28616830): Support multiple backends for execution.
  std::unique_ptr<Backend> execute_backend_;

  TF_DISALLOW_COPY_AND_ASSIGN(Service);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_H_
