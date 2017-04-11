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
#include "tensorflow/compiler/xla/legacy_flags/service_flags.h"
#include "tensorflow/compiler/xla/service/allocation_tracker.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/channel_tracker.h"
#include "tensorflow/compiler/xla/service/compilation_cache.h"
#include "tensorflow/compiler/xla/service/compiler.h"
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
  // programs. The default is -1 meaning that the value is read from
  // the xla_replicas flag.
  ServiceOptions& set_number_of_replicas(int number_of_replicas);
  int number_of_replicas() const;

 private:
  perftools::gputools::Platform* platform_ = nullptr;
  int number_of_replicas_ = -1;
};

// The XLA service object, which is the same across all
// platforms. It maintains the service state of computations and allocations,
// and delegates target-specific requests to the target-specific infrastructure
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
  // replicas, etc. Each returned device handles represent the device with the
  // replica id 0.
  tensorflow::Status GetDeviceHandles(
      const GetDeviceHandlesRequest* arg,
      GetDeviceHandlesResponse* result) override;

  // Asynchronously executes a computation with provided arguments. Invokes
  // the provided computation with the provided global data passed as
  // immutable arguments. Returns a handle to the execution.
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

  // Requests that global data be copied into a buffer supplied by the client.
  tensorflow::Status TransferToClientInProcess(
      const TransferToClientInProcessRequest* arg,
      TransferToClientInProcessResponse* result) override;

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

  // Transfers data from a buffer provided by the client, into device memory.
  tensorflow::Status TransferToServerInProcess(
      const TransferToServerInProcessRequest* arg,
      TransferToServerInProcessResponse* result) override;

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

 protected:
  friend class LocalExecutable;

  // The constructor is private. Use the NewService factory to create new
  // service objects.
  Service(std::unique_ptr<Backend> backend,
          std::unique_ptr<Backend> compute_constant_backend);

  static StatusOr<std::unique_ptr<Backend>> CreateComputeConstantBackend();

  // Resolves the given argument handles in the allocation tracker and returns
  // the corresponding allocations. The function also verifies that each
  // allocation matches the given backend and device ordinal.
  StatusOr<std::vector<const Allocation*>> ResolveAndValidateArguments(
      tensorflow::gtl::ArraySlice<const GlobalDataHandle*> arguments,
      const Backend* backend, int device_ordinal);

  // Create a Hlo module config foe the given program shape and arguments.
  StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
      const ProgramShape& program_shape,
      tensorflow::gtl::ArraySlice<const Allocation*> arguments,
      const ExecutionOptions& execution_options);

  // Builds an Executable for the given parameters. If
  // executable_for_compute_constant is true, then the executable is intended to
  // be used for ComputeConstant which means dead parameter instructions are not
  // included in the executable.The parameter "profile" can optionally point to
  // an ExecutionProfile object which will be filled in with profile data
  // relevant to compilation.
  StatusOr<std::unique_ptr<Executable>> BuildExecutable(
      const VersionedComputationHandle& versioned_handle,
      std::unique_ptr<HloModuleConfig> module_config,
      bool executable_for_compute_constant,
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
      tensorflow::gtl::ArraySlice<perftools::gputools::StreamExecutor*>
          executors,
      tensorflow::gtl::ArraySlice<string> result_tags);

  // Dumps the executed HLO according to service-associated flags.
  static void DumpExecutedHlo(const HloModule& module, const string& label,
                              const HloExecutionProfile* profile);

  // Returns an HLO dumper for use in the compiler (it refers to flags
  // associated with the service).
  static Compiler::HloDumper MakeHloDumper();

  // Convenience function for adding a function to a user computation.
  template <typename RequestT, typename ResponseT>
  tensorflow::Status AddInstruction(
      const RequestT* arg, ResponseT* result,
      const std::function<StatusOr<ComputationDataHandle>(UserComputation*)>&
          adder);

  // If the service is running in the client process
  // (runs_in_client_process_ is true) then return
  // tensorflow::Status::OK. Otherwise return an appropriate error
  // status with the given method name. Used for "InProcess" methods.
  tensorflow::Status CheckRunsInClientProcess(const string& method_name) const;

  // Convenience function which checks whether the given shape_with_layout
  // (presumably passed by the client to set the result layout) is valid for the
  // given computation result shape.
  tensorflow::Status ValidateResultShapeWithLayout(
      const Shape& shape_with_layout, const Shape& result_shape) const;

  // Convenience wrapper for calling Executable::ExecuteOnStream. Sets up a
  // timer for the execution, sets up HLO profiling if enabled, and fills in the
  // given ExecutionProfile if non-null. The given execute_func should be a
  // function which calls the desired ExecuteOnStream overload with the supplied
  // arguments. The ExecuteOnStream overloads return different types so this
  // method is templated on return-type of the execute function.
  template <typename ReturnT>
  static ReturnT ExecuteOnStreamWrapper(
      Executable* executable, const ServiceExecutableRunOptions* run_options,
      ExecutionProfile* profile, Backend* backend,
      std::function<ReturnT(Executable* executable,
                            const ServiceExecutableRunOptions* run_options,
                            HloExecutionProfile* hlo_execution_profile)>
          execute_func);

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

  // Backend to use when executing ComputeConstant.
  std::unique_ptr<Backend> compute_constant_backend_;

  // Whether the service runs in the same process as the client.
  bool runs_in_client_process_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(Service);
};

template <typename ReturnT>
ReturnT Service::ExecuteOnStreamWrapper(
    Executable* executable, const ServiceExecutableRunOptions* run_options,
    ExecutionProfile* profile, Backend* backend,
    std::function<ReturnT(Executable* executable,
                          const ServiceExecutableRunOptions* run_options,
                          HloExecutionProfile* hlo_execution_profile)>
        execute_func) {
  perftools::gputools::Stream* stream = run_options->stream();
  std::unique_ptr<perftools::gputools::Timer> timer;
  if (profile != nullptr) {
    timer.reset(new perftools::gputools::Timer(stream->parent()));
    stream->InitTimer(timer.get()).ThenStartTimer(timer.get());
  }

  VLOG(1) << "enqueueing executable on stream...";
  // If the profiling flag isn't enabled, we pass nullptr as the profile to
  // indicate profiling is not requested.
  HloExecutionProfile hlo_execution_profile;
  legacy_flags::ServiceFlags* flags = legacy_flags::GetServiceFlags();
  HloExecutionProfile* profile_ptr =
      flags->xla_hlo_profile && executable->hlo_profiling_enabled()
          ? &hlo_execution_profile
          : nullptr;

  auto return_value = execute_func(executable, run_options, profile_ptr);

  if (profile != nullptr) {
    VLOG(1) << "enqueueing 'stop timer' and blocking host until done...";
    stream->ThenStopTimer(timer.get()).BlockHostUntilDone();
    VLOG(1) << "done with block-host-until-done";

    // Merge in run time profile information from the executable.
    profile->MergeFrom(executable->execution_profile());

    // Overall execution time (in nanoseconds) from the executor timer.
    profile->set_compute_and_transfer_time_ns(timer->Nanoseconds());

    // TODO(b/28123297): On GPU we end up including transfer time in
    // the compute time this way. Instead, we should get the correct
    // value by measuring it. Setting the field here at least lets
    // benchmarks provide *some* value for GPU computations.
    //
    // TODO(b/28447609): The value in compute_and_transfer_time_ns is actually
    // the compute time without the transfer time, so this way we get the
    // correct compute time. We should instead have the correct value for
    // compute_and_transfer_time and set compute_time to the compute time.
    if (profile->compute_time_ns() == 0) {
      profile->set_compute_time_ns(profile->compute_and_transfer_time_ns());
    }
  }

  if (profile_ptr != nullptr) {
    HloCostAnalysis::ShapeSizeFunction shape_size =
        [backend](const Shape& shape) {
          return backend->compiler()->ShapeSizeBytes(shape);
        };
    std::unordered_set<const xla::HloComputation*> profiled_computations =
        profile_ptr->profiled_computations();
    // To ensure we have print the profiles in a stable order, iterate over the
    // computations in post order.
    std::list<xla::HloComputation*> all_computations =
        executable->module().MakeComputationPostOrder();
    for (xla::HloComputation* computation : all_computations) {
      if (profiled_computations.count(computation) > 0) {
        string profile_string = profile_ptr->ToString(
            *computation, stream->parent()->GetDeviceDescription(), shape_size);
        if (!profile_string.empty()) {
          XLA_LOG_LINES(tensorflow::INFO, profile_string);
        }
      }
    }
    DumpExecutedHlo(executable->module(), "Service::Execute", profile_ptr);
  }

  return return_value;
}
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_H_
