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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTABLE_H_

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/versioned_computation_handle.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace xla {

// A given platform's compiler will produce an Executable -- this is a uniform
// interface that is used for launching compiled programs across platforms.
class Executable {
 public:
  explicit Executable(std::unique_ptr<const HloModule> hlo_module,
                      std::unique_ptr<HloProfilePrinter> hlo_profile_printer,
                      std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
      : hlo_module_(std::move(hlo_module)),
        hlo_profile_printer_(std::move(hlo_profile_printer)),
        hlo_profile_index_map_(std::move(hlo_profile_index_map)) {
    CHECK_EQ(hlo_profile_printer_.get() == nullptr,
             hlo_profile_index_map_.get() == nullptr);
  }
  virtual ~Executable() {}

  // Enqueues the compilation result on the provided stream, passing the given
  // arguments. This call is blocking and returns after the execution is done.
  //
  // If the hlo_execution_profile is provided as non-nullptr, profiling will be
  // enabled.
  //
  // Returns the device memory region that a successful execution would
  // populate.
  virtual StatusOr<perftools::gputools::DeviceMemoryBase> ExecuteOnStream(
      const ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments,
      HloExecutionProfile* hlo_execution_profile) = 0;

  // Overload of ExecuteOnStream which returns and takes arguments as
  // ShapedBuffers. Used for LocalService execution.
  virtual StatusOr<std::unique_ptr<ShapedBuffer>> ExecuteOnStream(
      const ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      HloExecutionProfile* hlo_execution_profile) = 0;

  // Same as ExecuteOnStream(), but this call is non-blocking and returns as
  // soon as all of the operations are enqueued for launch on the stream.
  virtual StatusOr<perftools::gputools::DeviceMemoryBase> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments) = 0;

  // Same as ExecuteOnStream(), but runs this executable on multiple
  // streams. arguments[i] contains the arguments to the execution on
  // run_options[i]->stream() and the returned value is at index i of the
  // returned vector.
  virtual StatusOr<std::vector<perftools::gputools::DeviceMemoryBase>>
  ExecuteOnStreams(
      tensorflow::gtl::ArraySlice<const ServiceExecutableRunOptions>
          run_options,
      tensorflow::gtl::ArraySlice<
          tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>>
          arguments);

  // Populates `hlo_execution_profile` from `executor`. This is implicit in any
  // Execute* API call that takes a hlo_execution_profile argument, but must be
  // called explicitly for other (async, for example) variants after the stream
  // has completed.
  virtual Status PopulateExecutionProfile(
      HloExecutionProfile* hlo_execution_profile,
      perftools::gputools::StreamExecutor* executor) {
    return Status::OK();
  }

  // Convenience wrapper for calling Executable::ExecuteOnStream. Sets up a
  // timer for the execution, sets up HLO profiling if enabled, and fills in the
  // given ExecutionProfile if non-null.  The ExecuteOnStream overloads have
  // different argument types and return types, so this method is templated on
  // argument type and return type of the execute function.
  template <typename ReturnT, typename ArgT>
  StatusOr<ReturnT> ExecuteOnStreamWrapper(
      const ServiceExecutableRunOptions* run_options, ExecutionProfile* profile,
      const ArgT& arguments);

  // Returns the ExecutionProfile from executing on the device. This includes
  // the number of cycles taken for the computation or the compilation time.
  ExecutionProfile execution_profile() const {
    tensorflow::mutex_lock lock(mutex_);
    return execution_profile_;
  }

  // Returns Status::ok() if the two executables are equal to each other.
  //
  // An error status is returned otherwise.
  virtual const Status EqualOrFail(const Executable& executable) {
    return Unimplemented(
        "Equality test on this executable is not implemented.");
  }

  const HloProfilePrinter& hlo_profile_printer() const {
    CHECK(hlo_profiling_enabled());
    return *hlo_profile_printer_;
  }

  const HloProfileIndexMap& hlo_profile_index_map() const {
    CHECK(hlo_profiling_enabled());
    return *hlo_profile_index_map_;
  }

  // Returns whether this executable was compiled with HLO profilings support
  // enabled. If not, the caller should not expect an hlo_execution_profile
  // passed to ExecuteOnStream above to be populated during execution.
  bool hlo_profiling_enabled() const { return hlo_profile_printer_ != nullptr; }

  const HloModule& module() const { return *hlo_module_; }

  const bool has_module() const { return hlo_module_ != nullptr; }

  const HloModuleConfig& module_config() const { return hlo_module_->config(); }

  // Returns the versioned computation handle of the computation computed by
  // this executable.
  const VersionedComputationHandle& entry_computation_handle() const {
    return hlo_module_->entry_computation_handle();
  }

  // The shape (including layout) that results from this execution. This is the
  // shape of the DeviceMemoryBase result value in ExecuteOnStream above.
  const Shape& result_shape() const {
    return hlo_module_->config().entry_computation_layout().result_shape();
  }

  // Dumping helpers.
  void set_session_module(std::unique_ptr<xla::SessionModule> session_module) {
    session_module_ = std::move(session_module);
  }
  bool dumping() const { return session_module_ != nullptr; }
  SessionModule* session_module() const { return session_module_.get(); }
  Status DumpSessionModule();

  // Dump session_module to directory_path/filename.
  static Status DumpToDirectory(const string& directory_path, string filename,
                                const SessionModule& session_module);

 protected:
  mutable tensorflow::mutex mutex_;

  // Execution profile data on the device.
  ExecutionProfile execution_profile_ GUARDED_BY(mutex_);

  // HloModule this was compiled from. BufferAssignment keeps pointers to
  // HloInstructions owned by the HloModule so we need to keep the HloModule
  // around.
  const std::unique_ptr<const HloModule> hlo_module_;

  // SessionModule this was compiled from. Null if not dumping executions.
  std::unique_ptr<SessionModule> session_module_;

  // Execution count, used to generate a unique filename for each dumped
  // execution.
  int64 execution_count_ = 0;

  std::unique_ptr<HloProfilePrinter> hlo_profile_printer_;
  std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map_;
};

template <typename ReturnT, typename ArgT>
StatusOr<ReturnT> Executable::ExecuteOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options, ExecutionProfile* profile,
    const ArgT& arguments) {
  perftools::gputools::Stream* stream = run_options->stream();
  std::unique_ptr<perftools::gputools::Timer> timer;
  if (profile != nullptr) {
    timer.reset(new perftools::gputools::Timer(stream->parent()));
    stream->InitTimer(timer.get()).ThenStartTimer(timer.get());
  }

  VLOG(1) << "enqueueing executable on stream...";
  // If the profiling flag isn't enabled, we pass nullptr as the profile to
  // indicate profiling is not requested.
  std::unique_ptr<HloExecutionProfile> profile_ptr =
      module_config().debug_options().xla_hlo_profile() &&
              hlo_profiling_enabled()
          ? MakeUnique<HloExecutionProfile>(&hlo_profile_printer(),
                                            &hlo_profile_index_map())
          : nullptr;

  auto return_value =
      ExecuteOnStream(run_options, arguments, profile_ptr.get());

  if (profile != nullptr) {
    VLOG(1) << "enqueueing 'stop timer' and blocking host until done...";
    stream->ThenStopTimer(timer.get()).BlockHostUntilDone();
    VLOG(1) << "done with block-host-until-done";

    // Merge in run-time profile information from execution_profile.
    profile->MergeFrom(execution_profile());

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
    XLA_LOG_LINES(
        tensorflow::INFO,
        profile_ptr->ToString(stream->parent()->GetDeviceDescription()));
    hlo_graph_dumper::MaybeDumpHloModule(module(), "Service::Execute",
                                         profile_ptr.get());
  }

  return return_value;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTABLE_H_
