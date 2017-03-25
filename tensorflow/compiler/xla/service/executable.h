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

#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/versioned_computation_handle.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace xla {

// A given platform's compiler will produce an Executable -- this is a uniform
// interface that is used for launching compiled programs across platforms.
//
// TODO(leary) will need to extend this to support multiple streams/devices as
// we begin to compile single programs to run on multiple devices.
class Executable {
 public:
  explicit Executable(std::unique_ptr<HloModule> hlo_module,
                      std::unique_ptr<HloModuleConfig> module_config)
      : hlo_module_(std::move(hlo_module)),
        module_config_(std::move(module_config)) {}
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

  // Returns the ExecutionProfile from executing on the device. This includes
  // the number of cycles taken for the computation or the compilation time.
  ExecutionProfile execution_profile() const {
    tensorflow::mutex_lock lock(mutex_);
    return execution_profile_;
  }

  // Returns whether this executable was compiled with HLO profilings support
  // enabled. If not, the caller should not expect an hlo_execution_profile
  // passed to ExecuteOnStream above to be populated during execution.
  bool hlo_profiling_enabled() const {
    return module_config_->hlo_profiling_enabled();
  }

  const HloModule& module() const { return *hlo_module_; }

  const HloModuleConfig& module_config() const { return *module_config_; }

  // Returns whether this executable has an associated HloModuleConfig.
  bool has_module_config() const { return module_config_ != nullptr; }

  // Returns the versioned computation handle of the computation computed by
  // this executable.
  const VersionedComputationHandle& entry_computation_handle() const {
    return hlo_module_->entry_computation_handle();
  }

  // The shape (including layout) that results from this execution. This is the
  // shape of the DeviceMemoryBase result value in ExecuteOnStream above.
  const Shape& result_shape() const {
    return module_config_->entry_computation_layout().result_shape();
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
  std::unique_ptr<HloModule> hlo_module_;

  // The configuration used to build this executable (parameter layouts, result
  // layout, profiling enabled, etc).
  std::unique_ptr<HloModuleConfig> module_config_;

  // SessionModule this was compiled from. Null if not dumping executions.
  std::unique_ptr<SessionModule> session_module_;

  // Execution count, used to generate a unique filename for each dumped
  // execution.
  int64 execution_count_ = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTABLE_H_
