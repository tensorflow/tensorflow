/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_INTERFACE_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_INTERFACE_H_

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_profile_printer_data.pb.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace xla {

// An executable capable of being fed to a TPU device.
class TpuExecutableInterface : public Executable {
 public:
  explicit TpuExecutableInterface(std::shared_ptr<HloModule> hlo_module)
      : Executable(std::move(hlo_module)) {}
  ~TpuExecutableInterface() override = default;

  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  // Same as AllocateOutputMemory, except that input buffers can be reused
  // as output buffers. See UserBufferAlias class comment for more details on
  // the buffer reuse.
  //
  // `alias_config` indicates which input and output buffers can be aliased.
  //
  // `arguments` are ExecutionInput containing the input parameters. Currently
  // only a single input parameter (typically a tuple) is supported on TPU. For
  // each element in the shape tree, if the element holds the ownership of the
  // memory, it is considered donated and XLA will potentially reuse it as
  // output buffers.
  //
  // The optional 'transfer_stream' parameter enables transfers (for tuple
  // tables) to be performed on a separate stream to 'stream'.
  StatusOr<ExecutionOutput> AllocateOutputMemoryWithInputReuse(
      const Shape& shape, const HloInputOutputAliasConfig& alias_config,
      se::DeviceMemoryAllocator* allocator,
      std::vector<ExecutionInput>* arguments, se::Stream* stream,
      se::Stream* transfer_stream = nullptr);

  virtual Status LoadProgramAndEnqueueToStream(
      const ServiceExecutableRunOptions& run_options,
      absl::Span<const stream_executor::DeviceMemoryBase> arguments,
      stream_executor::DeviceMemoryBase result,
      absl::optional<stream_executor::DeviceMemoryBase>
          cross_program_prefetch_addr) = 0;

  virtual absl::string_view fingerprint() const = 0;

 protected:
  virtual Shape HostShapeToDeviceShape(const Shape& host_shape) = 0;

  virtual int64_t ShapeSize(const Shape& shape) = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_INTERFACE_H_
