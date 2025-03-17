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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_INTERFACE_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_INTERFACE_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

// An executable capable of being fed to a TPU device.
class TpuExecutableInterface : public Executable {
 public:
  explicit TpuExecutableInterface(std::shared_ptr<HloModule> hlo_module)
      : Executable(std::move(hlo_module)) {}
  ~TpuExecutableInterface() override = default;

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override;

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
  absl::StatusOr<ExecutionOutput> AllocateOutputMemoryWithInputReuse(
      const Shape& shape, const HloInputOutputAliasConfig& alias_config,
      se::DeviceMemoryAllocator* allocator,
      std::vector<ExecutionInput>* arguments, se::Stream* stream,
      se::Stream* transfer_stream = nullptr);

  virtual absl::Status LoadProgramAndEnqueueToStream(
      const ServiceExecutableRunOptions& run_options,
      absl::Span<const stream_executor::DeviceMemoryBase> arguments,
      stream_executor::DeviceMemoryBase result,
      const std::vector<stream_executor::DeviceMemoryBase>&
          cross_program_prefetch_addrs,
      const std::vector<uint32_t>& cross_program_prefetch_offsets) = 0;

  virtual absl::string_view fingerprint() const = 0;
};

}  // namespace xla

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTABLE_INTERFACE_H_
