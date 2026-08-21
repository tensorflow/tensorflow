/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_MOCK_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_MOCK_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/testlib/test.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {

// Implements CommandBuffer for testing.
class MockCommandBuffer : public CommandBuffer {
 public:
  MockCommandBuffer() = default;

  MOCK_METHOD(absl::StatusOr<const Command*>, CreateEmptyCmd,
              (absl::Span<const Command* const> dependencies,
               StreamPriority priority),
              (override));
  MOCK_METHOD(absl::StatusOr<const Command*>, CreateLaunch,
              (const ThreadDim& threads, const BlockDim& blocks,
               const std::optional<ClusterDim>& cluster_dims,
               const Kernel& kernel, const KernelArgs& args,
               absl::Span<const Command* const> dependencies,
               StreamPriority priority),
              (override));
  MOCK_METHOD(absl::Status, UpdateLaunch,
              (const Command* command, const ThreadDim& threads,
               const BlockDim& blocks,
               const std::optional<ClusterDim>& cluster_dims,
               const Kernel& kernel, const KernelArgs& args),
              (override));
  MOCK_METHOD(absl::StatusOr<const Command*>, CreateChildCommand,
              (const CommandBuffer& nested,
               absl::Span<const Command* const> dependencies),
              (override));
  MOCK_METHOD(absl::Status, UpdateChildCommand,
              (const Command* command, const CommandBuffer& nested),
              (override));
  MOCK_METHOD(absl::StatusOr<const Command*>, CreateChildCommand,
              (absl::AnyInvocable<absl::Status(CommandBuffer*)> record_fn,
               absl::Span<const Command* const> dependencies),
              (override));
  MOCK_METHOD(absl::Status, UpdateChildCommand,
              (const Command* command,
               absl::AnyInvocable<absl::Status(CommandBuffer*)> update_fn),
              (override));
  MOCK_METHOD(absl::StatusOr<const Command*>, CreateMemcpyD2D,
              (DeviceAddressBase * dst, const DeviceAddressBase& src,
               uint64_t size, absl::Span<const Command* const> dependencies),
              (override));
  MOCK_METHOD(absl::Status, UpdateMemcpyD2D,
              (const Command* command, DeviceAddressBase* dst,
               const DeviceAddressBase& src, uint64_t size),
              (override));
  MOCK_METHOD(absl::StatusOr<const Command*>, CreateMemset,
              (DeviceAddressBase * dst, BitPattern bit_pattern,
               size_t num_elements,
               absl::Span<const Command* const> dependencies),
              (override));
  MOCK_METHOD(absl::Status, UpdateMemset,
              (const Command* command, DeviceAddressBase* dst,
               const BitPattern& bit_pattern, size_t num_elements),
              (override));
  MOCK_METHOD(absl::StatusOr<const Command*>, CreateDnnGraphCommand,
              (dnn::DnnGraph&, Stream&, absl::Span<DeviceAddressBase> operands,
               absl::Span<const Command* const> dependencies),
              (override));
  MOCK_METHOD(absl::Status, UpdateDnnGraphCommand,
              (const Command*, dnn::DnnGraph&, Stream&,
               absl::Span<DeviceAddressBase> operands),
              (override));
  MOCK_METHOD(absl::StatusOr<const Command*>, CreateCase,
              (DeviceAddress<int32_t> index,
               std::vector<CreateCommands> create_branches,
               absl::Span<const Command* const> dependencies),
              (override));
  MOCK_METHOD(absl::StatusOr<const Command*>, CreateCase,
              (DeviceAddress<bool> index,
               std::vector<CreateCommands> create_branches,
               absl::Span<const Command* const> dependencies),
              (override));
  MOCK_METHOD(absl::Status, UpdateCase,
              (const Command* command, DeviceAddress<int32_t> index,
               std::vector<UpdateCommands> update_branches),
              (override));
  MOCK_METHOD(absl::Status, UpdateCase,
              (const Command* command, DeviceAddress<bool> index,
               std::vector<UpdateCommands> update_branches),
              (override));
  MOCK_METHOD(absl::StatusOr<const Command*>, CreateWhile,
              (DeviceAddress<bool> pred, CreateCommands create_cond,
               CreateCommands create_body,
               absl::Span<const Command* const> dependencies),
              (override));
  MOCK_METHOD(absl::Status, UpdateWhile,
              (const Command* command, DeviceAddress<bool> pred,
               UpdateCommands update_cond, UpdateCommands update_body),
              (override));
  MOCK_METHOD(absl::Status, SetPriority, (StreamPriority priority), (override));
  MOCK_METHOD(absl::Status, Submit, (Stream * stream), (override));
  MOCK_METHOD(absl::Status, Finalize, (), (override));
  MOCK_METHOD(absl::Status, Update, (), (override));
  MOCK_METHOD(Mode, mode, (), (const, override));
  MOCK_METHOD(State, state, (), (const, override));
  MOCK_METHOD(std::string, ToString, (), (const, override));

 private:
  MOCK_METHOD(absl::Status, Trace,
              (Stream * stream,
               absl::AnyInvocable<absl::Status(Stream*)> function),
              (override));
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MOCK_COMMAND_BUFFER_H_
