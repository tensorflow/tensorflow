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

#ifndef XLA_SERVICE_CPU_CPU_TRANSFER_MANAGER_H_
#define XLA_SERVICE_CPU_CPU_TRANSFER_MANAGER_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/xfeed_manager.h"
#include "xla/literal.h"
#include "xla/service/generic_transfer_manager.h"
#include "xla/service/transfer_manager.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// An implementation of the XLA GenericTransferManager that
// handles CPU-specific infeed.
class CpuTransferManager : public GenericTransferManager {
 public:
  CpuTransferManager();
  ~CpuTransferManager() override {}

  absl::Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                       const LiteralSlice& literal) override;
  absl::Status TransferLiteralFromOutfeed(
      se::StreamExecutor* executor, MutableBorrowingLiteral literal) override;

  bool CanShapedBufferBeAccessedNow(
      se::StreamExecutor* executor,
      const ShapedBuffer& device_buffer) const override {
    return true;
  }

  bool CanBufferBeAccessedNow(
      se::StreamExecutor* executor,
      const se::DeviceMemoryBase& device_buffer) const override {
    return true;
  }

  absl::Status ReadDynamicShapes(se::Stream* stream,
                                 const ShapedBuffer* device_buffer,
                                 Shape* device_shape) override;

 private:
  bool PackSubbyteTypes() const override { return true; }

  CpuTransferManager(const CpuTransferManager&) = delete;
  CpuTransferManager& operator=(const CpuTransferManager&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_CPU_CPU_TRANSFER_MANAGER_H_
