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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_CONV_UTILS_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_CONV_UTILS_H_

#include <optional>
#include <unordered_map>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/sycl/onednn_util.h"

namespace stream_executor {

namespace sycl {

struct ConvFwd {
  dnnl::convolution_forward primitive;
  dnnl::memory src;
  dnnl::memory filter;
  dnnl::memory dst;
  dnnl::memory bias;
  dnnl::memory side_input;
  dnnl::memory internal_filter;
  dnnl::memory scratchpad;
  std::unordered_map<int, dnnl::memory> args;
};

struct ConvBwdData {
  dnnl::convolution_backward_data primitive;
  dnnl::memory src;
  dnnl::memory filter;
  dnnl::memory dst;
  dnnl::memory internal_filter;
  dnnl::memory scratchpad;
  std::unordered_map<int, dnnl::memory> args;
};

struct ConvBwdWeights {
  dnnl::convolution_backward_weights primitive;
  dnnl::memory src;
  dnnl::memory filter;
  dnnl::memory dst;
  dnnl::memory internal_filter;
  dnnl::memory scratchpad;
  std::unordered_map<int, dnnl::memory> args;
};

struct ReorderOp {
  dnnl::reorder primitive;
  std::unordered_map<int, dnnl::memory> args;
};

struct OneDnnConvPrimitive {
  dnnl::engine engine;
  dnnl::stream stream;
  std::variant<ConvFwd, ConvBwdData, ConvBwdWeights> op;
  std::optional<ReorderOp> filter_reorder;
  std::optional<ReorderOp> side_input_reorder;
};

absl::StatusOr<OneDnnConvPrimitive> CreateOneDnnConvPrimitive(
    const xla::gpu::GpuConvConfig& config,
    absl::Span<const DeviceAddressBase> operand_buffers,
    DeviceAddressBase result_buffer, Stream* stream,
    ScratchAllocator* scratch_allocator);

absl::Status DoOnednnConv(const OneDnnConvPrimitive& onednn_primitive);

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_CONV_UTILS_H_
