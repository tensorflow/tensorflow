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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/protobuf/dnn.pb.h"

namespace stream_executor::gpu {
namespace {

// Physical (unpacked) problem shape. MHA (h_q == h_k) so that the GQA
// full-head dK/dV buffers do not enter the picture.
constexpr int64_t kBatch = 2;
constexpr int64_t kSeqLen = 2048;
constexpr int64_t kNumHeads = 16;
constexpr int64_t kHeadDim = 128;

absl::StatusOr<StreamExecutor*> GpuExecutor() {
  ABSL_ASSIGN_OR_RETURN(std::string canonical_name,
                   xla::PlatformUtil::CanonicalPlatformName("gpu"));
  ABSL_ASSIGN_OR_RETURN(
      Platform * platform,
      PlatformManager::PlatformWithName(absl::AsciiStrToUpper(canonical_name)));
  return platform->ExecutorForDevice(0);
}

// Builds the packed flash attention backward graph for the physical shape
// [kBatch, kSeqLen, kNumHeads, kHeadDim] with `max_seg_per_batch` segments per
// row and returns the cuDNN workspace size in bytes. Descriptors mirror the
// ones produced by cudnn_custom_call_compiler for a BSHD-layout backward call.
absl::StatusOr<int64_t> PackedBackwardWorkspaceSize(StreamExecutor* executor,
                                                    int max_seg_per_batch) {
  constexpr dnn::DataType kType = dnn::DataType::kBF16;
  const std::vector<int64_t> bshd = {kBatch, kSeqLen, kNumHeads, kHeadDim};
  const std::vector<int64_t> bshd_layout = {3, 2, 1, 0};
  // Gradients are BHSD-shaped with BSHD physical layout.
  const std::vector<int64_t> bhsd = {kBatch, kNumHeads, kSeqLen, kHeadDim};
  const std::vector<int64_t> bhsd_layout = {3, 1, 2, 0};

  dnn::MatmulTensorDescriptor q = dnn::MatmulTensorDescriptor::For(
      kType, bshd, bshd_layout, /*batch_dims=*/{0, 2},
      /*contracting_dims=*/{1});
  dnn::MatmulTensorDescriptor k = dnn::MatmulTensorDescriptor::For(
      kType, bshd, bshd_layout, /*batch_dims=*/{0, 2},
      /*contracting_dims=*/{1});
  dnn::MatmulTensorDescriptor p = dnn::MatmulTensorDescriptor::For(
      kType, {kBatch, kNumHeads, kSeqLen, kSeqLen}, bshd_layout,
      /*batch_dims=*/{0, 1}, /*contracting_dims=*/{2});
  dnn::MatmulTensorDescriptor v = dnn::MatmulTensorDescriptor::For(
      kType, bshd, bshd_layout, /*batch_dims=*/{0, 2},
      /*contracting_dims=*/{3});
  dnn::MatmulTensorDescriptor d_output = dnn::MatmulTensorDescriptor::For(
      kType, bshd, bshd_layout, /*batch_dims=*/{0, 2},
      /*contracting_dims=*/{1});
  dnn::TensorDescriptor dq =
      dnn::TensorDescriptor::For(kType, bhsd, bhsd_layout);
  dnn::TensorDescriptor dk =
      dnn::TensorDescriptor::For(kType, bhsd, bhsd_layout);
  dnn::TensorDescriptor dv =
      dnn::TensorDescriptor::For(kType, bhsd, bhsd_layout);

  ABSL_ASSIGN_OR_RETURN(
      CudnnGraph graph,
      GetCudnnFlashAttentionBackwardOperationGraph(
          executor->AsDnn(), executor->GetDeviceDescription(), q, k, p, v,
          d_output, dq, dk, dv, /*bias_descriptor=*/std::nullopt,
          /*dbias_descriptor=*/std::nullopt, /*dropout_rate=*/std::nullopt,
          /*seed=*/std::nullopt, /*scale=*/0.1, /*use_dropout=*/false,
          /*use_bias=*/false, dnn::FMHAMaskKind::NO_MASK,
          /*force_deterministic=*/false, /*sliding_window_length=*/0,
          max_seg_per_batch, /*score_mod=*/nullptr));
  return graph.Graph().get_workspace_size();
}

// The packed backward graph declares Q/K/V/dO/O with batch `B * M`, but real
// tokens can never exceed `B * S`. GetCudnnFlashAttentionBackwardOperationGraph
// passes that bound to cuDNN as max_total_seq_len_{q,kv}, so the dominant
// workspace terms (FP32 dQ_accum) must not grow with M. Only tiny
// per-segment buffers (TMA descriptors, batch remap) may.
TEST(CudnnFlashAttentionTest, PackedBackwardWorkspaceIndependentOfSegments) {
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor, GpuExecutor());
  if (executor->AsDnn() == nullptr) {
    GTEST_SKIP() << "DNN support is not available on this platform";
  }
  // The frontend drops max_total_seq_len on older backends.
  if (executor->AsDnn()->GetVersion().value_or(dnn::VersionInfo{0, 0, 0}) <
      dnn::VersionInfo(9, 6, 0)) {
    GTEST_SKIP() << "Requires cuDNN 9.6.0 or later.";
  }
  if (!executor->GetDeviceDescription().cuda_compute_capability().IsAtLeast(
          9, 0)) {
    GTEST_SKIP() << "Requires SM90 or newer.";
  }

  ASSERT_OK_AND_ASSIGN(int64_t ws_few,
                       PackedBackwardWorkspaceSize(executor, 2));
  ASSERT_OK_AND_ASSIGN(int64_t ws_many,
                       PackedBackwardWorkspaceSize(executor, 8));
  VLOG(1) << "workspace M=2: " << ws_few << " bytes, M=8: " << ws_many
          << " bytes";

  // Lower bound: the FP32 dQ_accum for B * S tokens. Guards against a vacuous
  // pass where the engine does not use the buffer this change shrinks.
  constexpr int64_t kDqAccumBytes =
      kBatch * kSeqLen * kNumHeads * kHeadDim * sizeof(float);
  if (ws_few < kDqAccumBytes) {
    GTEST_SKIP() << "Selected engine does not use a dQ_accum workspace.";
  }

  // Without max_total_seq_len, the workspace grows linearly with M. With it, it
  // must stay near constant. Measured on B200 with cuDNN 9.27 for this shape:
  //   without: M=2: 67,635,840 B, M=8: 270,539,904 B (ratio 4.0)
  //   with:    M=2: 34,081,792 B, M=8:  35,660,032 B (ratio 1.05)
  EXPECT_LT(static_cast<double>(ws_many), 1.1 * static_cast<double>(ws_few));
}

}  // namespace
}  // namespace stream_executor::gpu
