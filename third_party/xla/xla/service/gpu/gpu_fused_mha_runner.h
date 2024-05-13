/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_FUSED_MHA_RUNNER_H_
#define XLA_SERVICE_GPU_GPU_FUSED_MHA_RUNNER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

inline absl::StatusOr<xla::gpu::CudnnfMHAMaskKind> AsCudnnFmhaMaskKind(
    xla::gpu::CudnnfMHABackendConfig_MaskType mask_type) {
  switch (mask_type) {
    case xla::gpu::CudnnfMHABackendConfig::NO_MASK:
      return xla::gpu::CudnnfMHAMaskKind::kNoMask;
    case xla::gpu::CudnnfMHABackendConfig::PADDING:
      return xla::gpu::CudnnfMHAMaskKind::kPadding;
    case xla::gpu::CudnnfMHABackendConfig::CAUSAL:
      return xla::gpu::CudnnfMHAMaskKind::kCausal;
    case xla::gpu::CudnnfMHABackendConfig::PADDING_CAUSAL:
      return xla::gpu::CudnnfMHAMaskKind::kPaddingCausal;
    case xla::gpu::CudnnfMHABackendConfig::ALIBI:
      return xla::gpu::CudnnfMHAMaskKind::kAlibi;
    default:
      return xla::Internal("Unknown fmha mask kind.");
  }
}

// This is an interim structure to hold the parameters to construct a
// GpufMHAConfig.
// Struct to describe properties of a FMHA without being tied to specific
// IR. Will be used to help build FMHA thunks from either XLA HLO or
// LHLO GPU dialect in MLIR.
struct GpufMHADescriptor {
  CudnnfMHAKind kind;
  CudnnfMHABackendConfig backend_config;
  CudnnfMHAMaskKind mask_type;
  Shape lhs_bmm1_shape;
  Shape rhs_bmm1_shape;
  Shape rhs_bmm2_shape;
  Shape intermediate_lhs_bmm2_shape;
  // This will contain both output shape and activation shape
  absl::InlinedVector<Shape, 2> output_shapes;
  DotDimensionNumbers bmm1_dnums;
  DotDimensionNumbers bmm2_dnums;

  std::optional<Shape> mask_shape;
  std::optional<Shape> bias_shape;
};

struct GpufMHABackwardDescriptor {
  CudnnfMHAKind kind;
  CudnnfMHABackendConfig backend_config;
  CudnnfMHAMaskKind mask_type;
  Shape bmm1_grad_gemm1_rhs_shape;
  Shape bmm1_grad_gemm2_rhs_shape;
  Shape bmm2_grad_gemm1_lhs_shape;
  Shape bmm2_grad_gemm2_rhs_shape;
  Shape d_output_shape;
  Shape d_bmm1_lhs_shape;
  Shape d_bmm1_rhs_shape;
  Shape d_bmm2_rhs_shape;
  DotDimensionNumbers bmm1_grad_gemm1_dnums;
  DotDimensionNumbers bmm1_grad_gemm2_dnums;
  DotDimensionNumbers bmm2_grad_gemm1_dnums;
  DotDimensionNumbers bmm2_grad_gemm2_dnums;

  std::optional<Shape> d_s_shape;
  std::optional<Shape> fwd_output_shape;
  std::optional<Shape> mask_shape;
  std::optional<Shape> d_bias_shape;
  std::optional<Shape> bias_shape;
};
// Structure to describe static properties of a GPU fused Multi-Headed
// Attention.
struct GpufMHAConfig {
  static absl::StatusOr<GpufMHAConfig> For(const GpufMHADescriptor& fmha_desc);
  PrimitiveType
      input_type;  // Capture the primitive type of one of the inputs of BMM1
  PrimitiveType output_type;
  CudnnfMHAKind kind;
  std::optional<double> fmha_scale;
  std::optional<double> dropout_rate;
  std::optional<int64_t> seed;

  se::dnn::AlgorithmDesc algorithm;
  CudnnfMHAMaskKind mask_type;
  // bias -> [1, num_attn_heads, q_seq_len, kv_seq_len]
  // mask -> [batch_size, 1, q_seq_len, kv_seq_len]
  se::dnn::MatmulTensorDescriptor lhs_bmm1;
  se::dnn::MatmulTensorDescriptor rhs_bmm1;
  se::dnn::MatmulTensorDescriptor rhs_bmm2;
  se::dnn::MatmulTensorDescriptor intermediate_lhs_bmm2;
  se::dnn::TensorDescriptor output;

  std::optional<se::dnn::TensorDescriptor> activation;
  std::optional<se::dnn::TensorDescriptor> mask;
  std::optional<se::dnn::TensorDescriptor> bias;
};

// Structure to describe static properties of a GPU fused Multi-Headed
// Attention backward.
struct GpufMHABackwardConfig {
  static absl::StatusOr<GpufMHABackwardConfig> For(
      const GpufMHABackwardDescriptor& fmha_desc);
  PrimitiveType
      input_type;  // Capture the primitive type of one of the inputs of BMM1
  PrimitiveType output_type;
  CudnnfMHAKind kind;
  std::optional<double> fmha_scale;
  std::optional<double> dropout_rate;
  std::optional<int64_t> seed;

  se::dnn::AlgorithmDesc algorithm;
  CudnnfMHAMaskKind mask_type;
  // mask -> [batch_size, 1, q_seq_len, kv_seq_len]
  // d_bias -> [1, num_heads, q_seq_len, kv_seq_len]
  se::dnn::MatmulTensorDescriptor bmm1_grad_gemm1_rhs;
  se::dnn::MatmulTensorDescriptor bmm1_grad_gemm2_rhs;
  se::dnn::MatmulTensorDescriptor bmm2_grad_gemm1_lhs;
  se::dnn::MatmulTensorDescriptor bmm2_grad_gemm2_rhs;
  se::dnn::MatmulTensorDescriptor d_output;
  se::dnn::TensorDescriptor d_bmm1_lhs;
  se::dnn::TensorDescriptor d_bmm1_rhs;
  se::dnn::TensorDescriptor d_bmm2_rhs;
  std::optional<se::dnn::TensorDescriptor> d_s;
  std::optional<se::dnn::TensorDescriptor> mask;
  std::optional<se::dnn::TensorDescriptor> d_bias;
  std::optional<se::dnn::TensorDescriptor> fwd_output;
  std::optional<se::dnn::TensorDescriptor> bias;
};

// Implementation struct exposed for debugging and log analysis.
struct GpufMHAParams {
  static absl::StatusOr<GpufMHAParams> For(
      const GpufMHAConfig& config, se::DeviceMemoryBase lhs_bmm1_buffer,
      se::DeviceMemoryBase rhs_bmm1_buffer,
      se::DeviceMemoryBase rhs_bmm2_buffer, se::DeviceMemoryBase output_buffer,
      std::optional<se::DeviceMemoryBase> bias_buffer,
      std::optional<se::DeviceMemoryBase> activation_buffer,
      std::optional<se::DeviceMemoryBase> seqlen_q_buffer,
      std::optional<se::DeviceMemoryBase> seqlen_k_buffer);

  const GpufMHAConfig* config;  // Not owned
  se::DeviceMemoryBase lhs_bmm1_buffer;
  se::DeviceMemoryBase rhs_bmm1_buffer;
  se::DeviceMemoryBase rhs_bmm2_buffer;
  se::DeviceMemoryBase output_buffer;
  std::optional<se::DeviceMemoryBase> activation_buffer;
  std::optional<se::DeviceMemoryBase> bias_buffer;
  std::optional<se::DeviceMemoryBase> seqlen_q_buffer;
  std::optional<se::DeviceMemoryBase> seqlen_k_buffer;
};

struct GpufMHABackwardParams {
  static absl::StatusOr<GpufMHABackwardParams> For(
      const GpufMHABackwardConfig& config,
      se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer,
      se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer,
      se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer,
      se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer,
      se::DeviceMemoryBase d_output_buffer,
      se::DeviceMemoryBase d_bmm1_lhs_buffer,
      se::DeviceMemoryBase d_bmm1_rhs_buffer,
      se::DeviceMemoryBase d_bmm2_rhs_buffer,
      std::optional<se::DeviceMemoryBase> d_s_buffer,
      std::optional<se::DeviceMemoryBase> d_bias_buffer,
      std::optional<se::DeviceMemoryBase> fwd_output_buffer,
      std::optional<se::DeviceMemoryBase> bias_buffer,
      std::optional<se::DeviceMemoryBase> seqlen_q_buffer,
      std::optional<se::DeviceMemoryBase> seqlen_k_buffer);

  const GpufMHABackwardConfig* config;  // Not owned
  se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer;
  se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer;
  se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer;
  se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer;
  se::DeviceMemoryBase d_output_buffer;
  se::DeviceMemoryBase d_bmm1_lhs_buffer;
  se::DeviceMemoryBase d_bmm1_rhs_buffer;
  se::DeviceMemoryBase d_bmm2_rhs_buffer;
  std::optional<se::DeviceMemoryBase> d_s_buffer;
  std::optional<se::DeviceMemoryBase> d_bias_buffer;
  std::optional<se::DeviceMemoryBase> fwd_output_buffer;
  std::optional<se::DeviceMemoryBase> bias_buffer;
  std::optional<se::DeviceMemoryBase> seqlen_q_buffer;
  std::optional<se::DeviceMemoryBase> seqlen_k_buffer;
};

class FusedMultiHeadedAttentionRunner {
 public:
  using Repr =
      std::variant<std::monostate,  // To allow XXX default ctor
                   std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHAOp>>>;

  FusedMultiHeadedAttentionRunner() = default;

  explicit FusedMultiHeadedAttentionRunner(
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHAOp>> runner)
      : repr_(std::move(runner)) {}

  explicit FusedMultiHeadedAttentionRunner(Repr runner)
      : repr_(std::move(runner)) {}

  explicit FusedMultiHeadedAttentionRunner(const GpufMHAConfig& config)
      : FusedMultiHeadedAttentionRunner(CreateRunner(config)) {
    if (std::holds_alternative<std::monostate>(repr_)) {
      CHECK(false) << "Cannot construct FusedMultiHeadedAttentionRunner with "
                      "std::monostate";
    }
  }

  se::dnn::AlgorithmDesc ToAlgorithmDesc() const {
    return std::visit(ToAlgorithmDescVisitor{}, repr_);
  }

  se::dnn::LazyOpRunner<se::dnn::FusedMHAOp>* AsFusedMHARunner() {
    CHECK(std::holds_alternative<
          std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHAOp>>>(repr_));
    return std::get<
               std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHAOp>>>(
               repr_)
        .get();
  }

 private:
  //  The CreateRunner function is defined as static because it
  //  doesn't need access to any non-static member variables of the
  //  FusedMultiHeadedAttentionRunner class. Defining it static makes it easy to
  //  use and makes it clear that it is a utility function that doesn't rely on
  //  the state of any specific instance of the class.
  static Repr CreateRunner(const GpufMHAConfig& config) {
    switch (config.kind) {
      case CudnnfMHAKind::kSoftmaxDropout:
      case CudnnfMHAKind::kSoftmax:
      case CudnnfMHAKind::kScaleBiasSoftmax:
      case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
        return std::make_unique<se::dnn::LazyOpRunner<se::dnn::FusedMHAOp>>(
            config.algorithm);
      default:
        LOG(FATAL) << "Internal error: unsupported CUDNN MHA kind in "
                      "FusedMultiHeadedAttentionRunner";
    }
  }

  struct ToAlgorithmDescVisitor {
    template <typename RunnerPtr>
    se::dnn::AlgorithmDesc operator()(const RunnerPtr& runner) {
      return runner->ToAlgorithmDesc();
    }

    se::dnn::AlgorithmDesc operator()(const std::monostate&) {
      CHECK(false) << "Internal error: uninitialized runner in ToAlgorithmDesc";
    }
  };

  Repr repr_;
};

class FusedMultiHeadedAttentionBackwardRunner {
 public:
  using Repr = std::variant<
      std::monostate,  // To allow XXX default ctor
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardOp>>>;

  FusedMultiHeadedAttentionBackwardRunner() = default;

  explicit FusedMultiHeadedAttentionBackwardRunner(
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardOp>>
          runner)
      : repr_(std::move(runner)) {}

  explicit FusedMultiHeadedAttentionBackwardRunner(Repr runner)
      : repr_(std::move(runner)) {}

  explicit FusedMultiHeadedAttentionBackwardRunner(
      const GpufMHABackwardConfig& config)
      : FusedMultiHeadedAttentionBackwardRunner(CreateRunner(config)) {
    if (std::holds_alternative<std::monostate>(repr_)) {
      CHECK(false)
          << "Cannot construct FusedMultiHeadedAttentionBackwardRunner with "
             "std::monostate";
    }
  }

  se::dnn::AlgorithmDesc ToAlgorithmDesc() const {
    return std::visit(ToAlgorithmDescVisitor{}, repr_);
  }

  se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardOp>*
  AsFusedMHABackwardRunner() {
    CHECK(std::holds_alternative<
          std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardOp>>>(
        repr_));
    return std::get<std::unique_ptr<
        se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardOp>>>(repr_)
        .get();
  }

 private:
  //  The CreateRunner function is defined as static because it
  //  doesn't need access to any non-static member variables of the
  //  FusedMultiHeadedAttentionBackwardRunner class. Defining it static makes it
  //  easy to use and makes it clear that it is a utility function that doesn't
  //  rely on the state of any specific instance of the class.
  static Repr CreateRunner(const GpufMHABackwardConfig& config) {
    switch (config.kind) {
      case CudnnfMHAKind::kBackwardSoftmaxDropout:
      case CudnnfMHAKind::kBackwardSoftmax:
      case CudnnfMHAKind::kBackwardScaleBiasSoftmax:
      case CudnnfMHAKind::kBackwardScaleBiasSoftmaxDropout:
        return std::make_unique<
            se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardOp>>(
            config.algorithm);
      default:
        LOG(FATAL) << "Internal error: unsupported CUDNN MHA kind in "
                      "FusedMultiHeadedAttentionBackwardRunner";
    }
  }

  struct ToAlgorithmDescVisitor {
    template <typename RunnerPtr>
    se::dnn::AlgorithmDesc operator()(const RunnerPtr& runner) {
      return runner->ToAlgorithmDesc();
    }

    se::dnn::AlgorithmDesc operator()(const std::monostate&) {
      CHECK(false) << "Internal error: uninitialized runner in ToAlgorithmDesc";
    }
  };

  Repr repr_;
};

struct RunFusedMHAOptions {
  // Nullable output-parameter pointer for profiling results.
  // Profile results remain unused for now since cuDNN FMHA has only one
  // algorithm for now.
  se::dnn::ProfileResult* profile_result = nullptr;

  // Use this runner cache (and its configured algorithm), instead of the one
  // from the instruction.
  FusedMultiHeadedAttentionRunner* runner_cache;
};

struct RunFusedMHABackwardOptions {
  // Nullable output-parameter pointer for profiling results.
  // Profile results remain unused for now since cuDNN FMHA has only one
  // algorithm for now.
  se::dnn::ProfileResult* profile_result = nullptr;

  // Use this runner cache (and its configured algorithm), instead of the one
  // from the instruction.
  FusedMultiHeadedAttentionBackwardRunner* runner_cache;
};

absl::Status RunGpuFMHA(const GpufMHAConfig& fmha_config,
                        se::DeviceMemoryBase lhs_bmm1_buffer,
                        se::DeviceMemoryBase rhs_bmm1_buffer,
                        se::DeviceMemoryBase rhs_bmm2_buffer,
                        se::DeviceMemoryBase output_buffer,
                        se::DeviceMemoryBase scratch_buffer,
                        std::optional<se::DeviceMemoryBase> bias_buffer,
                        std::optional<se::DeviceMemoryBase> activation_buffer,
                        std::optional<se::DeviceMemoryBase> seqlen_q_buffer,
                        std::optional<se::DeviceMemoryBase> seqlen_k_buffer,
                        se::Stream* stream, RunFusedMHAOptions = {});

absl::Status RunGpuFMHABackward(
    const GpufMHABackwardConfig& fmha_config,
    se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer,
    se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase d_output_buffer, se::DeviceMemoryBase scratch_buffer,
    se::DeviceMemoryBase d_bmm1_lhs_buffer,
    se::DeviceMemoryBase d_bmm1_rhs_buffer,
    se::DeviceMemoryBase d_bmm2_rhs_buffer,
    std::optional<se::DeviceMemoryBase> d_s_buffer,
    std::optional<se::DeviceMemoryBase> d_bias_buffer,
    std::optional<se::DeviceMemoryBase> fwd_output_buffer,
    std::optional<se::DeviceMemoryBase> bias_buffer,
    std::optional<se::DeviceMemoryBase> seqlen_q_buffer,
    std::optional<se::DeviceMemoryBase> seqlen_k_buffer, se::Stream* stream,
    RunFusedMHABackwardOptions = {});

std::string ToString(const GpufMHAConfig& config);

}  // namespace gpu
}  // namespace xla
#endif  // XLA_SERVICE_GPU_GPU_FUSED_MHA_RUNNER_H_
