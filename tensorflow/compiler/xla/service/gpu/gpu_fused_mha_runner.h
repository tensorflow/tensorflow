/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSED_MHA_RUNNER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSED_MHA_RUNNER_H_

#include <optional>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/stream_executor/lazy_op_runner.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// This is an interim structure to hold the parameters to construct a
// GpufMHAConfig.
// Struct to describe properties of a FMHA without being tied to specific
// IR. Will be used to help build FMHA thunks from either XLA HLO or
// LHLO GPU dialect in MLIR.
struct GpufMHADescriptor {
  CudnnfMHAKind kind;
  FusedMHABackendConfig backend_config;
  Shape lhs_bmm1_shape;
  Shape rhs_bmm1_shape;
  Shape rhs_bmm2_shape;
  Shape intermediate_lhs_bmm2_shape;
  Shape output_shape;
  DotDimensionNumbers bmm1_dnums;
  DotDimensionNumbers bmm2_dnums;

  std::optional<Shape> mask_shape;
  std::optional<Shape> bias_shape;
};

// Structure to describe static properties of a GPU fused Multi-Headed
// Attention.
struct GpufMHAConfig {
  static StatusOr<GpufMHAConfig> For(const GpufMHADescriptor& fmha_desc);
  PrimitiveType
      input_type;  // Capture the primitive type of one of the inputs of BMM1
  PrimitiveType output_type;
  CudnnfMHAKind kind;
  std::optional<double> fmha_scale;
  std::optional<double> dropout_rate;

  se::dnn::AlgorithmDesc algorithm;

  // bias -> [1, num_attn_heads, q_seq_len, kv_seq_len]
  // mask -> [batch_size, 1, q_seq_len, kv_seq_len]
  se::dnn::MatmulTensorDescriptor lhs_bmm1;
  se::dnn::MatmulTensorDescriptor rhs_bmm1;
  se::dnn::MatmulTensorDescriptor rhs_bmm2;
  se::dnn::MatmulTensorDescriptor intermediate_lhs_bmm2;
  se::dnn::TensorDescriptor output;

  std::optional<se::dnn::TensorDescriptor> mask;
  std::optional<se::dnn::TensorDescriptor> bias;
};

// Implementation struct exposed for debugging and log analysis.
struct GpufMHAParams {
  static StatusOr<GpufMHAParams> For(const GpufMHAConfig& config,
                                     se::DeviceMemoryBase lhs_bmm1_buffer,
                                     se::DeviceMemoryBase rhs_bmm1_buffer,
                                     se::DeviceMemoryBase rhs_bmm2_buffer,
                                     se::DeviceMemoryBase output_buffer,
                                     se::DeviceMemoryBase mask_buffer,
                                     se::DeviceMemoryBase bias_buffer);

  const GpufMHAConfig* config;  // Not owned
  se::DeviceMemoryBase lhs_bmm1_buffer;
  se::DeviceMemoryBase rhs_bmm1_buffer;
  se::DeviceMemoryBase rhs_bmm2_buffer;
  se::DeviceMemoryBase output_buffer;
  std::optional<se::DeviceMemoryBase> mask_buffer;
  std::optional<se::DeviceMemoryBase> bias_buffer;
};

class FusedMultiHeadedAttentionRunner {
 public:
  FusedMultiHeadedAttentionRunner() = default;

  explicit FusedMultiHeadedAttentionRunner(
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHASimpleOp>> runner)
      : repr_(std::move(runner)) {}

  explicit FusedMultiHeadedAttentionRunner(
      std::unique_ptr<
          se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleMaskSoftmaxOp>>
          runner)
      : repr_(std::move(runner)) {}

  explicit FusedMultiHeadedAttentionRunner(
      std::unique_ptr<
          se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleBiasMaskSoftmaxOp>>
          runner)
      : repr_(std::move(runner)) {}

  explicit FusedMultiHeadedAttentionRunner(const GpufMHAConfig& config)
      : FusedMultiHeadedAttentionRunner(
            config.kind == CudnnfMHAKind::kBmmBmm ||
                    config.kind == CudnnfMHAKind::kSoftmaxDropout
                ? FusedMultiHeadedAttentionRunner(
                      std::make_unique<
                          se::dnn::LazyOpRunner<se::dnn::FusedMHASimpleOp>>(
                          config.algorithm))
                : config.kind == CudnnfMHAKind::kScaleMaskSoftmax ||
                          config.kind == CudnnfMHAKind::kScaleMaskSoftmaxDropout
                      ? FusedMultiHeadedAttentionRunner(
                            std::make_unique<se::dnn::LazyOpRunner<
                                se::dnn::FusedMHAScaleMaskSoftmaxOp>>(
                                config.algorithm))
                      : FusedMultiHeadedAttentionRunner(
                            std::make_unique<se::dnn::LazyOpRunner<
                                se::dnn::FusedMHAScaleBiasMaskSoftmaxOp>>(
                                config.algorithm))) {}

  se::dnn::AlgorithmDesc ToAlgorithmDesc() const {
    return std::visit(ToAlgorithmDescVisitor{}, repr_);
  }

  se::dnn::LazyOpRunner<se::dnn::FusedMHASimpleOp>* AsFusedMHASimpleRunner() {
    CHECK(std::holds_alternative<
          std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHASimpleOp>>>(
        repr_));
    return std::
        get<std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHASimpleOp>>>(
               repr_)
            .get();
  }

  se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleMaskSoftmaxOp>*
  AsFusedMHAMaskRunner() {
    CHECK(std::holds_alternative<std::unique_ptr<
              se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleMaskSoftmaxOp>>>(
        repr_));
    return std::get<std::unique_ptr<
        se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleMaskSoftmaxOp>>>(repr_)
        .get();
  }

  se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleBiasMaskSoftmaxOp>*
  AsFusedMHABiasMaskRunner() {
    CHECK(std::holds_alternative<std::unique_ptr<
              se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleBiasMaskSoftmaxOp>>>(
        repr_));
    return std::get<std::unique_ptr<
        se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleBiasMaskSoftmaxOp>>>(repr_)
        .get();
  }

 private:
  struct ToAlgorithmDescVisitor {
    template <typename RunnerPtr>
    se::dnn::AlgorithmDesc operator()(const RunnerPtr& runner) {
      return runner->ToAlgorithmDesc();
    }

    se::dnn::AlgorithmDesc operator()(const std::monostate&) {
      CHECK(false) << "Internal error: uninitialized runner in ToAlgorithmDesc";
    }
  };
  using Repr = std::variant<
      std::monostate,  // To allow XXX default ctor
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedMHASimpleOp>>,
      std::unique_ptr<
          se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleMaskSoftmaxOp>>,
      std::unique_ptr<
          se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleBiasMaskSoftmaxOp>>>;
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

Status RunGpuFMHA(
    const GpufMHAConfig& fmha_config, se::DeviceMemoryBase lhs_bmm1_buffer,
    se::DeviceMemoryBase rhs_bmm1_buffer, se::DeviceMemoryBase rhs_bmm2_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase scratch_buffer,
    se::DeviceMemoryBase mask_buffer, se::DeviceMemoryBase bias_buffer,
    se::Stream* stream, RunFusedMHAOptions = {});

}  // namespace gpu
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSED_MHA_RUNNER_H_