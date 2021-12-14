/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/lazy_op_runner.h"

namespace xla {
namespace gpu {

// Structure to describe static properties of a GPU convolution.
struct GpuConvConfig {
  // Field related to cuDNN's fused convolution are in FusionConfig &
  // FusionParams structures. The result thus is defined as:
  //   activation(conv_result_scale * conv(x, w) +
  //       side_input_scale * side_input + broadcast(bias))
  //
  // The most common fused conv is conv forward + relu/identity, for example.
  //
  // bias_buf is a single-dimensional array, with the length equal to the number
  // of output features. It'll be broadcasted to the output shape in order to be
  // added to the final results.
  //
  // side_input_buf, if valid, must have the same shape as the output buffer.
  struct FusionConfig {
    se::dnn::ActivationMode mode;
    double side_input_scale;
  };

  PrimitiveType input_type;
  PrimitiveType output_type;
  CudnnConvKind kind;
  se::dnn::AlgorithmDesc algorithm;
  double conv_result_scale;

  se::dnn::BatchDescriptor input_descriptor;
  se::dnn::FilterDescriptor filter_descriptor;
  se::dnn::BatchDescriptor output_descriptor;
  se::dnn::ConvolutionDescriptor conv_desc;

  Shape input_shape;
  Shape filter_shape;
  Shape output_shape;
  absl::optional<FusionConfig> fusion;
};

// Implementation struct exposed for debugging and log analysis.
struct GpuConvParams {
  GpuConvConfig config;
  struct FusionParams {
    se::DeviceMemoryBase bias_buf;
    se::DeviceMemoryBase side_input_buf;  // nullable
  };

  se::DeviceMemoryBase input_buf;
  se::DeviceMemoryBase filter_buf;
  se::DeviceMemoryBase output_buf;

  absl::optional<FusionParams> fusion;
};

// The XLA convolution plumbing is all dynamically-typed w.r.t. whether a
// convolution is fused (and has extra arguments) or unfused, which doesn't
// naturally play well with the typed APIs provided by StreamExecutor; rather
// than rewriting everything here, just propagate the dynamic typing to one more
// place by having either a FusedConvRunner or a ConvRunner.
class MaybeFusedConvRunner {
 public:
  MaybeFusedConvRunner() = default;

  explicit MaybeFusedConvRunner(
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedConvOp>> runner)
      : repr_(std::move(runner)) {}

  explicit MaybeFusedConvRunner(
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::ConvOp>> runner)
      : repr_(std::move(runner)) {}

  explicit MaybeFusedConvRunner(const GpuConvConfig& config)
      : MaybeFusedConvRunner(
            config.kind == CudnnConvKind::kForwardActivation
                ? MaybeFusedConvRunner(
                      std::make_unique<
                          se::dnn::LazyOpRunner<se::dnn::FusedConvOp>>(
                          config.algorithm))
                : MaybeFusedConvRunner(
                      std::make_unique<se::dnn::LazyOpRunner<se::dnn::ConvOp>>(
                          config.algorithm))) {}

  se::dnn::AlgorithmDesc ToAlgorithmDesc() const {
    return absl::visit(ToAlgorithmDescVisitor{}, repr_);
  }

  se::dnn::LazyOpRunner<se::dnn::ConvOp>* AsConvRunner() {
    CHECK(absl::holds_alternative<
          std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::ConvOp>>>(repr_));
    return absl::get<std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::ConvOp>>>(
               repr_)
        .get();
  }

  se::dnn::LazyOpRunner<se::dnn::FusedConvOp>* AsFusedConvRunner() {
    CHECK(absl::holds_alternative<
          std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedConvOp>>>(repr_));
    return absl::get<
               std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedConvOp>>>(
               repr_)
        .get();
  }

 private:
  struct ToAlgorithmDescVisitor {
    template <typename RunnerPtr>
    se::dnn::AlgorithmDesc operator()(const RunnerPtr& runner) {
      return runner->ToAlgorithmDesc();
    }

    se::dnn::AlgorithmDesc operator()(const absl::monostate&) {
      CHECK(false) << "Internal error: uninitialized runner in ToAlgorithmDesc";
    }
  };

  using Repr = absl::variant<
      absl::monostate,  // To allow GpuConvConfig default ctor
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedConvOp>>,
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::ConvOp>>>;
  Repr repr_;
};

struct RunConvOptions {
  // Nullable output-parameter pointer for profiling results.
  se::dnn::ProfileResult* profile_result = nullptr;

  // Use this runner cache (and its configured algorithm), instead of the one
  // from the instruction.
  MaybeFusedConvRunner* runner_cache;
};

// This file contains low-level routines for running cudnn convolutions.

// Calls into cudnn to run the specified convolution.
//
// We provide one overload which takes a scratch buffer, and another which takes
// an allocator which is responsible for allocating the scratch space.  In
// theory the second one shouldn't be necessary -- users of this function could
// just ask cudnn how much scratch space it needs for a particular convolution.
// But in practice, StreamExecutor does not expose such an API, and in the name
// of parsimony, perhaps it's better not to add it.  Instead, the first time you
// call a convolution, you should call the version that takes a scratch
// allocator and take note of how much memory is used.  The next time you call
// the same conv, you can provide an explicitly preallocated scratch buffer of
// that size, if you like.
Status RunGpuConv(const GpuConvConfig& conv_config,
                  absl::Span<const se::DeviceMemoryBase> operand_buffers,
                  se::DeviceMemoryBase result_buffer,
                  se::DeviceMemoryBase scratch_memory, se::Stream* stream,
                  RunConvOptions = {});

// Struct to describe properties of a convolution without being tied to specific
// IR. Will be used to help build Convolution thunks from either XLA HLO or
// LHLO GPU dialect in MLIR.
struct GpuConvDescriptor {
  CudnnConvKind kind;
  CudnnConvBackendConfig backend_config;
  Shape operand0_shape;
  Shape operand1_shape;
  Shape result_shape;
  size_t scratch_size;
  Window window;
  ConvolutionDimensionNumbers dnums;
  int64_t feature_group_count;
};

// Returns the convolution configuration given a XLA HLO instruction.
StatusOr<GpuConvConfig> GetGpuConvConfig(
    const HloCustomCallInstruction* cudnn_call);

// Returns the convolution configuration given a convolution descriptor `desc`
// and a string representation of the convolution instruction `inst_as_string`
// (for error reporting).
StatusOr<GpuConvConfig> GetGpuConvConfig(const GpuConvDescriptor& desc,
                                         absl::string_view inst_as_string);

// Implementation details exposed for debugging and log analysis.
StatusOr<GpuConvParams> GetGpuConvParams(
    const GpuConvConfig& conv_config,
    absl::Span<const se::DeviceMemoryBase> operand_buffers,
    se::DeviceMemoryBase result_buffer);

se::dnn::BatchDescriptor GetBiasDescriptor(const GpuConvConfig& config);

inline se::dnn::DataType BiasTypeForInputType(se::dnn::DataType input_type) {
  switch (input_type) {
    default:
      return input_type;
    case se::dnn::DataType::kInt8:
      return se::dnn::DataType::kFloat;
  }
}

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_
