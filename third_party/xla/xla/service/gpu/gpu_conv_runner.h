/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_
#define XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
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
    double leakyrelu_alpha = 0.0;
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
  se::dnn::BatchDescriptor bias_descriptor;

  Shape input_shape;
  Shape filter_shape;
  Shape output_shape;
  std::optional<FusionConfig> fusion;

  // String serialization of the subgraph of adjacent ops to be fused into the
  // cuDNN convolution Custom Call. Currently used for FP8 convolutions only.
  // Additional information is provided in gpu_fused_conv_rewriter.cc.
  std::string serialized_graph;
};

// Implementation struct exposed for debugging and log analysis.
struct GpuConvParams {
  const GpuConvConfig* config;  // Not owned
  struct FusionParams {
    se::DeviceMemoryBase bias_buf;
    se::DeviceMemoryBase side_input_buf;  // nullable
  };

  se::DeviceMemoryBase input_buf;
  se::DeviceMemoryBase filter_buf;
  se::DeviceMemoryBase output_buf;

  // Buffers for operands of ops to be fused into the cuDNN
  // convolution Custom Call.
  std::vector<se::DeviceMemoryBase> operand_bufs;

  // Buffers for additional outputs of ops to be fused into the cuDNN
  // convolution Custom Call.
  std::vector<se::DeviceMemoryBase> aux_bufs;

  std::optional<FusionParams> fusion;
};

// The XLA convolution plumbing is all dynamically-typed w.r.t. whether a
// convolution is fused (and has extra arguments) or unfused, which doesn't
// naturally play well with the typed APIs provided by StreamExecutor; rather
// than rewriting everything here, just propagate the dynamic typing to one more
// place by having a ConvRunner, FusedConvRunner or GraphConvRunner.
class GenericConvRunner {
 public:
  GenericConvRunner() = default;

  explicit GenericConvRunner(
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedConvOp>> runner)
      : repr_(std::move(runner)) {}

  explicit GenericConvRunner(
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::GraphConvOp>> runner)
      : repr_(std::move(runner)) {}

  explicit GenericConvRunner(
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::ConvOp>> runner)
      : repr_(std::move(runner)) {}

  explicit GenericConvRunner(const GpuConvConfig& config)
      : GenericConvRunner(FromGpuConvConfig(config)) {}

  static GenericConvRunner FromGpuConvConfig(const GpuConvConfig& config) {
    if (config.kind == CudnnConvKind::kForwardGraph) {
      return GenericConvRunner(
          std::make_unique<se::dnn::LazyOpRunner<se::dnn::GraphConvOp>>(
              config.algorithm));
    } else if (config.kind == CudnnConvKind::kForwardActivation) {
      return GenericConvRunner(
          std::make_unique<se::dnn::LazyOpRunner<se::dnn::FusedConvOp>>(
              config.algorithm));
    } else {
      return GenericConvRunner(
          std::make_unique<se::dnn::LazyOpRunner<se::dnn::ConvOp>>(
              config.algorithm));
    }
  }

  se::dnn::AlgorithmDesc ToAlgorithmDesc() const {
    return std::visit(ToAlgorithmDescVisitor{}, repr_);
  }

  se::dnn::LazyOpRunner<se::dnn::ConvOp>* AsConvRunner() {
    CHECK(std::holds_alternative<
          std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::ConvOp>>>(repr_));
    return std::get<std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::ConvOp>>>(
               repr_)
        .get();
  }

  se::dnn::LazyOpRunner<se::dnn::GraphConvOp>* AsGraphConvRunner() {
    CHECK(std::holds_alternative<
          std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::GraphConvOp>>>(repr_));
    return std::get<
               std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::GraphConvOp>>>(
               repr_)
        .get();
  }

  se::dnn::LazyOpRunner<se::dnn::FusedConvOp>* AsFusedConvRunner() {
    CHECK(std::holds_alternative<
          std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedConvOp>>>(repr_));
    return std::get<
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

    se::dnn::AlgorithmDesc operator()(const std::monostate&) {
      CHECK(false) << "Internal error: uninitialized runner in ToAlgorithmDesc";
    }
  };

  using Repr =
      std::variant<std::monostate,  // To allow GpuConvConfig default ctor
                   std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::FusedConvOp>>,
                   std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::GraphConvOp>>,
                   std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::ConvOp>>>;
  Repr repr_;
};

struct RunConvOptions {
  // Nullable output-parameter pointer for profiling results.
  se::dnn::ProfileResult* profile_result = nullptr;

  // Use this runner cache (and its configured algorithm), instead of the one
  // from the instruction.
  GenericConvRunner* runner_cache;
};

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
absl::Status RunGpuConv(const GpuConvConfig& conv_config,
                        absl::Span<const se::DeviceMemoryBase> operand_buffers,
                        absl::Span<const se::DeviceMemoryBase> result_buffers,
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
absl::StatusOr<GpuConvConfig> GetGpuConvConfig(
    const HloCustomCallInstruction* cudnn_call);

// Returns the convolution configuration given a convolution descriptor `desc`
// and a string representation of the convolution instruction `inst_as_string`
// (for error reporting).
absl::StatusOr<GpuConvConfig> GetGpuConvConfig(
    const GpuConvDescriptor& desc, absl::string_view inst_as_string);

// Implementation details exposed for debugging and log analysis.
absl::StatusOr<GpuConvParams> GetGpuConvParams(
    const GpuConvConfig& conv_config,
    absl::Span<const se::DeviceMemoryBase> operand_buffers,
    absl::Span<const se::DeviceMemoryBase> result_buffers);

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

#endif  // XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_
