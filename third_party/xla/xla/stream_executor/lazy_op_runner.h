/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------*/

#ifndef XLA_STREAM_EXECUTOR_LAZY_OP_RUNNER_H_
#define XLA_STREAM_EXECUTOR_LAZY_OP_RUNNER_H_

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace dnn {

// A lazily-initialized OpRunner from an AlgorithmDesc.
//
// This exists to hold a choice of conv algorithm for a particular config,
// initialize its OpRunner at most once, and defer that initialization until the
// config is first needed.  This allows AoT autotuning to load configurations
// for all convolutions it knows about, without doing expensive initialization
// (e.g. runtime codegen) and retaining non-negligible resources (e.g.  compiled
// kernels) for potentially irrelevant configurations.  It also enables XLA conv
// thunks to defer binding to a particular stream executor until the first run.
//
// `Op` must satisfy the following "concept":
//
// struct Op {
//   // The function type signature parameter of an OpRunner.
//   using Signature = _;
//
//   // The parameter to be used by GetOrCreateRunner.
//   struct Config;
//
//   // Use a StreamExecutor to create an OpRunner.
//   static StatusOr<OpRunner<Config>> OpRunnerFromDesc(
//       const AlgorithmDesc& desc, Config config, StreamExecutor* stream);
// };
template <typename Op>
class LazyOpRunner {
 public:
  // Construct from a pre-initialized OpRunner; all calls to GetOrCreateRunner
  // will return a pointer to exactly this runner.
  static tsl::StatusOr<std::unique_ptr<LazyOpRunner>> FromOpRunner(
      std::unique_ptr<const OpRunner<typename Op::Signature>> runner) {
    if (!runner) {
      return tsl::errors::Internal("Null runner argument to FromOpRunner");
    }
    TF_ASSIGN_OR_RETURN(auto desc, runner->ToAlgorithmDesc());
    // Private constructor cannot be called by make_unique :(
    return {std::unique_ptr<LazyOpRunner>(
        new LazyOpRunner(desc, std::move(runner)))};
  }

  // Construct from an AlgorithmDesc, with no pre-initialized OpRunner; it will
  // be created on the first call to GetOrCreateRunner.
  explicit LazyOpRunner(AlgorithmDesc desc) : LazyOpRunner(desc, nullptr) {}

  // Returns an already-initialized OpRunner if available, or creates one.
  //
  // Invariant: a particular instance of this class shall only receive calls
  // with identical `config`s and `stream_executor`s.  If the config is changed,
  // only the first config it sees will have any effect, and second and
  // subsequent configs will be ignored.  If the stream executor is changed,
  // some operations on the returned `OpRunner` using the changed stream
  // executor will be errors.
  //
  // The result is owned by LazyOpRunner.
  tsl::StatusOr<const OpRunner<typename Op::Signature>*> GetOrCreateRunner(
      typename Op::Config config, Stream* stream) {
    absl::call_once(once_flag_, [&] {
      if (runner_) return;  // runner was passed via constructor argument

      auto r = Op::RunnerFromAlgorithmDesc(desc_, std::move(config), stream);
      if (!r.ok()) {
        error_ = std::move(r).status();
      } else {
        runner_ = std::move(r).value();
      }
    });

    if (!error_.ok()) return error_;
    return runner_.get();
  }

  // Get the contained runner with the invariant that it's already initialized.
  tsl::StatusOr<const OpRunner<typename Op::Signature>*> GetRunner() {
    if (auto* runner = runner_ptr_.load(std::memory_order_acquire)) {
      return runner;
    }
    return tsl::errors::Internal("LazyOpRunner::GetRunner: not initialized");
  }

  bool operator==(const LazyOpRunner& other) const {
    return desc_ == other.desc_;
  }

  std::string ToString() const { return desc_.ToString(); }

  const AlgorithmDesc& ToAlgorithmDesc() const { return desc_; }

 private:
  LazyOpRunner(AlgorithmDesc desc,
               std::unique_ptr<const OpRunner<typename Op::Signature>> runner)
      : desc_(std::move(desc)),
        error_(tsl::OkStatus()),
        runner_(std::move(runner)),
        runner_ptr_(runner_.get()) {}

  AlgorithmDesc desc_;

  // We use absl::call_once to lazily initialize `runner_` (or `error_`).
  absl::once_flag once_flag_;
  tsl::Status error_;  // holds error if runner can't be initialized
  std::unique_ptr<const OpRunner<typename Op::Signature>> runner_;

  // Once we initialize `runner_` we publish a pointer through atomic so that
  // `GetRunner` can read it without data races with initialization.
  std::atomic<const OpRunner<typename Op::Signature>*> runner_ptr_;
};

// Implementation of the concept required by LazyOpRunner, for ConvRunner.
struct ConvOp {
  using Signature = ConvSignature;

  struct Config {
    ConvolutionKind kind;
    DataType input_type, output_type;
    const BatchDescriptor& input_descriptor;
    const FilterDescriptor& filter_descriptor;
    const BatchDescriptor& output_descriptor;
    const ConvolutionDescriptor& convolution_descriptor;
  };

  static tsl::StatusOr<std::unique_ptr<const OpRunner<ConvSignature>>>
  RunnerFromAlgorithmDesc(const AlgorithmDesc& desc, Config config,
                          Stream* stream) {
    return stream->ConvolveRunnerFromDesc(
        desc, config.kind, config.input_type, config.output_type,
        config.input_descriptor, config.filter_descriptor,
        config.output_descriptor, config.convolution_descriptor);
  }
};

// Implementation of the concept required by LazyOpRunner, for
// GraphConvolveRunner.
struct GraphConvOp {
  using Signature = GraphConvSignature;

  struct Config {
    ConvolutionKind kind;
    DataType input_type, output_type;
    const BatchDescriptor& input_descriptor;
    const FilterDescriptor& filter_descriptor;
    const BatchDescriptor& output_descriptor;
    const ConvolutionDescriptor& convolution_descriptor;
    std::string serialized_graph;
  };

  static tsl::StatusOr<std::unique_ptr<const OpRunner<Signature>>>
  RunnerFromAlgorithmDesc(const AlgorithmDesc& desc, Config config,
                          Stream* stream) {
    return stream->GraphConvolveRunnerFromDesc(
        desc, config.kind, config.input_type, config.output_type,
        config.input_descriptor, config.filter_descriptor,
        config.output_descriptor, config.convolution_descriptor,
        config.serialized_graph);
  }
};

// Implementation of the concept required by LazyOpRunner, for LazyConvRunner.
struct FusedConvOp {
  using Signature = FusedConvSignature;

  struct Config {
    ConvolutionKind kind;
    DataType input_type, bias_type, output_type;
    double conv_scale, side_input_scale, leakyrelu_alpha;
    const BatchDescriptor& input_descriptor;
    const FilterDescriptor& filter_descriptor;
    const BatchDescriptor& bias_descriptor;
    const BatchDescriptor& output_descriptor;
    const ConvolutionDescriptor& convolution_descriptor;
    ActivationMode activation_mode;
  };

  static tsl::StatusOr<std::unique_ptr<const OpRunner<FusedConvSignature>>>
  RunnerFromAlgorithmDesc(const AlgorithmDesc& desc, Config config,
                          Stream* stream) {
    return stream->FusedConvolveRunnerFromDesc(
        desc, config.kind, config.input_type, config.bias_type,
        config.output_type, config.conv_scale, config.side_input_scale,
        config.leakyrelu_alpha, config.input_descriptor,
        config.filter_descriptor, config.bias_descriptor,
        config.output_descriptor, config.convolution_descriptor,
        config.activation_mode);
  }
};

// Implementation of the concept required by LazyOpRunner, for NormRunner.
struct NormOp {
  using Signature = NormSignature;

  struct Config {
    double epsilon;
    const TensorDescriptor& input_descriptor;
    const TensorDescriptor& scale_descriptor;
    const TensorDescriptor& bias_descriptor;
    const TensorDescriptor& output_descriptor;
    std::optional<dnn::TensorDescriptor> expectation_descriptor;
    std::optional<dnn::TensorDescriptor> norm_factor_descriptor;
  };

  static tsl::StatusOr<std::unique_ptr<const OpRunner<Signature>>>
  RunnerFromAlgorithmDesc(const AlgorithmDesc& desc, Config config,
                          Stream* stream) {
    return stream->NormRunnerFromDesc(
        desc, config.epsilon, config.input_descriptor, config.scale_descriptor,
        config.bias_descriptor, config.output_descriptor,
        config.expectation_descriptor, config.norm_factor_descriptor);
  }
};

// Implementation of the concept required by LazyOpRunner, for FusedMatmul.
struct FusedMatmulOp {
  using Signature = FusedMatmulSignature;

  // Config is mainly used in RunnerFromAlgorithmDesc() to lazily create the
  // runner. At this moment we only get existing runners and don't implement
  // this feature.
  struct Config {};

  static tsl::StatusOr<std::unique_ptr<const OpRunner<Signature>>>
  RunnerFromAlgorithmDesc(const AlgorithmDesc& desc, Config config,
                          Stream* stream) {
    return tsl::errors::Unimplemented("Unimplemented");
  }
};

struct FusedMHAOp {
  using Signature = FusedMHASignature;
  struct Config {
    FusedMHAKind kind;
    double scale;
    const MatmulTensorDescriptor& bmm1_lhs_descriptor;
    const MatmulTensorDescriptor& bmm1_rhs_descriptor;
    const MatmulTensorDescriptor& bmm2_rhs_descriptor;
    const MatmulTensorDescriptor& intermediate_bmm2_lhs_descriptor;
    const TensorDescriptor& output_descriptor;
    std::optional<TensorDescriptor> bias_descriptor;
    std::optional<TensorDescriptor> mask_descriptor;
    std::optional<TensorDescriptor> activation_descriptor;
    std::optional<double> dropout_rate;
    std::optional<int64_t> seed;
    bool is_flash_attention;
    bool is_causal_mask;
  };

  static tsl::StatusOr<std::unique_ptr<const OpRunner<FusedMHASignature>>>
  RunnerFromAlgorithmDesc(const AlgorithmDesc& desc, Config config,
                          Stream* stream) {
    return stream->FusedMHARunnerFromDesc(
        desc, config.kind, config.bmm1_lhs_descriptor,
        config.bmm1_rhs_descriptor, config.bmm2_rhs_descriptor,
        config.intermediate_bmm2_lhs_descriptor, config.output_descriptor,
        config.activation_descriptor, config.mask_descriptor,
        config.bias_descriptor, config.scale, config.dropout_rate, config.seed,
        config.is_flash_attention, config.is_causal_mask);
  }
};

struct FusedMHABackwardOp {
  using Signature = FusedMHABackwardSignature;

  struct Config {
    FusedMHAKind kind;
    double scale;
    const MatmulTensorDescriptor& bmm1_grad_gemm1_rhs_descriptor;
    const MatmulTensorDescriptor& bmm1_grad_gemm2_rhs_descriptor;
    const MatmulTensorDescriptor& bmm2_grad_gemm1_lhs_descriptor;
    const MatmulTensorDescriptor& bmm2_grad_gemm2_rhs_descriptor;
    const MatmulTensorDescriptor& d_output_descriptor;
    const TensorDescriptor& d_bmm1_lhs_descriptor;
    const TensorDescriptor& d_bmm1_rhs_descriptor;
    const TensorDescriptor& d_bmm2_rhs_descriptor;
    std::optional<TensorDescriptor> d_s_descriptor;
    std::optional<TensorDescriptor> mask_descriptor;
    std::optional<TensorDescriptor> d_bias_descriptor;
    std::optional<TensorDescriptor> fwd_output_descriptor;
    std::optional<TensorDescriptor> bias_descriptor;
    std::optional<double> dropout_rate;
    std::optional<int64_t> seed;
    bool is_flash_attention;
    bool is_causal_mask;
  };

  static tsl::StatusOr<
      std::unique_ptr<const OpRunner<FusedMHABackwardSignature>>>
  RunnerFromAlgorithmDesc(const AlgorithmDesc& desc, Config config,
                          Stream* stream) {
    return stream->FusedMHABackwardRunnerFromDesc(
        desc, config.kind, config.bmm1_grad_gemm1_rhs_descriptor,
        config.bmm1_grad_gemm2_rhs_descriptor,
        config.bmm2_grad_gemm1_lhs_descriptor,
        config.bmm2_grad_gemm2_rhs_descriptor, config.d_output_descriptor,
        config.d_bmm1_lhs_descriptor, config.d_bmm1_rhs_descriptor,
        config.d_bmm2_rhs_descriptor, config.d_s_descriptor,
        config.mask_descriptor, config.d_bias_descriptor,
        config.fwd_output_descriptor, config.bias_descriptor, config.scale,
        config.dropout_rate, config.seed, config.is_flash_attention,
        config.is_causal_mask);
  }
};

}  // namespace dnn
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_LAZY_OP_RUNNER_H_
