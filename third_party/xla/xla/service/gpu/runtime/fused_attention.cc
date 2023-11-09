/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.1
==============================================================================*/

#include "xla/service/gpu/runtime/fused_attention.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/Sequence.h"
#include "xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/executable.h"
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/service/gpu/gpu_fused_mha_runner.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/xla.pb.h"

namespace xla {

using xla::runtime::CustomCall;
using xla::runtime::EnumAttrEncoding;
using xla::runtime::FlatMemrefView;
using xla::runtime::State;
using xla::runtime::StridedMemrefView;
using xla::runtime::Tagged;

namespace lmhlo_gpu = ::mlir::lmhlo_gpu;
namespace gpu {
//===----------------------------------------------------------------------===//
// Structs for encoding fused attention attributes defined in LMHLO dialect.
//===----------------------------------------------------------------------===//
struct AlgorithmConfig {
  int64_t algorithm;
  absl::Span<const int64_t> knob_ids;
  absl::Span<const int64_t> knob_values;
  int64_t workspace_size;
};

}  // namespace gpu

//===----------------------------------------------------------------------===//
// Register fused attention attributes decoding with the Xla runtime.
//===----------------------------------------------------------------------===//
namespace runtime {
XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(xla::gpu::CudnnfMHAKind);

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    xla::gpu::AlgorithmConfig,  //
    AggregateMember<int64_t>("algorithm"),
    AggregateMember<absl::Span<const int64_t>>("knob_ids"),
    AggregateMember<absl::Span<const int64_t>>("knob_values"),
    AggregateMember<int64_t>("workspace_size"));

}  // namespace runtime

//===----------------------------------------------------------------------===//
// Type names for encoded attributes.
//===----------------------------------------------------------------------===//

namespace gpu {

// Register type names for fused attention attributes defined by LMHLO dialect.
void RegisterFusedAttentionTypeIdNames(runtime::TypeIDNameRegistry& registry) {
  registry.Register<Tagged<AlgorithmConfig>>("__type_id_algorithm_config");
  registry.Register<Tagged<xla::gpu::CudnnfMHAKind>>(
      "__type_id_xla_gpu_cudnn_fmha_kind");
}

static auto EncodeFusedAttentionDAGSignature(
    lmhlo_gpu::FusedMhaDagSignature signature) {
  switch (signature) {
    case mlir::lmhlo_gpu::FusedMhaDagSignature::Default:
      return xla::gpu::CudnnfMHAKind::kBmmBmm;
    case mlir::lmhlo_gpu::FusedMhaDagSignature::ScaleBiasMaskSoftmax:
      return xla::gpu::CudnnfMHAKind::kScaleBiasMaskSoftmax;
    case mlir::lmhlo_gpu::FusedMhaDagSignature::ScaleBiasMaskSoftmaxDropout:
      return xla::gpu::CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout;
    case mlir::lmhlo_gpu::FusedMhaDagSignature::ScaleMaskSoftmax:
      return xla::gpu::CudnnfMHAKind::kScaleMaskSoftmax;
    case mlir::lmhlo_gpu::FusedMhaDagSignature::ScaleMaskSoftmaxDropout:
      return xla::gpu::CudnnfMHAKind::kScaleMaskSoftmaxDropout;
    case mlir::lmhlo_gpu::FusedMhaDagSignature::SoftmaxDropout:
      return xla::gpu::CudnnfMHAKind::kSoftmaxDropout;
    case mlir::lmhlo_gpu::FusedMhaDagSignature::Softmax:
      return xla::gpu::CudnnfMHAKind::kSoftmax;
    case mlir::lmhlo_gpu::FusedMhaDagSignature::ScaleBiasSoftmax:
      return xla::gpu::CudnnfMHAKind::kScaleBiasSoftmax;
    case mlir::lmhlo_gpu::FusedMhaDagSignature::ScaleBiasSoftmaxDropout:
      return xla::gpu::CudnnfMHAKind::kScaleBiasSoftmaxDropout;
  }
}

static auto EncodeFusedAttentionBackwardDAGSignature(
    lmhlo_gpu::FusedMhaBackwardDagSignature signature) {
  switch (signature) {
    // backward
    case mlir::lmhlo_gpu::FusedMhaBackwardDagSignature::
        BackwardScaleBiasSoftmax:
      return xla::gpu::CudnnfMHAKind::kBackwardScaleBiasSoftmax;
    case mlir::lmhlo_gpu::FusedMhaBackwardDagSignature::
        BackwardScaleBiasSoftmaxDropout:
      return xla::gpu::CudnnfMHAKind::kBackwardScaleBiasSoftmaxDropout;
    case mlir::lmhlo_gpu::FusedMhaBackwardDagSignature::
        BackwardScaleBiasMaskSoftmax:
      return xla::gpu::CudnnfMHAKind::kBackwardScaleBiasMaskSoftmax;
    case mlir::lmhlo_gpu::FusedMhaBackwardDagSignature::
        BackwardScaleBiasMaskSoftmaxDropout:
      return xla::gpu::CudnnfMHAKind::kBackwardScaleBiasMaskSoftmaxDropout;
  }
}

void PopulateFusedAttentionForwardDAGSignatureAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding) {
  {  // --- Encode `lmhlo_gpu::FusedMhaDagSignatureAttr`.
    encoding.Add<EnumAttrEncoding<lmhlo_gpu::FusedMhaDagSignatureAttr,
                                  lmhlo_gpu::FusedMhaDagSignature,
                                  xla::gpu::CudnnfMHAKind>>(
        EncodeFusedAttentionDAGSignature);
  }
}

void PopulateFusedAttentionBackwardDAGSignatureAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding) {
  {  // --- Encode `lmhlo_gpu::FusedMhaBackwardDagSignatureAttr`.
    encoding.Add<EnumAttrEncoding<lmhlo_gpu::FusedMhaBackwardDagSignatureAttr,
                                  lmhlo_gpu::FusedMhaBackwardDagSignature,
                                  xla::gpu::CudnnfMHAKind>>(
        EncodeFusedAttentionBackwardDAGSignature);
  }
}

void PopulateFusedAttentionAlgorithmConfigAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding) {
  {  // --- Encode `lmhlo_gpu::FusedMHAAlgorithmConfigAttr`.
    using Attr = mlir::lmhlo_gpu::FusedMHAAlgorithmConfigAttr;
    encoding.Add<xla::runtime::AggregateAttrEncoding<Attr, AlgorithmConfig>>(
        encoding, xla::runtime::AggregateAttrDef<Attr>()
                      .Add("algorithm", &Attr::getAlgorithm)
                      .Add("knob_ids", &Attr::getKnobIds)
                      .Add("knob_values", &Attr::getKnobValues)
                      .Add("workspace_size", &Attr::getWorkspaceSize));
  }
}

//===----------------------------------------------------------------------===//
// Fused Dot Attention runners caching.
//===----------------------------------------------------------------------===//

StreamExecutorFusedAttentionRunners* FusedAttentionRunners::operator()(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(&mutex_);
  return &runners_[executor];
}

StreamExecutorFusedAttentionBackwardRunners*
FusedAttentionBackwardRunners::operator()(se::StreamExecutor* executor) {
  absl::MutexLock lock(&mutex_);
  return &runners_[executor];
}

namespace {
struct DropoutAttrs {
  double dropout_rate;
  int64_t seed;
};
}  // namespace

static GpufMHADescriptor GetGpufMHADescriptor(
    CudnnfMHAKind kind, StridedMemrefView lhs_bmm1, StridedMemrefView rhs_bmm1,
    StridedMemrefView rhs_bmm2, std::optional<StridedMemrefView> mask,
    std::optional<StridedMemrefView> bias, StridedMemrefView output,
    std::optional<StridedMemrefView> activation, double fmha_scale,
    absl::Span<const int64_t> intermediate_tensor_dimensions,
    absl::Span<const int64_t> intermediate_tensor_layout, AlgorithmConfig algo,
    DotDimensionNumbers bmm1_dot_dimension_numbers,
    DotDimensionNumbers bmm2_dot_dimension_numbers,
    std::optional<DropoutAttrs> dropout = std::nullopt) {
  GpufMHADescriptor descriptor;
  descriptor.backend_config.set_fmha_scale(fmha_scale);

  auto* algorithm = descriptor.backend_config.mutable_algorithm();
  algorithm->set_algo_id(algo.algorithm);
  for (unsigned i = 0; i < algo.knob_ids.size(); ++i) {
    algorithm->mutable_tuning_knobs()->insert(
        {algo.knob_ids[i], algo.knob_values[i]});
  }
  algorithm->set_is_cudnn_frontend(true);
  if (algo.workspace_size >= 0) {
    algorithm->mutable_workspace_size()->set_value(algo.workspace_size);
  }
  descriptor.bmm1_dnums =
      ConvertDotDimensionNumbers(bmm1_dot_dimension_numbers.lhs_batch,
                                 bmm1_dot_dimension_numbers.lhs_contract,
                                 bmm1_dot_dimension_numbers.rhs_batch,
                                 bmm1_dot_dimension_numbers.rhs_contract);
  descriptor.bmm2_dnums =
      ConvertDotDimensionNumbers(bmm2_dot_dimension_numbers.lhs_batch,
                                 bmm2_dot_dimension_numbers.lhs_contract,
                                 bmm2_dot_dimension_numbers.rhs_batch,
                                 bmm2_dot_dimension_numbers.rhs_contract);
  // Apply backend config layout to the shape.
  auto apply_shape = [](StridedMemrefView& memref) {
    Shape shape = ToShape(memref);
    return ShapeUtil::MakeShapeWithDenseLayout(shape.element_type(),
                                               shape.dimensions(),
                                               shape.layout().minor_to_major());
  };
  descriptor.lhs_bmm1_shape = apply_shape(lhs_bmm1);
  descriptor.rhs_bmm1_shape = apply_shape(rhs_bmm1);
  descriptor.rhs_bmm2_shape = apply_shape(rhs_bmm2);
  descriptor.output_shapes.push_back(apply_shape(output));
  if (activation.has_value()) {
    descriptor.output_shapes.push_back(apply_shape(*activation));
  }
  if (bias.has_value()) {
    descriptor.bias_shape = apply_shape(*bias);
  }
  if (mask.has_value()) {
    descriptor.mask_shape = apply_shape(*mask);
  }

  Shape out_shape = ToShape(output);
  descriptor.intermediate_lhs_bmm2_shape = ShapeUtil::MakeShapeWithDenseLayout(
      out_shape.element_type(), intermediate_tensor_dimensions,
      intermediate_tensor_layout);

  if (dropout.has_value()) {
    descriptor.backend_config.set_dropout_rate(dropout->dropout_rate);
    descriptor.backend_config.set_seed(dropout->seed);
  }

  descriptor.kind = kind;

  return descriptor;
}

static GpufMHABackwardDescriptor GetGpufMHABackwardDescriptor(
    CudnnfMHAKind kind, StridedMemrefView bmm1_grad_gemm1_rhs,
    StridedMemrefView bmm1_grad_gemm2_rhs,
    StridedMemrefView bmm2_grad_gemm2_rhs,
    StridedMemrefView bmm2_grad_gemm1_lhs, StridedMemrefView d_output,
    std::optional<StridedMemrefView> mask,
    std::optional<StridedMemrefView> d_bias, StridedMemrefView d_bmm1_lhs,
    StridedMemrefView d_bmm1_rhs, StridedMemrefView d_bmm2_rhs,
    StridedMemrefView d_S, double fmha_scale, AlgorithmConfig algo,
    DotDimensionNumbers bmm1_grad_gemm1_dot_dimension_numbers,
    DotDimensionNumbers bmm1_grad_gemm2_dot_dimension_numbers,
    DotDimensionNumbers bmm2_grad_gemm1_dot_dimension_numbers,
    DotDimensionNumbers bmm2_grad_gemm2_dot_dimension_numbers,
    std::optional<DropoutAttrs> dropout_attrs = std::nullopt) {
  GpufMHABackwardDescriptor descriptor;
  descriptor.backend_config.set_fmha_scale(fmha_scale);

  auto* algorithm = descriptor.backend_config.mutable_algorithm();
  algorithm->set_algo_id(algo.algorithm);
  for (unsigned i = 0; i < algo.knob_ids.size(); ++i) {
    algorithm->mutable_tuning_knobs()->insert(
        {algo.knob_ids[i], algo.knob_values[i]});
  }
  algorithm->set_is_cudnn_frontend(true);
  if (algo.workspace_size >= 0) {
    algorithm->mutable_workspace_size()->set_value(algo.workspace_size);
  }

  descriptor.bmm1_grad_gemm1_dnums = ConvertDotDimensionNumbers(
      bmm1_grad_gemm1_dot_dimension_numbers.lhs_batch,
      bmm1_grad_gemm1_dot_dimension_numbers.lhs_contract,
      bmm1_grad_gemm1_dot_dimension_numbers.rhs_batch,
      bmm1_grad_gemm1_dot_dimension_numbers.rhs_contract);
  descriptor.bmm1_grad_gemm2_dnums = ConvertDotDimensionNumbers(
      bmm1_grad_gemm2_dot_dimension_numbers.lhs_batch,
      bmm1_grad_gemm2_dot_dimension_numbers.lhs_contract,
      bmm1_grad_gemm2_dot_dimension_numbers.rhs_batch,
      bmm1_grad_gemm2_dot_dimension_numbers.rhs_contract);
  descriptor.bmm2_grad_gemm1_dnums = ConvertDotDimensionNumbers(
      bmm2_grad_gemm1_dot_dimension_numbers.lhs_batch,
      bmm2_grad_gemm1_dot_dimension_numbers.lhs_contract,
      bmm2_grad_gemm1_dot_dimension_numbers.rhs_batch,
      bmm2_grad_gemm1_dot_dimension_numbers.rhs_contract);
  descriptor.bmm2_grad_gemm2_dnums = ConvertDotDimensionNumbers(
      bmm2_grad_gemm2_dot_dimension_numbers.lhs_batch,
      bmm2_grad_gemm2_dot_dimension_numbers.lhs_contract,
      bmm2_grad_gemm2_dot_dimension_numbers.rhs_batch,
      bmm2_grad_gemm2_dot_dimension_numbers.rhs_contract);

  // Apply backend config layout to the shape.
  auto apply_shape = [](StridedMemrefView& memref) {
    Shape shape = ToShape(memref);
    return ShapeUtil::MakeShapeWithDenseLayout(shape.element_type(),
                                               shape.dimensions(),
                                               shape.layout().minor_to_major());
  };
  descriptor.bmm1_grad_gemm1_rhs_shape = apply_shape(bmm1_grad_gemm1_rhs);
  descriptor.bmm1_grad_gemm2_rhs_shape = apply_shape(bmm1_grad_gemm2_rhs);
  descriptor.bmm2_grad_gemm2_rhs_shape = apply_shape(bmm2_grad_gemm2_rhs);
  descriptor.bmm2_grad_gemm1_lhs_shape = apply_shape(bmm2_grad_gemm1_lhs);

  descriptor.d_output_shape = apply_shape(d_output);
  descriptor.d_bmm1_lhs_shape = apply_shape(d_bmm1_lhs);
  descriptor.d_bmm1_rhs_shape = apply_shape(d_bmm1_rhs);
  descriptor.d_bmm2_rhs_shape = apply_shape(d_bmm2_rhs);

  if (mask.has_value()) {
    descriptor.mask_shape = apply_shape(*mask);
  }
  if (d_bias.has_value()) {
    descriptor.d_bias_shape = apply_shape(*d_bias);
  }

  if (dropout_attrs.has_value()) {
    descriptor.backend_config.set_dropout_rate(dropout_attrs->dropout_rate);
    descriptor.backend_config.set_seed(dropout_attrs->seed);
  }

  descriptor.kind = kind;

  return descriptor;
}

static absl::Status FusedAttentionForwardImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, State<FusedAttentionRunner> runner,
    StridedMemrefView lhs_bmm1, StridedMemrefView rhs_bmm1,
    StridedMemrefView rhs_bmm2, std::optional<StridedMemrefView> mask,
    std::optional<StridedMemrefView> bias, StridedMemrefView output,
    FlatMemrefView scratch, std::optional<StridedMemrefView> activation,
    int64_t uid, double fmha_scale,
    absl::Span<const int64_t> intermediate_tensor_dimensions,
    absl::Span<const int64_t> intermediate_tensor_layout,
    DotDimensionNumbers bmm1_dot_dimension_numbers,
    DotDimensionNumbers bmm2_dot_dimension_numbers,
    xla::gpu::CudnnfMHAKind kind, AlgorithmConfig algorithm_config,
    std::optional<double> dropout_rate = std::nullopt,
    std::optional<int64_t> seed = std::nullopt) {
  std::optional<DropoutAttrs> dropout_attrs = std::nullopt;
  if (dropout_rate.has_value() && seed.has_value()) {
    dropout_attrs = {*dropout_rate, *seed};
  }
  // Get or create the fused attention runner state.
  absl::StatusOr<FusedAttentionRunner*> fda =
      runner.GetOrCreate([&]() -> absl::StatusOr<FusedAttentionRunner> {
        GpufMHADescriptor descriptor = GetGpufMHADescriptor(
            kind, lhs_bmm1, rhs_bmm1, rhs_bmm2, mask, bias, output, activation,
            fmha_scale, intermediate_tensor_dimensions,
            intermediate_tensor_layout, algorithm_config,
            bmm1_dot_dimension_numbers, bmm2_dot_dimension_numbers,
            dropout_attrs);

        StatusOr<GpufMHAConfig> config = GpufMHAConfig::For(descriptor);
        if (!config.ok()) return tsl::ToAbslStatus(config.status());

        return FusedAttentionRunner(*std::move(config));
      });
  if (!fda.ok()) return fda.status();

  se::DeviceMemoryBase lhs_bmm1_buffer = GetDeviceAddress(lhs_bmm1);
  se::DeviceMemoryBase rhs_bmm1_buffer = GetDeviceAddress(rhs_bmm1);
  se::DeviceMemoryBase rhs_bmm2_buffer = GetDeviceAddress(rhs_bmm2);
  se::DeviceMemoryBase output_buffer = GetDeviceAddress(output);
  se::DeviceMemoryBase scratch_buffer = GetDeviceAddress(scratch);

  se::DeviceMemoryBase mask_buffer;
  if (mask.has_value()) {
    mask_buffer = GetDeviceAddress(*mask);
  }
  se::DeviceMemoryBase bias_buffer;
  if (bias.has_value()) {
    bias_buffer = GetDeviceAddress(*bias);
  }
  se::DeviceMemoryBase activation_buffer;
  if (activation.has_value()) {
    activation_buffer = GetDeviceAddress(*activation);
  }

  RunFusedMHAOptions opts;
  opts.runner_cache = &(*fda)->runner;

  // Run the fused dot attention.
  auto st =
      RunGpuFMHA((*fda)->config, lhs_bmm1_buffer, rhs_bmm1_buffer,
                 rhs_bmm2_buffer, output_buffer, scratch_buffer, mask_buffer,
                 bias_buffer, activation_buffer, run_options->stream(), opts);
  if (!st.ok() || !run_options->stream()->ok()) {
    return tsl::ToAbslStatus(st);
  }
  return absl::OkStatus();
}

static absl::Status FusedAttentionBackwardImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options,
    State<FusedAttentionBackwardRunner> runner,
    StridedMemrefView bmm1_grad_gemm1_rhs,
    StridedMemrefView bmm1_grad_gemm2_rhs,
    StridedMemrefView bmm2_grad_gemm2_rhs,
    StridedMemrefView bmm2_grad_gemm1_lhs, StridedMemrefView d_output,
    std::optional<StridedMemrefView> mask, StridedMemrefView d_bmm1_lhs,
    StridedMemrefView d_bmm1_rhs, StridedMemrefView d_bmm2_rhs,
    StridedMemrefView d_S, FlatMemrefView scratch,
    std::optional<StridedMemrefView> d_bias, int64_t uid, double fmha_scale,
    DotDimensionNumbers bmm1_grad_gemm1_dot_dimension_numbers,
    DotDimensionNumbers bmm1_grad_gemm2_dot_dimension_numbers,
    DotDimensionNumbers bmm2_grad_gemm1_dot_dimension_numbers,
    DotDimensionNumbers bmm2_grad_gemm2_dot_dimension_numbers,
    xla::gpu::CudnnfMHAKind kind, AlgorithmConfig algorithm_config,
    std::optional<double> dropout_rate = std::nullopt,
    std::optional<int64_t> seed = std::nullopt) {
  std::optional<DropoutAttrs> dropout_attrs = std::nullopt;
  if (dropout_rate.has_value() && seed.has_value()) {
    dropout_attrs = {*dropout_rate, *seed};
  }

  // Get or create the fused attention runner state.
  absl::StatusOr<FusedAttentionBackwardRunner*> fda =
      runner.GetOrCreate([&]() -> absl::StatusOr<FusedAttentionBackwardRunner> {
        GpufMHABackwardDescriptor descriptor = GetGpufMHABackwardDescriptor(
            kind, bmm1_grad_gemm1_rhs, bmm1_grad_gemm2_rhs, bmm2_grad_gemm2_rhs,
            bmm2_grad_gemm1_lhs, d_output, mask, d_bias, d_bmm1_lhs, d_bmm1_rhs,
            d_bmm2_rhs, d_S, fmha_scale, algorithm_config,
            bmm1_grad_gemm1_dot_dimension_numbers,
            bmm1_grad_gemm2_dot_dimension_numbers,
            bmm2_grad_gemm1_dot_dimension_numbers,
            bmm2_grad_gemm2_dot_dimension_numbers, dropout_attrs);

        StatusOr<GpufMHABackwardConfig> config =
            GpufMHABackwardConfig::For(descriptor);
        if (!config.ok()) return tsl::ToAbslStatus(config.status());

        return FusedAttentionBackwardRunner(*std::move(config));
      });
  if (!fda.ok()) return fda.status();

  se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer =
      GetDeviceAddress(bmm1_grad_gemm1_rhs);
  se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer =
      GetDeviceAddress(bmm1_grad_gemm2_rhs);
  se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer =
      GetDeviceAddress(bmm2_grad_gemm2_rhs);
  se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer =
      GetDeviceAddress(bmm2_grad_gemm1_lhs);

  se::DeviceMemoryBase d_output_buffer = GetDeviceAddress(d_output);
  se::DeviceMemoryBase d_bmm1_lhs_buffer = GetDeviceAddress(d_bmm1_lhs);
  se::DeviceMemoryBase d_bmm1_rhs_buffer = GetDeviceAddress(d_bmm1_rhs);
  se::DeviceMemoryBase d_bmm2_rhs_buffer = GetDeviceAddress(d_bmm2_rhs);
  se::DeviceMemoryBase d_S_buffer = GetDeviceAddress(d_S);
  se::DeviceMemoryBase scratch_buffer = GetDeviceAddress(scratch);

  se::DeviceMemoryBase mask_buffer;
  if (mask.has_value()) {
    mask_buffer = GetDeviceAddress(*mask);
  }

  se::DeviceMemoryBase d_bias_buffer;
  if (d_bias.has_value()) {
    d_bias_buffer = GetDeviceAddress(*d_bias);
  }

  RunFusedMHABackwardOptions opts;
  opts.runner_cache = &(*fda)->runner;

  // Run the fused attention backward.
  auto st = RunGpuFMHABackward(
      (*fda)->config, bmm1_grad_gemm1_rhs_buffer, bmm1_grad_gemm2_rhs_buffer,
      bmm2_grad_gemm1_lhs_buffer, bmm2_grad_gemm2_rhs_buffer, d_output_buffer,
      scratch_buffer, d_bmm1_lhs_buffer, d_bmm1_rhs_buffer, d_bmm2_rhs_buffer,
      d_S_buffer, mask_buffer, d_bias_buffer, run_options->stream(), opts);
  if (!st.ok() || !run_options->stream()->ok()) {
    return tsl::ToAbslStatus(st);
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Fused Attention custom calls bindings and registration.
//===----------------------------------------------------------------------===//

template <typename... Ts>
auto BindFusedAttentionAttributes(runtime::CustomCallBinding<Ts...> binding) {
  return std::move(binding)
      .template Attr<int64_t>("uid")
      .template Attr<double>("fmha_scale")
      .template Attr<absl::Span<const int64_t>>(
          "intermediate_tensor_dimensions")
      .template Attr<absl::Span<const int64_t>>("intermediate_tensor_layout")
      .template Attr<DotDimensionNumbers>("bmm1_dot_dimension_numbers")
      .template Attr<DotDimensionNumbers>("bmm2_dot_dimension_numbers")
      .template Attr<xla::gpu::CudnnfMHAKind>("fused_mha_dag")
      .template Attr<AlgorithmConfig>("algorithm_config");
}

auto FusedAttentionCall(const char* name) {
  return CustomCall::Bind(name)
      .UserData<const ServiceExecutableRunOptions*>()
      .UserData<const DebugOptions*>()
      .State<FusedAttentionRunner>("uid")
      .Arg<StridedMemrefView>()   // lhs_bmm1
      .Arg<StridedMemrefView>()   // rhs_bmm1
      .Arg<StridedMemrefView>();  // rhs_bmm2
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionBmmBmmInference, FunctionWrapper<FusedAttentionForwardImpl>(),
    checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall("xla.gpu.fused.attention.bmm.bmm.inference")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Value(std::optional<StridedMemrefView>())  // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // activation
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionBmmBmmForward, FunctionWrapper<FusedAttentionForwardImpl>(),
    checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall("xla.gpu.fused.attention.bmm.bmm.forward")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Value(std::optional<StridedMemrefView>())  // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Arg<StridedMemrefView>()                   // activation
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionSoftmaxInference,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall("xla.gpu.fused.attention.softmax.inference")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Value(std::optional<StridedMemrefView>())  // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // activation
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionSoftmaxForward, FunctionWrapper<FusedAttentionForwardImpl>(),
    checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall("xla.gpu.fused.attention.softmax.forward")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Value(std::optional<StridedMemrefView>())  // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Arg<StridedMemrefView>()                   // activation
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionSoftmaxDropoutInference,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall("xla.gpu.fused.attention.softmax.dropout.inference")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Value(std::optional<StridedMemrefView>())  // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // activation
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionSoftmaxDropoutForward,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall("xla.gpu.fused.attention.softmax.dropout.forward")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Value(std::optional<StridedMemrefView>())  // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Arg<StridedMemrefView>()                   // activation
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasSoftmaxInference,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall(
            "xla.gpu.fused.attention.scale.bias.softmax.inference")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Arg<StridedMemrefView>()                   // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // activation
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasSoftmaxForward,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall("xla.gpu.fused.attention.scale.bias.softmax.forward")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Arg<StridedMemrefView>()                   // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Arg<StridedMemrefView>()                   // activation
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasSoftmaxDropoutInference,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall(
            "xla.gpu.fused.attention.scale.bias.softmax.dropout.inference")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Arg<StridedMemrefView>()                   // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // activation
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasSoftmaxDropoutForward,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall(
            "xla.gpu.fused.attention.scale.bias.softmax.dropout.forward")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Arg<StridedMemrefView>()                   // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Arg<StridedMemrefView>()                   // activation
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleMaskSoftmaxInference,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall(
            "xla.gpu.fused.attention.scale.mask.softmax.inference")
            .Arg<StridedMemrefView>()                   // mask
            .Value(std::optional<StridedMemrefView>())  // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // activation
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleMaskSoftmaxForward,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall("xla.gpu.fused.attention.scale.mask.softmax.forward")
            .Arg<StridedMemrefView>()                   // mask
            .Value(std::optional<StridedMemrefView>())  // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Arg<StridedMemrefView>()                   // activation
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleMaskSoftmaxDropoutInference,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall(
            "xla.gpu.fused.attention.scale.mask.softmax.dropout.inference")
            .Arg<StridedMemrefView>()                   // mask
            .Value(std::optional<StridedMemrefView>())  // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // activation
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleMaskSoftmaxDropoutForward,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall(
            "xla.gpu.fused.attention.scale.mask.softmax.dropout.forward")
            .Arg<StridedMemrefView>()                   // mask
            .Value(std::optional<StridedMemrefView>())  // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Arg<StridedMemrefView>()                   // activation
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasMaskSoftmaxInference,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall(
            "xla.gpu.fused.attention.scale.bias.mask.softmax.inference")
            .Arg<StridedMemrefView>()                   // mask
            .Arg<StridedMemrefView>()                   // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // activation
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasMaskSoftmaxForward,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall(
            "xla.gpu.fused.attention.scale.bias.mask.softmax.forward")
            .Arg<StridedMemrefView>()  // mask
            .Arg<StridedMemrefView>()  // bias
            .Arg<StridedMemrefView>()  // output
            .Arg<FlatMemrefView>()     // scratch
            .Arg<StridedMemrefView>()  // activation
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasMaskSoftmaxDropoutInference,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall(
            "xla.gpu.fused.attention.scale.bias.mask.softmax.dropout.inference")
            .Arg<StridedMemrefView>()                   // mask
            .Arg<StridedMemrefView>()                   // bias
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // activation
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasMaskSoftmaxDropoutForward,
    FunctionWrapper<FusedAttentionForwardImpl>(), checks,
    BindFusedAttentionAttributes(
        FusedAttentionCall(
            "xla.gpu.fused.attention.scale.bias.mask.softmax.dropout.forward")
            .Arg<StridedMemrefView>()  // mask
            .Arg<StridedMemrefView>()  // bias
            .Arg<StridedMemrefView>()  // output
            .Arg<FlatMemrefView>()     // scratch
            .Arg<StridedMemrefView>()  // activation
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

template <typename... Ts>
auto BindFusedAttentionBackwardAttributes(
    runtime::CustomCallBinding<Ts...> binding) {
  return std::move(binding)
      .template Attr<int64_t>("uid")
      .template Attr<double>("fmha_scale")
      .template Attr<DotDimensionNumbers>(
          "bmm1_grad_gemm1_dot_dimension_numbers")
      .template Attr<DotDimensionNumbers>(
          "bmm1_grad_gemm2_dot_dimension_numbers")
      .template Attr<DotDimensionNumbers>(
          "bmm2_grad_gemm1_dot_dimension_numbers")
      .template Attr<DotDimensionNumbers>(
          "bmm2_grad_gemm2_dot_dimension_numbers")
      .template Attr<xla::gpu::CudnnfMHAKind>("fused_mha_dag")
      .template Attr<AlgorithmConfig>("algorithm_config");
}

auto FusedAttentionBackwardCall(const char* name) {
  return CustomCall::Bind(name)
      .UserData<const ServiceExecutableRunOptions*>()
      .UserData<const DebugOptions*>()
      .State<FusedAttentionBackwardRunner>("uid")
      .Arg<StridedMemrefView>()  // bmm1_grad_gemm1_rhs
      .Arg<StridedMemrefView>()  // bmm1_grad_gemm2_rhs
      .Arg<StridedMemrefView>()  // bmm2_grad_gemm2_rhs
      .Arg<StridedMemrefView>()  // bmm2_grad_gemm1_lhs
      .Arg<StridedMemrefView>();
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasSoftmaxBackward,
    FunctionWrapper<FusedAttentionBackwardImpl>(), checks,
    BindFusedAttentionBackwardAttributes(
        FusedAttentionBackwardCall(
            "xla.gpu.fused.attention.backward.scale.dbias.softmax")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Arg<StridedMemrefView>()                   // d_bmm1_lhs
            .Arg<StridedMemrefView>()                   // d_bmm1_rhs
            .Arg<StridedMemrefView>()                   // d_bmm2_rhs
            .Arg<StridedMemrefView>()                   // d_S
            .Arg<FlatMemrefView>()                      // scratch
            .Arg<StridedMemrefView>()                   // d_bias
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleSoftmaxBackward,
    FunctionWrapper<FusedAttentionBackwardImpl>(), checks,
    BindFusedAttentionBackwardAttributes(
        FusedAttentionBackwardCall(
            "xla.gpu.fused.attention.backward.scale.softmax")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Arg<StridedMemrefView>()                   // d_bmm1_lhs
            .Arg<StridedMemrefView>()                   // d_bmm1_rhs
            .Arg<StridedMemrefView>()                   // d_bmm2_rhs
            .Arg<StridedMemrefView>()                   // d_S
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // d_bias
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasSoftmaxDropoutBackward,
    FunctionWrapper<FusedAttentionBackwardImpl>(), checks,
    BindFusedAttentionBackwardAttributes(
        FusedAttentionBackwardCall(
            "xla.gpu.fused.attention.backward.scale.dbias.softmax.dropout")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Arg<StridedMemrefView>()                   // d_bmm1_lhs
            .Arg<StridedMemrefView>()                   // d_bmm1_rhs
            .Arg<StridedMemrefView>()                   // d_bmm2_rhs
            .Arg<StridedMemrefView>()                   // d_S
            .Arg<FlatMemrefView>()                      // scratch
            .Arg<StridedMemrefView>()                   // d_bias
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleSoftmaxDropoutBackward,
    FunctionWrapper<FusedAttentionBackwardImpl>(), checks,
    BindFusedAttentionBackwardAttributes(
        FusedAttentionBackwardCall(
            "xla.gpu.fused.attention.backward.scale.softmax.dropout")
            .Value(std::optional<StridedMemrefView>())  // mask
            .Arg<StridedMemrefView>()                   // d_bmm1_lhs
            .Arg<StridedMemrefView>()                   // d_bmm1_rhs
            .Arg<StridedMemrefView>()                   // d_bmm2_rhs
            .Arg<StridedMemrefView>()                   // d_S
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // d_bias
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasMaskSoftmaxBackward,
    FunctionWrapper<FusedAttentionBackwardImpl>(), checks,
    BindFusedAttentionBackwardAttributes(
        FusedAttentionBackwardCall(
            "xla.gpu.fused.attention.backward.scale.dbias.mask.softmax")
            .Arg<StridedMemrefView>()  // mask
            .Arg<StridedMemrefView>()  // d_bmm1_lhs
            .Arg<StridedMemrefView>()  // d_bmm1_rhs
            .Arg<StridedMemrefView>()  // d_bmm2_rhs
            .Arg<StridedMemrefView>()  // d_S
            .Arg<FlatMemrefView>()     // scratch
            .Arg<StridedMemrefView>()  // d_bias
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleMaskSoftmaxBackward,
    FunctionWrapper<FusedAttentionBackwardImpl>(), checks,
    BindFusedAttentionBackwardAttributes(
        FusedAttentionBackwardCall(
            "xla.gpu.fused.attention.backward.scale.mask.softmax")
            .Arg<StridedMemrefView>()                   // mask
            .Arg<StridedMemrefView>()                   // d_bmm1_lhs
            .Arg<StridedMemrefView>()                   // d_bmm1_rhs
            .Arg<StridedMemrefView>()                   // d_bmm2_rhs
            .Arg<StridedMemrefView>()                   // d_S
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // d_bias
        )
        .Value(std::optional<double>())   // dropout_rate
        .Value(std::optional<int64_t>())  // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleBiasMaskSoftmaxDropoutBackward,
    FunctionWrapper<FusedAttentionBackwardImpl>(), checks,
    BindFusedAttentionBackwardAttributes(
        FusedAttentionBackwardCall(
            "xla.gpu.fused.attention.backward.scale.dbias.mask.softmax.dropout")
            .Arg<StridedMemrefView>()  // mask
            .Arg<StridedMemrefView>()  // d_bmm1_lhs
            .Arg<StridedMemrefView>()  // d_bmm1_rhs
            .Arg<StridedMemrefView>()  // d_bmm2_rhs
            .Arg<StridedMemrefView>()  // d_S
            .Arg<FlatMemrefView>()     // scratch
            .Arg<StridedMemrefView>()  // d_bias
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    FusedAttentionScaleMaskSoftmaxDropoutBackward,
    FunctionWrapper<FusedAttentionBackwardImpl>(), checks,
    BindFusedAttentionBackwardAttributes(
        FusedAttentionBackwardCall(
            "xla.gpu.fused.attention.backward.scale.mask.softmax.dropout")
            .Arg<StridedMemrefView>()                   // mask
            .Arg<StridedMemrefView>()                   // d_bmm1_lhs
            .Arg<StridedMemrefView>()                   // d_bmm1_rhs
            .Arg<StridedMemrefView>()                   // d_bmm2_rhs
            .Arg<StridedMemrefView>()                   // d_S
            .Arg<FlatMemrefView>()                      // scratch
            .Value(std::optional<StridedMemrefView>())  // d_bias
        )
        .Attr<double>("dropout_rate")  // dropout_rate
        .Attr<int64_t>("seed")         // seed
);
//===----------------------------------------------------------------------===//
// cuBLASLt custom calls bindings and registration.
//===----------------------------------------------------------------------===//
void RegisterFusedAttentionCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  auto fused_attention = [](std::string name) {
    return "xla.gpu.fused.attention." + name;
  };
  registry.Register(fused_attention("bmm.bmm.inference"),
                    FusedAttentionBmmBmmInference);
  registry.Register(fused_attention("bmm.bmm.forward"),
                    FusedAttentionBmmBmmForward);
  registry.Register(fused_attention("softmax.inference"),
                    FusedAttentionSoftmaxInference);
  registry.Register(fused_attention("softmax.forward"),
                    FusedAttentionSoftmaxForward);
  registry.Register(fused_attention("softmax.dropout.inference"),
                    FusedAttentionSoftmaxDropoutInference);
  registry.Register(fused_attention("softmax.dropout.forward"),
                    FusedAttentionSoftmaxDropoutForward);
  registry.Register(fused_attention("scale.bias.softmax.inference"),
                    FusedAttentionScaleBiasSoftmaxInference);
  registry.Register(fused_attention("scale.bias.softmax.forward"),
                    FusedAttentionScaleBiasSoftmaxForward);
  registry.Register(fused_attention("scale.bias.softmax.dropout.inference"),
                    FusedAttentionScaleBiasSoftmaxDropoutInference);
  registry.Register(fused_attention("scale.bias.softmax.dropout.forward"),
                    FusedAttentionScaleBiasSoftmaxDropoutForward);
  registry.Register(fused_attention("scale.mask.softmax.inference"),
                    FusedAttentionScaleMaskSoftmaxInference);
  registry.Register(fused_attention("scale.mask.softmax.forward"),
                    FusedAttentionScaleMaskSoftmaxForward);
  registry.Register(fused_attention("scale.mask.softmax.dropout.inference"),
                    FusedAttentionScaleMaskSoftmaxDropoutInference);
  registry.Register(fused_attention("scale.mask.softmax.dropout.forward"),
                    FusedAttentionScaleMaskSoftmaxDropoutForward);
  registry.Register(fused_attention("scale.bias.mask.softmax.inference"),
                    FusedAttentionScaleBiasMaskSoftmaxInference);
  registry.Register(fused_attention("scale.bias.mask.softmax.forward"),
                    FusedAttentionScaleBiasMaskSoftmaxForward);
  registry.Register(
      fused_attention("scale.bias.mask.softmax.dropout.inference"),
      FusedAttentionScaleBiasMaskSoftmaxDropoutInference);
  registry.Register(fused_attention("scale.bias.mask.softmax.dropout.forward"),
                    FusedAttentionScaleBiasMaskSoftmaxDropoutForward);
}

void RegisterFusedAttentionBackwardCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  auto fused_attention = [](std::string name) {
    return "xla.gpu.fused.attention.backward." + name;
  };
  registry.Register(fused_attention("scale.dbias.softmax"),
                    FusedAttentionScaleBiasSoftmaxBackward);
  registry.Register(fused_attention("scale.softmax"),
                    FusedAttentionScaleSoftmaxBackward);
  registry.Register(fused_attention("scale.dbias.softmax.dropout"),
                    FusedAttentionScaleBiasSoftmaxDropoutBackward);
  registry.Register(fused_attention("scale.softmax.dropout"),
                    FusedAttentionScaleSoftmaxDropoutBackward);
  registry.Register(fused_attention("scale.dbias.mask.softmax"),
                    FusedAttentionScaleBiasMaskSoftmaxBackward);
  registry.Register(fused_attention("scale.mask.softmax"),
                    FusedAttentionScaleMaskSoftmaxBackward);
  registry.Register(fused_attention("scale.dbias.mask.softmax.dropout"),
                    FusedAttentionScaleBiasMaskSoftmaxDropoutBackward);
  registry.Register(fused_attention("scale.mask.softmax.dropout"),
                    FusedAttentionScaleMaskSoftmaxDropoutBackward);
}
}  // namespace gpu
}  // namespace xla
