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

#include "tensorflow/compiler/xla/service/gpu/gpu_fused_mha_runner.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

namespace {
using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::dnn::DataType;
using se::dnn::MatmulTensorDescriptor;
using se::dnn::TensorDescriptor;

template <typename ElementType, typename OutputType>
Status RunFusedMHASimple(GpufMHAParams params, se::Stream *stream,
                         RunFusedMHAOptions options,
                         DeviceMemory<ElementType> lhs_bmm1_buffer,
                         DeviceMemory<ElementType> rhs_bmm1_buffer,
                         DeviceMemory<ElementType> rhs_bmm2_buffer,
                         DeviceMemory<OutputType> output_buffer,
                         DeviceMemoryBase scratch_memory) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHASimpleOp> *lazy_runner =
      options.runner_cache->AsFusedMHASimpleRunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::FusedMHASimpleOp>> local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }
  TF_ASSIGN_OR_RETURN(se::dnn::FusedMHAKind kind,
                      GetDNNFusedMHAKindFromCudnnfMHAKind(params.config->kind));
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate) dropout_rate = *params.config->dropout_rate;
  se::dnn::FusedMHASimpleOp::Config config{kind,
                                           params.config->lhs_bmm1,
                                           params.config->rhs_bmm1,
                                           params.config->rhs_bmm2,
                                           params.config->intermediate_lhs_bmm2,
                                           params.config->output,
                                           dropout_rate};
  TF_ASSIGN_OR_RETURN(auto *runner,
                      lazy_runner->GetOrCreateRunner(config, stream));
  return (*runner)(stream, options.profile_result, scratch_memory,
                   lhs_bmm1_buffer, rhs_bmm1_buffer, rhs_bmm2_buffer,
                   output_buffer);
  return OkStatus();
}

template <typename ElementType, typename OutputType>
Status RunFusedMHAScaleMaskSoftmax(GpufMHAParams params, se::Stream *stream,
                                   RunFusedMHAOptions options,
                                   DeviceMemory<ElementType> lhs_bmm1_buffer,
                                   DeviceMemory<ElementType> rhs_bmm1_buffer,
                                   DeviceMemory<ElementType> rhs_bmm2_buffer,
                                   DeviceMemory<OutputType> output_buffer,
                                   DeviceMemory<uint8_t> mask_buffer,
                                   DeviceMemoryBase scratch_memory) {
  return InternalError("Unimplemented.");
}

template <typename ElementType, typename BiasType, typename OutputType>
Status RunFusedMHAScaleBiasMaskSoftmax(
    GpufMHAParams params, se::Stream *stream, RunFusedMHAOptions options,
    DeviceMemory<ElementType> lhs_bmm1_buffer,
    DeviceMemory<ElementType> rhs_bmm1_buffer,
    DeviceMemory<ElementType> rhs_bmm2_buffer,
    DeviceMemory<OutputType> output_buffer, DeviceMemory<uint8_t> mask_buffer,
    DeviceMemory<BiasType> bias_buffer, DeviceMemoryBase scratch_memory) {
  return InternalError("Unimplemented.");
}

template <typename ElementType, typename BiasType, typename OutputType>
Status RunGpuFMHAImpl(const GpufMHAParams &params, se::Stream *stream,
                      se::DeviceMemoryBase scratch_memory,
                      RunFusedMHAOptions options) {
  auto lhs_bmm1_buffer = se::DeviceMemory<ElementType>(params.lhs_bmm1_buffer);
  auto rhs_bmm1_buffer = se::DeviceMemory<ElementType>(params.rhs_bmm1_buffer);
  auto rhs_bmm2_buffer = se::DeviceMemory<ElementType>(params.rhs_bmm2_buffer);
  auto output_buffer = se::DeviceMemory<OutputType>(params.output_buffer);

  se::dnn::AlgorithmDesc algorithm = params.config->algorithm;
  if (options.runner_cache) {
    algorithm = options.runner_cache->ToAlgorithmDesc();
  }

  Status run_status = OkStatus();
  switch (params.config->kind) {
    case CudnnfMHAKind::kBmmBmm:
    case CudnnfMHAKind::kSoftmaxDropout:
      run_status = RunFusedMHASimple<ElementType, OutputType>(
          params, stream, options, lhs_bmm1_buffer, rhs_bmm1_buffer,
          rhs_bmm2_buffer, output_buffer, scratch_memory);
      break;

    case CudnnfMHAKind::kScaleMaskSoftmax:
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
      TF_RET_CHECK(params.mask_buffer.has_value());
      run_status = RunFusedMHAScaleMaskSoftmax<ElementType, OutputType>(
          params, stream, options, lhs_bmm1_buffer, rhs_bmm1_buffer,
          rhs_bmm2_buffer, output_buffer,
          se::DeviceMemory<uint8_t>(*params.mask_buffer), scratch_memory);
      break;

    case CudnnfMHAKind::kScaleBiasMaskSoftmax:
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
      TF_RET_CHECK(params.bias_buffer.has_value());
      TF_RET_CHECK(params.mask_buffer.has_value());
      run_status =
          RunFusedMHAScaleBiasMaskSoftmax<ElementType, BiasType, OutputType>(
              params, stream, options, lhs_bmm1_buffer, rhs_bmm1_buffer,
              rhs_bmm2_buffer, output_buffer,
              se::DeviceMemory<uint8_t>(*params.mask_buffer),
              se::DeviceMemory<BiasType>(*params.bias_buffer), scratch_memory);
      break;

    default:
      return InternalError("Invalid cuDNN fMHA kind");
  }

  if (run_status != OkStatus()) {
    return run_status;
  }

  if (!stream->ok()) {
    return InternalError("Unable to launch FMHA with type %s and algorithm %s",
                         CudnnfMHAKindToString(params.config->kind),
                         algorithm.ToString());
  }

  return OkStatus();
}
}  // namespace

/*static*/ StatusOr<GpufMHAConfig> GpufMHAConfig::For(
    const GpufMHADescriptor &desc) {
  // Get shapes from desc.
  const Shape &lhs_bmm1_shape = desc.lhs_bmm1_shape;
  const Shape &rhs_bmm1_shape = desc.rhs_bmm1_shape;
  const Shape &rhs_bmm2_shape = desc.rhs_bmm2_shape;
  const Shape &intermediate_lhs_bmm2_shape = desc.intermediate_lhs_bmm2_shape;
  const Shape &output_shape = desc.output_shape;

  // Get DNN dtype from primtive types
  TF_ASSIGN_OR_RETURN(
      DataType lhs_bmm1_type,
      GetDNNDataTypeFromPrimitiveType(lhs_bmm1_shape.element_type()));
  TF_ASSIGN_OR_RETURN(
      DataType rhs_bmm1_type,
      GetDNNDataTypeFromPrimitiveType(rhs_bmm1_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType rhs_bmm2_type,
      GetDNNDataTypeFromPrimitiveType(rhs_bmm2_shape.element_type()));
  TF_ASSIGN_OR_RETURN(DataType lhs_bmm2_type,
                      GetDNNDataTypeFromPrimitiveType(
                          intermediate_lhs_bmm2_shape.element_type()));

  TF_ASSIGN_OR_RETURN(DataType output_type, GetDNNDataTypeFromPrimitiveType(
                                                output_shape.element_type()));
  GpufMHAConfig config;
  config.input_type = lhs_bmm1_shape.element_type();
  config.output_type = output_shape.element_type();

  // Get MatmulTensorDescriptors for BMM1
  config.lhs_bmm1 =
      MatmulTensorDescriptor::For(lhs_bmm1_type, lhs_bmm1_shape.dimensions(),
                                  desc.lhs_bmm1_shape.layout().minor_to_major(),
                                  desc.bmm1_dnums.lhs_batch_dimensions(),
                                  desc.bmm1_dnums.lhs_contracting_dimensions());
  config.rhs_bmm1 =
      MatmulTensorDescriptor::For(rhs_bmm1_type, rhs_bmm1_shape.dimensions(),
                                  desc.rhs_bmm1_shape.layout().minor_to_major(),
                                  desc.bmm1_dnums.rhs_batch_dimensions(),
                                  desc.bmm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2
  config.rhs_bmm2 =
      MatmulTensorDescriptor::For(rhs_bmm2_type, rhs_bmm2_shape.dimensions(),
                                  desc.rhs_bmm2_shape.layout().minor_to_major(),
                                  desc.bmm2_dnums.rhs_batch_dimensions(),
                                  desc.bmm2_dnums.rhs_contracting_dimensions());
  config.intermediate_lhs_bmm2 = MatmulTensorDescriptor::For(
      lhs_bmm2_type, intermediate_lhs_bmm2_shape.dimensions(),
      desc.intermediate_lhs_bmm2_shape.layout().minor_to_major(),
      desc.bmm2_dnums.lhs_batch_dimensions(),
      desc.bmm2_dnums.lhs_contracting_dimensions());

  config.output = TensorDescriptor::For(output_type, output_shape.dimensions(),
                                        output_shape.layout().minor_to_major());

  config.kind = desc.kind;
  const CudnnfMHABackendConfig &backend_config = desc.backend_config;
  config.algorithm = se::dnn::AlgorithmDesc(backend_config.algorithm());

  auto check_and_assign_mask = [&]() -> Status {
    if (desc.mask_shape) {
      const Shape &mask_shape = *desc.mask_shape;
      TF_ASSIGN_OR_RETURN(DataType mask_type, GetDNNDataTypeFromPrimitiveType(
                                                  mask_shape.element_type()));
      config.mask.emplace();
      TensorDescriptor &mask = *config.mask;
      mask = TensorDescriptor::For(mask_type, mask_shape.dimensions(),
                                   mask_shape.layout().minor_to_major());
      return OkStatus();
    } else {
      return InternalError(
          "GpufMHADescriptor should have non-nul mask shape but found null "
          "mask shape");
    }
  };

  auto check_and_assign_bias = [&]() -> Status {
    if (desc.bias_shape) {
      const Shape &bias_shape = *desc.bias_shape;
      TF_ASSIGN_OR_RETURN(DataType bias_type, GetDNNDataTypeFromPrimitiveType(
                                                  bias_shape.element_type()));
      config.bias.emplace();
      TensorDescriptor &bias = *config.bias;
      bias = TensorDescriptor::For(bias_type, bias_shape.dimensions(),
                                   bias_shape.layout().minor_to_major());
      return OkStatus();
    } else {
      return InternalError(
          "GpufMHADescriptor should have non-nul bias shape but found null "
          "bias shape");
    }
  };

  auto assign_scale = [&]() {
    config.fmha_scale.emplace();
    double &fmha_scale = *config.fmha_scale;
    fmha_scale = backend_config.fmha_scale();
  };

  auto assign_dropout_rate = [&]() {
    config.dropout_rate.emplace();
    double &dropout_rate = *config.dropout_rate;
    dropout_rate = backend_config.dropout_rate();
  };

  switch (config.kind) {
    case CudnnfMHAKind::kScaleBiasMaskSoftmax:
      TF_RETURN_IF_ERROR(check_and_assign_mask());
      TF_RETURN_IF_ERROR(check_and_assign_bias());
      assign_scale();
      break;
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
      TF_RETURN_IF_ERROR(check_and_assign_mask());
      TF_RETURN_IF_ERROR(check_and_assign_bias());
      assign_scale();
      assign_dropout_rate();
      break;
    case CudnnfMHAKind::kScaleMaskSoftmax:
      TF_RETURN_IF_ERROR(check_and_assign_mask());
      assign_scale();
      break;
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
      TF_RETURN_IF_ERROR(check_and_assign_mask());
      assign_scale();
      assign_dropout_rate();
      break;
    case CudnnfMHAKind::kBmmBmm:
    case CudnnfMHAKind::kSoftmaxDropout:
      break;
    default:
      return InternalError("Unknown fmha kind");
  }
  return config;
}

/*static*/ StatusOr<GpufMHAParams> GpufMHAParams::For(
    const GpufMHAConfig &config, se::DeviceMemoryBase lhs_bmm1_buffer,
    se::DeviceMemoryBase rhs_bmm1_buffer, se::DeviceMemoryBase rhs_bmm2_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase mask_buffer,
    se::DeviceMemoryBase bias_buffer) {
  GpufMHAParams params;
  params.config = &config;
  params.lhs_bmm1_buffer = lhs_bmm1_buffer;
  params.rhs_bmm1_buffer = rhs_bmm1_buffer;
  params.rhs_bmm2_buffer = rhs_bmm2_buffer;
  params.output_buffer = output_buffer;

  auto assign_mask_buffer = [&]() {
    params.mask_buffer.emplace();
    se::DeviceMemoryBase &mask = *params.mask_buffer;
    mask = mask_buffer;
  };

  switch (config.kind) {
    case CudnnfMHAKind::kBmmBmm:
    case CudnnfMHAKind::kSoftmaxDropout:
      break;
    case CudnnfMHAKind::kScaleMaskSoftmax:
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
      TF_RET_CHECK(!mask_buffer.is_null());
      assign_mask_buffer();
      break;
    case CudnnfMHAKind::kScaleBiasMaskSoftmax:
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
      TF_RET_CHECK(!mask_buffer.is_null());
      TF_RET_CHECK(!bias_buffer.is_null());
      assign_mask_buffer();

      params.bias_buffer.emplace();
      se::DeviceMemoryBase &bias = *params.bias_buffer;
      bias = bias_buffer;
      break;
  }
  return params;
}

Status RunGpuFMHA(
    const GpufMHAConfig &fmha_config, se::DeviceMemoryBase lhs_bmm1_buffer,
    se::DeviceMemoryBase rhs_bmm1_buffer, se::DeviceMemoryBase rhs_bmm2_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase scratch_buffer,
    se::DeviceMemoryBase mask_buffer, se::DeviceMemoryBase bias_buffer,
    se::Stream *stream, RunFusedMHAOptions options) {
  TF_ASSIGN_OR_RETURN(
      GpufMHAParams params,
      GpufMHAParams::For(fmha_config, lhs_bmm1_buffer, rhs_bmm1_buffer,
                         rhs_bmm2_buffer, output_buffer, mask_buffer,
                         bias_buffer));
  PrimitiveType input_primitive_type = fmha_config.input_type;
  switch (input_primitive_type) {
    case F16:
      return RunGpuFMHAImpl<Eigen::half, Eigen::half, Eigen::half>(
          params, stream, scratch_buffer, options);
    case BF16:
      return RunGpuFMHAImpl<Eigen::bfloat16, Eigen::bfloat16, Eigen::bfloat16>(
          params, stream, scratch_buffer, options);
    default:
      return Unimplemented("Unimplemented fused MHA");
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla