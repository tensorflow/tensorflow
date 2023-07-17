/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <optional>
#include <string>

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
Status RunFusedMHABmmBmm(GpufMHAParams params, se::Stream *stream,
                         RunFusedMHAOptions options,
                         DeviceMemory<ElementType> lhs_bmm1_buffer,
                         DeviceMemory<ElementType> rhs_bmm1_buffer,
                         DeviceMemory<ElementType> rhs_bmm2_buffer,
                         DeviceMemory<OutputType> output_buffer,
                         DeviceMemoryBase scratch_memory) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHASoftmaxOp> *lazy_runner =
      options.runner_cache->AsFusedMHASoftmaxRunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::FusedMHASoftmaxOp>> local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }
  TF_ASSIGN_OR_RETURN(se::dnn::FusedMHAKind kind,
                      GetDNNFusedMHAKindFromCudnnfMHAKind(params.config->kind));
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate) dropout_rate = *params.config->dropout_rate;

  std::optional<int64_t> seed;
  if (params.config->seed) seed = *params.config->seed;

  se::dnn::FusedMHASoftmaxOp::Config config{
      kind,
      params.config->lhs_bmm1,
      params.config->rhs_bmm1,
      params.config->rhs_bmm2,
      params.config->intermediate_lhs_bmm2,
      params.config->output,
      dropout_rate,
      seed};
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
                                   DeviceMemory<ElementType> mask_buffer,
                                   DeviceMemoryBase scratch_memory) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleMaskSoftmaxOp> *lazy_runner =
      options.runner_cache->AsFusedMHAMaskRunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleMaskSoftmaxOp>>
      local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }
  TF_ASSIGN_OR_RETURN(se::dnn::FusedMHAKind kind,
                      GetDNNFusedMHAKindFromCudnnfMHAKind(params.config->kind));
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate) dropout_rate = *params.config->dropout_rate;

  double scale = 1.0;
  if (params.config->fmha_scale) scale = *params.config->fmha_scale;

  std::optional<int64_t> seed;
  if (params.config->seed) seed = *params.config->seed;

  se::dnn::FusedMHAScaleMaskSoftmaxOp::Config config{
      kind,
      scale,
      params.config->lhs_bmm1,
      params.config->rhs_bmm1,
      params.config->rhs_bmm2,
      params.config->intermediate_lhs_bmm2,
      params.config->output,
      *params.config->mask,
      dropout_rate,
      seed};
  TF_ASSIGN_OR_RETURN(auto *runner,
                      lazy_runner->GetOrCreateRunner(config, stream));
  return (*runner)(stream, options.profile_result, scratch_memory,
                   lhs_bmm1_buffer, rhs_bmm1_buffer, mask_buffer,
                   rhs_bmm2_buffer, output_buffer);
  return OkStatus();
}

template <typename ElementType, typename BiasType, typename OutputType>
Status RunFusedMHAScaleBiasMaskSoftmax(
    GpufMHAParams params, se::Stream *stream, RunFusedMHAOptions options,
    DeviceMemory<ElementType> lhs_bmm1_buffer,
    DeviceMemory<ElementType> rhs_bmm1_buffer,
    DeviceMemory<ElementType> rhs_bmm2_buffer,
    DeviceMemory<OutputType> output_buffer,
    DeviceMemory<ElementType> mask_buffer, DeviceMemory<BiasType> bias_buffer,
    DeviceMemoryBase scratch_memory) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleBiasMaskSoftmaxOp> *lazy_runner =
      options.runner_cache->AsFusedMHABiasMaskRunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleBiasMaskSoftmaxOp>>
      local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }
  TF_ASSIGN_OR_RETURN(se::dnn::FusedMHAKind kind,
                      GetDNNFusedMHAKindFromCudnnfMHAKind(params.config->kind));
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate) dropout_rate = *params.config->dropout_rate;

  double scale = 1.0;
  if (params.config->fmha_scale) scale = *params.config->fmha_scale;

  std::optional<int64_t> seed;
  if (params.config->seed) seed = *params.config->seed;

  se::dnn::FusedMHAScaleBiasMaskSoftmaxOp::Config config{
      kind,
      scale,
      params.config->lhs_bmm1,
      params.config->rhs_bmm1,
      params.config->rhs_bmm2,
      params.config->intermediate_lhs_bmm2,
      params.config->output,
      *params.config->bias,
      *params.config->mask,
      dropout_rate,
      seed};
  TF_ASSIGN_OR_RETURN(auto *runner,
                      lazy_runner->GetOrCreateRunner(config, stream));
  return (*runner)(stream, options.profile_result, scratch_memory,
                   lhs_bmm1_buffer, rhs_bmm1_buffer, mask_buffer, bias_buffer,
                   rhs_bmm2_buffer, output_buffer);
}

template <typename ElementType, typename BiasType, typename OutputType>
Status RunFusedMHAScaleBiasSoftmax(GpufMHAParams params, se::Stream *stream,
                                   RunFusedMHAOptions options,
                                   DeviceMemory<ElementType> lhs_bmm1_buffer,
                                   DeviceMemory<ElementType> rhs_bmm1_buffer,
                                   DeviceMemory<ElementType> rhs_bmm2_buffer,
                                   DeviceMemory<OutputType> output_buffer,
                                   DeviceMemory<BiasType> bias_buffer,
                                   DeviceMemoryBase scratch_memory) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleBiasSoftmaxOp> *lazy_runner =
      options.runner_cache->AsFusedMHABiasRunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::FusedMHAScaleBiasSoftmaxOp>>
      local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }
  TF_ASSIGN_OR_RETURN(se::dnn::FusedMHAKind kind,
                      GetDNNFusedMHAKindFromCudnnfMHAKind(params.config->kind));
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate) dropout_rate = *params.config->dropout_rate;

  double scale = 1.0;
  if (params.config->fmha_scale) scale = *params.config->fmha_scale;

  std::optional<int64_t> seed;
  if (params.config->seed) seed = *params.config->seed;

  se::dnn::FusedMHAScaleBiasSoftmaxOp::Config config{
      kind,
      scale,
      params.config->lhs_bmm1,
      params.config->rhs_bmm1,
      params.config->rhs_bmm2,
      params.config->intermediate_lhs_bmm2,
      params.config->output,
      *params.config->bias,
      dropout_rate,
      seed};
  TF_ASSIGN_OR_RETURN(auto *runner,
                      lazy_runner->GetOrCreateRunner(config, stream));
  return (*runner)(stream, options.profile_result, scratch_memory,
                   lhs_bmm1_buffer, rhs_bmm1_buffer, bias_buffer,
                   rhs_bmm2_buffer, output_buffer);
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
    case CudnnfMHAKind::kSoftmax:
      run_status = RunFusedMHABmmBmm<ElementType, OutputType>(
          params, stream, options, lhs_bmm1_buffer, rhs_bmm1_buffer,
          rhs_bmm2_buffer, output_buffer, scratch_memory);
      break;

    case CudnnfMHAKind::kScaleMaskSoftmax:
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
      TF_RET_CHECK(params.mask_buffer.has_value());
      run_status = RunFusedMHAScaleMaskSoftmax<ElementType, OutputType>(
          params, stream, options, lhs_bmm1_buffer, rhs_bmm1_buffer,
          rhs_bmm2_buffer, output_buffer,
          se::DeviceMemory<ElementType>(*params.mask_buffer), scratch_memory);
      break;

    case CudnnfMHAKind::kScaleBiasMaskSoftmax:
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
      TF_RET_CHECK(params.bias_buffer.has_value());
      TF_RET_CHECK(params.mask_buffer.has_value());
      run_status =
          RunFusedMHAScaleBiasMaskSoftmax<ElementType, BiasType, OutputType>(
              params, stream, options, lhs_bmm1_buffer, rhs_bmm1_buffer,
              rhs_bmm2_buffer, output_buffer,
              se::DeviceMemory<ElementType>(*params.mask_buffer),
              se::DeviceMemory<BiasType>(*params.bias_buffer), scratch_memory);
      break;
    case CudnnfMHAKind::kScaleBiasSoftmax:
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
      TF_RET_CHECK(params.bias_buffer.has_value());
      run_status =
          RunFusedMHAScaleBiasSoftmax<ElementType, BiasType, OutputType>(
              params, stream, options, lhs_bmm1_buffer, rhs_bmm1_buffer,
              rhs_bmm2_buffer, output_buffer,
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

Status CheckAndAssignMask(const GpufMHADescriptor &desc,
                          GpufMHAConfig &config) {
  switch (config.kind) {
    case CudnnfMHAKind::kScaleBiasMaskSoftmax:
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
    case CudnnfMHAKind::kScaleMaskSoftmax:
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
      if (desc.mask_shape) {
        const Shape &mask_shape = *desc.mask_shape;

        TF_ASSIGN_OR_RETURN(DataType mask_type, GetDNNDataTypeFromPrimitiveType(
                                                    mask_shape.element_type()));
        config.mask =
            TensorDescriptor::For(mask_type, mask_shape.dimensions(),
                                  mask_shape.layout().minor_to_major());
        return OkStatus();
      } else {
        return InternalError(
            "GpufMHADescriptor should have non-null mask shape but found null "
            "mask shape");
      }
    default:
      return OkStatus();
  }
}

Status CheckAndAssignBias(const GpufMHADescriptor &desc,
                          GpufMHAConfig &config) {
  switch (config.kind) {
    case CudnnfMHAKind::kScaleBiasMaskSoftmax:
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
    case CudnnfMHAKind::kScaleBiasSoftmax:
      if (desc.bias_shape) {
        const Shape &bias_shape = *desc.bias_shape;

        TF_ASSIGN_OR_RETURN(DataType bias_type, GetDNNDataTypeFromPrimitiveType(
                                                    bias_shape.element_type()));

        config.bias =
            TensorDescriptor::For(bias_type, bias_shape.dimensions(),
                                  bias_shape.layout().minor_to_major());
        return OkStatus();
      } else {
        return InternalError(
            "GpufMHADescriptor should have non-null bias shape but found null "
            "bias shape");
      }
    default:
      return OkStatus();
  }
}

void AssignScale(GpufMHAConfig &config,
                 const CudnnfMHABackendConfig &backend_config) {
  double fmha_scale = 0.0;

  switch (config.kind) {
    case CudnnfMHAKind::kScaleBiasMaskSoftmax:
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
    case CudnnfMHAKind::kScaleMaskSoftmax:
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
    case CudnnfMHAKind::kScaleBiasSoftmax:
      fmha_scale = backend_config.fmha_scale();
      config.fmha_scale.emplace(fmha_scale);
      break;
    default:
      break;
  }
}

void AssignDropoutRate(GpufMHAConfig &config,
                       const CudnnfMHABackendConfig &backend_config) {
  double dropout_rate = 0.0;
  switch (config.kind) {
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
    case CudnnfMHAKind::kSoftmaxDropout:
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
      dropout_rate = backend_config.dropout_rate();
      config.dropout_rate.emplace(dropout_rate);
      break;
    default:
      break;
  }
}

void AssignSeed(GpufMHAConfig &config,
                const CudnnfMHABackendConfig &backend_config) {
  int64_t seed_value = 0;

  switch (config.kind) {
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
    case CudnnfMHAKind::kSoftmaxDropout:
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
      seed_value = backend_config.seed();
      config.seed.emplace(seed_value);
      break;
    default:
      break;
  }
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

  TF_RETURN_IF_ERROR(CheckAndAssignMask(desc, config));
  TF_RETURN_IF_ERROR(CheckAndAssignBias(desc, config));
  AssignScale(config, backend_config);
  AssignDropoutRate(config, backend_config);
  AssignSeed(config, backend_config);
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

  auto assign_bias_buffer = [&]() {
    params.bias_buffer.emplace();
    se::DeviceMemoryBase &bias = *params.bias_buffer;
    bias = bias_buffer;
  };

  switch (config.kind) {
    case CudnnfMHAKind::kBmmBmm:
    case CudnnfMHAKind::kSoftmaxDropout:
    case CudnnfMHAKind::kSoftmax:
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
      assign_bias_buffer();
      break;
    case CudnnfMHAKind::kScaleBiasSoftmax:
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
      TF_RET_CHECK(!bias_buffer.is_null());
      assign_bias_buffer();
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
      return absl::UnimplementedError(absl::StrFormat(
          "Unimplemented fused MHA with %s", ToString(fmha_config)));
  }
  return OkStatus();
}

std::string ToString(const GpufMHAConfig &config) {
  std::string result = "GpufMHAConfig:\n";
  absl::StrAppend(&result,
                  "input_type: ", PrimitiveType_Name(config.input_type), ", ");
  absl::StrAppend(
      &result, "output_type: ", PrimitiveType_Name(config.output_type), ", ");
  absl::StrAppend(&result, "Kind: ", CudnnfMHAKindToString(config.kind), ", ");
  if (config.fmha_scale) {
    absl::StrAppend(&result, "fmha_scale: ", *config.fmha_scale, ", ");
  }
  if (config.dropout_rate) {
    absl::StrAppend(&result, "dropout_rate: ", *config.dropout_rate, ", ");
  }
  if (config.seed) {
    absl::StrAppend(&result, "seed: ", *config.seed, ", ");
  }
  absl::StrAppend(&result, "Algorithm Desc: ", config.algorithm.ToString(),
                  "\n");
  absl::StrAppend(&result, "lhs_bmm1: ", config.lhs_bmm1.ToString(), "\n");
  absl::StrAppend(&result, "rhs_bmm1: ", config.rhs_bmm1.ToString(), "\n");
  absl::StrAppend(&result, "rhs_bmm2: ", config.rhs_bmm2.ToString(), "\n");
  absl::StrAppend(&result, "intermediate_lhs_bmm2: ",
                  config.intermediate_lhs_bmm2.ToString(), "\n");
  absl::StrAppend(&result, "output: ", config.output.ToString(), "\n");

  if (config.mask) {
    absl::StrAppend(&result, "mask: ", (*config.mask).ToString(), "\n");
  }

  if (config.bias) {
    absl::StrAppend(&result, "bias: ", (*config.bias).ToString(), "\n");
  }

  return result;
}

}  // namespace gpu
}  // namespace xla
