/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_GEMMLOWP_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_GEMMLOWP_H_

#include <cstdint>
#include <type_traits>

#include "public/gemmlowp.h"
#include "tensorflow/lite/experimental/ruy/ruy.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_ruy.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

template <typename DstScalar>
struct GemmlowpSaturatingCastStage {};

template <>
struct GemmlowpSaturatingCastStage<std::uint8_t> {
  using Type = gemmlowp::OutputStageSaturatingCastToUint8;
};

template <>
struct GemmlowpSaturatingCastStage<std::int8_t> {
  using Type = gemmlowp::OutputStageSaturatingCastToInt8;
};

template <>
struct GemmlowpSaturatingCastStage<std::int16_t> {
  using Type = gemmlowp::OutputStageSaturatingCastToInt16;
};

template <typename DstScalar>
struct GemmlowpBitDepthParams {};

template <>
struct GemmlowpBitDepthParams<std::uint8_t> {
  using Type = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;
};

template <>
struct GemmlowpBitDepthParams<std::int8_t> {
  using Type = gemmlowp::SignedL8R8WithLhsNonzeroBitDepthParams;
};

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplUsingGemmlowp {};

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct GemmImplUsingGemmlowp<
    LhsScalar, RhsScalar, AccumScalar, DstScalar,
    QuantizationFlavor::kIntegerWithUniformMultiplier> {
  static_assert(std::is_same<LhsScalar, RhsScalar>::value, "");
  static_assert(std::is_same<AccumScalar, std::int32_t>::value, "");
  using SrcScalar = LhsScalar;

  static void Run(
      const MatrixParams<SrcScalar>& lhs_params, const SrcScalar* lhs_data,
      const MatrixParams<SrcScalar>& rhs_params, const SrcScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<std::int32_t, DstScalar,
                       QuantizationFlavor::kIntegerWithUniformMultiplier>&
          params,
      CpuBackendContext* context) {
    gemmlowp::MatrixMap<const SrcScalar, gemmlowp::MapOrder::RowMajor>
        gemmlowp_lhs(lhs_data, lhs_params.rows, lhs_params.cols);
    gemmlowp::MatrixMap<const SrcScalar, gemmlowp::MapOrder::ColMajor>
        gemmlowp_rhs(rhs_data, rhs_params.rows, rhs_params.cols);
    gemmlowp::MatrixMap<DstScalar, gemmlowp::MapOrder::ColMajor> gemmlowp_dst(
        dst_data, dst_params.rows, dst_params.cols);

    using ColVectorMap =
        gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>;
    ColVectorMap bias_vector(params.bias, lhs_params.rows);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent scale_stage;
    scale_stage.result_offset_after_shift = dst_params.zero_point;
    scale_stage.result_fixedpoint_multiplier = params.multiplier_fixedpoint;
    scale_stage.result_exponent = params.multiplier_exponent;
    using SaturatingCastStageType =
        typename GemmlowpSaturatingCastStage<DstScalar>::Type;
    gemmlowp::OutputStageClamp clamp_stage;
    clamp_stage.min = params.clamp_min;
    clamp_stage.max = params.clamp_max;
    SaturatingCastStageType saturating_cast_stage;
    auto output_pipeline = std::make_tuple(bias_addition_stage, scale_stage,
                                           clamp_stage, saturating_cast_stage);
    using BitDepthParams = typename GemmlowpBitDepthParams<SrcScalar>::Type;
    gemmlowp::GemmWithOutputPipeline<SrcScalar, DstScalar, BitDepthParams>(
        context->gemmlowp_context(), gemmlowp_lhs, gemmlowp_rhs, &gemmlowp_dst,
        -lhs_params.zero_point, -rhs_params.zero_point, output_pipeline);
  }
};

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct GemmImplUsingGemmlowp<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                             QuantizationFlavor::kIntegerWithPerRowMultiplier> {
  static_assert(std::is_same<LhsScalar, RhsScalar>::value, "");
  static_assert(std::is_same<AccumScalar, std::int32_t>::value, "");
  using SrcScalar = LhsScalar;

  static void Run(
      const MatrixParams<SrcScalar>& lhs_params, const SrcScalar* lhs_data,
      const MatrixParams<SrcScalar>& rhs_params, const SrcScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<std::int32_t, DstScalar,
                       QuantizationFlavor::kIntegerWithPerRowMultiplier>&
          params,
      CpuBackendContext* context) {
    // gemmlowp support for this per-channel path is limited to NEON.
    // We fall back to ruy outside of NEON.
#ifdef GEMMLOWP_NEON
    gemmlowp::MatrixMap<const SrcScalar, gemmlowp::MapOrder::RowMajor>
        gemmlowp_lhs(lhs_data, lhs_params.rows, lhs_params.cols);
    gemmlowp::MatrixMap<const SrcScalar, gemmlowp::MapOrder::ColMajor>
        gemmlowp_rhs(rhs_data, rhs_params.rows, rhs_params.cols);
    gemmlowp::MatrixMap<DstScalar, gemmlowp::MapOrder::ColMajor> gemmlowp_dst(
        dst_data, dst_params.rows, dst_params.cols);

    using ColVectorMap =
        gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>;
    ColVectorMap bias_vector(params.bias, lhs_params.rows);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageScaleInt32ByFixedPointAndExponentPC<
        gemmlowp::VectorShape::Col>
        scale_stage;
    scale_stage.result_offset_after_shift = dst_params.zero_point;
    scale_stage.result_fixedpoint_multiplier =
        ColVectorMap(params.multiplier_fixedpoint_perchannel, dst_params.rows);
    scale_stage.result_exponent =
        ColVectorMap(params.multiplier_exponent_perchannel, dst_params.rows);
    using SaturatingCastStageType =
        typename GemmlowpSaturatingCastStage<DstScalar>::Type;
    gemmlowp::OutputStageClamp clamp_stage;
    clamp_stage.min = params.clamp_min;
    clamp_stage.max = params.clamp_max;
    SaturatingCastStageType saturating_cast_stage;
    auto output_pipeline = std::make_tuple(bias_addition_stage, scale_stage,
                                           clamp_stage, saturating_cast_stage);
    using BitDepthParams = typename GemmlowpBitDepthParams<SrcScalar>::Type;
    gemmlowp::GemmWithOutputPipeline<SrcScalar, DstScalar, BitDepthParams>(
        context->gemmlowp_context(), gemmlowp_lhs, gemmlowp_rhs, &gemmlowp_dst,
        -lhs_params.zero_point, -rhs_params.zero_point, output_pipeline);
#else
    GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                     QuantizationFlavor::kIntegerWithPerRowMultiplier>::
        Run(lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data,
            params, context);
#endif
  }
};

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_GEMMLOWP_H_
