// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"

namespace qnn {

UndefinedQuantizeParamsWrapper::UndefinedQuantizeParamsWrapper() = default;

UndefinedQuantizeParamsWrapper::UndefinedQuantizeParamsWrapper(
    const UndefinedQuantizeParamsWrapper&) = default;

UndefinedQuantizeParamsWrapper::UndefinedQuantizeParamsWrapper(
    UndefinedQuantizeParamsWrapper&&) = default;

void UndefinedQuantizeParamsWrapper::CloneTo(Qnn_QuantizeParams_t& dst) {
  dst = qnn_quantize_param_;
}

ScaleOffsetQuantizeParamsWrapper::ScaleOffsetQuantizeParamsWrapper(
    const float scale, const std::int32_t zero_point) {
  qnn_quantize_param_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnn_quantize_param_.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qnn_quantize_param_.scaleOffsetEncoding.scale = scale;
  qnn_quantize_param_.scaleOffsetEncoding.offset = -1 * zero_point;
}

ScaleOffsetQuantizeParamsWrapper::ScaleOffsetQuantizeParamsWrapper(
    const ScaleOffsetQuantizeParamsWrapper&) = default;

ScaleOffsetQuantizeParamsWrapper::ScaleOffsetQuantizeParamsWrapper(
    ScaleOffsetQuantizeParamsWrapper&&) = default;

void ScaleOffsetQuantizeParamsWrapper::CloneTo(Qnn_QuantizeParams_t& dst) {
  dst = qnn_quantize_param_;
}

AxisScaleOffsetQuantizeParamsWrapper::AxisScaleOffsetQuantizeParamsWrapper(
    const std::int32_t axis, const absl::Span<const float> scales,
    const absl::Span<const std::int32_t> zero_points)
    : scale_offsets_(scales.size()) {
  assert(scales.size() == zero_points.size());
  for (size_t i = 0; i < scale_offsets_.size(); ++i) {
    scale_offsets_[i].scale = scales[i];
    scale_offsets_[i].offset = -1 * zero_points[i];
  }

  qnn_quantize_param_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnn_quantize_param_.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  qnn_quantize_param_.axisScaleOffsetEncoding.axis = axis;
  qnn_quantize_param_.axisScaleOffsetEncoding.numScaleOffsets =
      scale_offsets_.size();
  qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset =
      scale_offsets_.data();
}

AxisScaleOffsetQuantizeParamsWrapper::AxisScaleOffsetQuantizeParamsWrapper(
    const AxisScaleOffsetQuantizeParamsWrapper& rhs)
    : qnn_quantize_param_{rhs.qnn_quantize_param_},
      scale_offsets_{rhs.scale_offsets_} {
  qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset =
      scale_offsets_.data();
}

AxisScaleOffsetQuantizeParamsWrapper::AxisScaleOffsetQuantizeParamsWrapper(
    AxisScaleOffsetQuantizeParamsWrapper&& rhs)
    : qnn_quantize_param_{rhs.qnn_quantize_param_},
      scale_offsets_{std::move(rhs.scale_offsets_)} {
  qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset =
      scale_offsets_.data();
}

void AxisScaleOffsetQuantizeParamsWrapper::CloneTo(Qnn_QuantizeParams_t& dst) {
  dst = qnn_quantize_param_;
}

std::int32_t AxisScaleOffsetQuantizeParamsWrapper::GetAxis() const {
  return qnn_quantize_param_.axisScaleOffsetEncoding.axis;
}

void AxisScaleOffsetQuantizeParamsWrapper::SetAxis(const std::int32_t axis) {
  qnn_quantize_param_.axisScaleOffsetEncoding.axis = axis;
}

void AxisScaleOffsetQuantizeParamsWrapper::GetScales(
    std::vector<float>& scales) const {
  scales.clear();
  scales.reserve(scale_offsets_.size());
  for (size_t i = 0; i < scale_offsets_.size(); ++i) {
    scales.emplace_back(scale_offsets_[i].scale);
  }
}

void AxisScaleOffsetQuantizeParamsWrapper::GetZeroPoints(
    std::vector<std::int32_t>& zero_points) const {
  zero_points.clear();
  zero_points.reserve(scale_offsets_.size());
  for (size_t i = 0; i < scale_offsets_.size(); ++i) {
    zero_points.emplace_back(-1 * scale_offsets_[i].offset);
  }
}

}  // namespace qnn
