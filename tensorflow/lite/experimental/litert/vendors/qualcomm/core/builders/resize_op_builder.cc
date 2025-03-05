// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/resize_op_builder.h"

#include <cstddef>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputIndex = 0;

std::vector<OpWrapper> BuildResizeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const char* op_type,
    const char* align_corners_param, const char* half_pixel_centers_param,
    const bool align_corners, const bool half_pixel_centers) {
  std::vector<OpWrapper> res;

  auto& resize_op = CreateOpWrapper(res, op_type);
  resize_op.AddInputTensor(inputs[kInputIndex]);
  resize_op.AddOutputTensor(outputs[kOutputIndex]);
  resize_op.AddScalarParam<bool>(align_corners_param, align_corners);
  resize_op.AddScalarParam<bool>(half_pixel_centers_param, half_pixel_centers);

  return res;
}
}  // namespace

std::vector<OpWrapper> BuildResizeBilinearOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool align_corners,
    const bool half_pixel_centers) {
  return BuildResizeOp(tensor_pool, inputs, outputs, QNN_OP_RESIZE_BILINEAR,
                       QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS,
                       QNN_OP_RESIZE_BILINEAR_PARAM_HALF_PIXEL_CENTERS,
                       align_corners, half_pixel_centers);
}

std::vector<OpWrapper> BuildResizeNearestOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool align_corners,
    const bool half_pixel_centers) {
  return BuildResizeOp(tensor_pool, inputs, outputs,
                       QNN_OP_RESIZE_NEAREST_NEIGHBOR,
                       QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_ALIGN_CORNERS,
                       QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_HALF_PIXEL_CENTERS,
                       align_corners, half_pixel_centers);
}

}  // namespace qnn
