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

#include <vector>

#include "absl/status/status.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/image_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace ops {
namespace {

REGISTER_NO_GRADIENT_OP("NonMaxSuppression");
REGISTER_NO_GRADIENT_OP("NonMaxSuppressionV2");
REGISTER_NO_GRADIENT_OP("NonMaxSuppressionV3");
REGISTER_NO_GRADIENT_OP("NonMaxSuppressionV4");
REGISTER_NO_GRADIENT_OP("NonMaxSuppressionV5");

absl::Status ResizeNearestNeighborGradHelper(
    const Scope& scope, const Operation& op,
    const std::vector<Output>& grad_inputs, std::vector<Output>* grad_outputs) {
  bool align_corners;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "align_corners", &align_corners));
  bool half_pixel_centers;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "half_pixel_centers",
                                 &half_pixel_centers));
  // The internal gradient implementation needs the shape of the input image.
  // x_shape = shape(x)[1:3]
  //         = slice(shape(x), {1}, {3 - 1})
  auto x_shape = Slice(scope, Shape(scope, op.input(0)), {1}, {2});
  grad_outputs->push_back(internal::ResizeNearestNeighborGrad(
      scope, grad_inputs[0], x_shape,
      internal::ResizeNearestNeighborGrad::AlignCorners(align_corners)
          .HalfPixelCenters(half_pixel_centers)));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ResizeNearestNeighbor", ResizeNearestNeighborGradHelper);

absl::Status ResizeBilinearGradHelper(const Scope& scope, const Operation& op,
                                      const std::vector<Output>& grad_inputs,
                                      std::vector<Output>* grad_outputs) {
  bool align_corners;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "align_corners", &align_corners));
  bool half_pixel_centers;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "half_pixel_centers",
                                 &half_pixel_centers));
  grad_outputs->push_back(internal::ResizeBilinearGrad(
      scope, grad_inputs[0], op.input(0),
      internal::ResizeBilinearGrad::AlignCorners(align_corners)
          .HalfPixelCenters(half_pixel_centers)));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ResizeBilinear", ResizeBilinearGradHelper);

absl::Status ResizeBicubicGradHelper(const Scope& scope, const Operation& op,
                                     const std::vector<Output>& grad_inputs,
                                     std::vector<Output>* grad_outputs) {
  bool align_corners;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "align_corners", &align_corners));
  bool half_pixel_centers;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "half_pixel_centers",
                                 &half_pixel_centers));

  grad_outputs->push_back(internal::ResizeBicubicGrad(
      scope, grad_inputs[0], op.input(0),
      internal::ResizeBicubicGrad::AlignCorners(align_corners)
          .HalfPixelCenters(half_pixel_centers)));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ResizeBicubic", ResizeBicubicGradHelper);

absl::Status ScaleAndTranslateGradHelper(const Scope& scope,
                                         const Operation& op,
                                         const std::vector<Output>& grad_inputs,
                                         std::vector<Output>* grad_outputs) {
  string kernel_type;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "kernel_type", &kernel_type));
  bool antialias;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "antialias", &antialias));
  grad_outputs->push_back(internal::ScaleAndTranslateGrad(
      scope, grad_inputs[0], op.input(0), op.input(2), op.input(3),
      internal::ScaleAndTranslateGrad::KernelType(kernel_type)
          .Antialias(antialias)));

  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_GRADIENT_OP("ScaleAndTranslate", ScaleAndTranslateGradHelper);

absl::Status CropAndResizeGradHelper(const Scope& scope, const Operation& op,
                                     const std::vector<Output>& grad_inputs,
                                     std::vector<Output>* grad_outputs) {
  DataType input_type;
  string method;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "method", &method));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "T", &input_type));
  auto image_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(CropAndResizeGradImage(
      scope, grad_inputs[0], op.input(1), op.input(2), image_shape, input_type,
      CropAndResizeGradImage::Method(method)));
  grad_outputs->push_back(CropAndResizeGradBoxes(
      scope, grad_inputs[0], op.input(0), op.input(1), op.input(2)));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_GRADIENT_OP("CropAndResize", CropAndResizeGradHelper);
}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
