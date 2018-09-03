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

#include "tensorflow/contrib/lite/tools/accuracy/ilsvrc/inception_preprocessing.h"

#include <memory>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace metrics {

namespace {
void CentralCropImage(const Scope& s, const tensorflow::Output& decoded_image,
                      double crop_fraction, tensorflow::Output* cropped_image) {
  auto image_dims = ops::Slice(s, ops::Shape(s, decoded_image), {0}, {2});
  auto height_width = ops::Cast(s, image_dims, DT_DOUBLE);
  auto cropped_begin = ops::Div(
      s, ops::Sub(s, height_width, ops::Mul(s, height_width, crop_fraction)),
      2.0);
  auto bbox_begin = ops::Cast(s, cropped_begin, DT_INT32);
  auto bbox_size = ops::Sub(s, image_dims, ops::Mul(s, bbox_begin, 2));
  auto slice_begin = ops::Concat(s, {bbox_begin, Input({0})}, 0);
  auto slice_size = ops::Concat(s, {bbox_size, {-1}}, 0);
  *cropped_image = ops::Slice(s, decoded_image, slice_begin, slice_size);
}

}  // namespace

void InceptionPreprocessingStage::AddToGraph(const Scope& scope,
                                             const Input& input) {
  if (!scope.ok()) return;
  Scope s = scope.WithOpName(name());
  ops::DecodeJpeg::Attrs attrs;
  attrs.channels_ = 3;
  auto decoded_jpeg = ops::DecodeJpeg(s, input, attrs);
  tensorflow::Output cropped_image;
  CentralCropImage(s, decoded_jpeg, params_.cropping_fraction, &cropped_image);
  auto dims_expander = ops::ExpandDims(s, cropped_image, 0);
  auto resized_image = ops::ResizeBilinear(
      s, dims_expander,
      ops::Const(s.WithOpName("size"), {image_height_, image_width_}));
  if (is_quantized_) {
    this->stage_output_ =
        ops::Cast(s.WithOpName(output_name()), resized_image, DT_UINT8);
  } else {
    auto squeezed_image = ops::Squeeze(s, resized_image);
    auto normalized_image =
        ops::Div(s,
                 ops::Sub(s, squeezed_image,
                          {params_.input_means[0], params_.input_means[1],
                           params_.input_means[2]}),
                 {params_.scale});
    this->stage_output_ =
        ops::ExpandDims(s.WithOpName(output_name()), normalized_image, {0});
  }
}

}  // namespace metrics
}  // namespace tensorflow
