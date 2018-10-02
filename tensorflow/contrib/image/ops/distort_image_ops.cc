/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;

// --------------------------------------------------------------------------
REGISTER_OP("AdjustHsvInYiq")
    .Input("images: T")
    .Input("delta_h: float")
    .Input("scale_s: float")
    .Input("scale_v: float")
    .Output("output: T")
    .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
    .Doc(R"Doc(
Adjust the YIQ hue of one or more images.

`images` is a tensor of at least 3 dimensions.  The last dimension is
interpreted as channels, and must be three.

We used linear transformation described in:
 beesbuzz.biz/code/hsv_color_transforms.php
The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into YIQ space, rotated around the Y channel by
delta_h in radians, multiplying the chrominance channels (I, Q)  by scale_s,
multiplying all channels (Y, I, Q)  by scale_v, and then remapped back to RGB
colorspace. Each operation described above is a linear transformation.

images: Images to adjust.  At least 3-D.
delta_h: A float scale that represents the hue rotation amount, in radians.
         Although delta_h can be any float value.
scale_s: A float scale that represents the factor to multiply the saturation by.
         scale_s needs to be non-negative.
scale_v: A float scale that represents the factor to multiply the value by.
         scale_v needs to be non-negative.
output: The hsv-adjusted image or images. No clipping will be done in this op.
        The client can clip them using additional ops in their graph.
)Doc");

}  // namespace tensorflow
