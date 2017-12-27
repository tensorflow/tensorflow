/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("SingleImageRandomDotStereograms")
    .Attr("T: {double,float,int64,int32}")
    .Input("depth_values: T")
    .Output("image: uint8")
    .Attr("hidden_surface_removal: bool = true")
    .Attr("convergence_dots_size: int = 8")
    .Attr("dots_per_inch: int = 72")
    .Attr("eye_separation: float = 2.5")
    .Attr("mu: float = .3333")
    .Attr("normalize: bool = true")
    .Attr("normalize_max: float = -100.0")
    .Attr("normalize_min: float = 100.0")
    .Attr("border_level: float = 0.0")
    .Attr("number_colors: int = 256")
    .Attr(
        "output_image_shape: shape = { dim {size:1024} dim {size: 768} dim "
        "{size: 1}}")
    .Attr("output_data_window: shape = { dim {size:1022} dim {size: 757}}")
    .SetShapeFn([](InferenceContext* c) {
      // Validate that the output_image_shape attr is correct.
      // NOTE: The output_image_shape is [X, Y, C]
      // while the output data is [Y, X, C] (or [H, W, C]).
      // As a result, by default the output_image_shape has the value
      // of [1024, 768, 1] but the output data will be [768, 1024, 1].
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("output_image_shape", &shape));
      ShapeHandle output_image_shape;
      TF_RETURN_IF_ERROR(
          c->MakeShapeFromPartialTensorShape(shape, &output_image_shape));
      DimensionHandle x_dim = c->Dim(output_image_shape, 0);
      DimensionHandle y_dim = c->Dim(output_image_shape, 1);
      DimensionHandle c_dim = c->Dim(output_image_shape, 2);

      int colors;
      TF_RETURN_IF_ERROR(c->GetAttr("number_colors", &colors));

      c->set_output(0, c->MakeShape({y_dim, x_dim, colors > 256? c->MakeDim(3) : c->MakeDim(1)}));
      return Status::OK();
    })
    .Doc(R"doc(
Outputs a single image random dot stereogram for export via encode_PNG/JPG OP.

Given the 2-D tensor 'depth_values' with encoded Z values, this operation will
encode 3-D data into a 2-D image.  The output of this Op is suitable for the
encode_PNG/JPG ops.  Be careful with image compression as this may corrupt the
encode 3-D data witin the image.

This Op is based upon:
'http://www.learningace.com/doc/4331582/b6ab058d1e206d68ab60e4e1ead2fe6e/sirds-paper'

Example use which outputs a SIRDS image as picture_out.png:
```python
img=[[1,2,3,3,2,1],
     [1,2,3,4,5,2],
     [1,2,3,4,5,3],
     [1,2,3,4,5,4],
     [6,5,4,4,5,5]]

session = tf.InteractiveSession()

sirds = single_image_random_dot_stereograms(img,convergence_dots_size=8,number_colors=256,normalize=True)

out = sirds.eval()

png = tf.image.encode_png(out).eval()

with open('picture_out.png', 'wb') as f:
    f.write(png)
```

depth_values: Z values of data to encode into 'output_data_window' window,
  lower values are further away {0.0 floor(far), 1.0 ceiling(near) after normalization}, must be 2-D tensor
hidden_surface_removal: Activate hidden surface removal
convergence_dots_size: Black dot size in pixels to help view converge image, drawn on bottom of image
dots_per_inch: Output device in dots/inch
eye_separation: Separation between eyes in inches
mu: Depth of field, Fraction of viewing distance (eg. 1/3 = .3333)
normalize: Normalize input data to [0.0, 1.0]
normalize_max: Fix MAX value for Normalization - if < MIN, autoscale
normalize_min: Fix MIN value for Normalization - if > MAX, autoscale
border_level: Value of border depth 0.0 {far} to 1.0 {near}
number_colors: 2 (Black & White),256 (grayscale), and Numbers > 256 (Full Color) are all that are supported currently
output_image_shape: Output size of returned image in X,Y, Channels 1-grayscale, 3 color (1024, 768, 1),
  channels will be updated to 3 if 'number_colors' > 256
output_data_window: Size of "DATA" window, must be equal to or smaller than 'output_image_shape', will be centered
  and use 'convergence_dots_size' for best fit to avoid overlap if possible

image:= A tensor of size 'output_image_shape' with the encloded 'depth_values'
)doc");

}  // namespace tensorflow
