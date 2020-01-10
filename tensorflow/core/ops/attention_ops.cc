#include "tensorflow/core/framework/op.h"

namespace tensorflow {

// Tout = extract_glimpse(Tin, size, offsets) extract the glimpse of size size
// centered at location offsets from the input tensor Tin
//
// REQUIRES: Tin.dims() == 4
//
REGISTER_OP("ExtractGlimpse")
    .Input("input: float")
    .Input("size: int32")
    .Input("offsets: float")
    .Output("glimpse: float")
    .Attr("centered: bool = true")
    .Attr("normalized: bool = true")
    .Attr("uniform_noise: bool = true")
    .Doc(R"doc(
Extracts a glimpse from the input tensor.

Returns a set of windows called glimpses extracted at location `offsets`
from the input tensor. If the windows only partially overlaps the inputs, the
non overlapping areas will be filled with random noise.

The result is a 4-D tensor of shape `[batch_size, glimpse_height,
glimpse_width, channels]`. The channels and batch dimensions are the same as that
of the input tensor. The height and width of the output windows are
specified in the `size` parameter.

The argument `normalized` and `centered` controls how the windows are built:
* If the coordinates are normalized but not centered, 0.0 and 1.0
  correspond to the minimum and maximum of each height and width dimension.
* If the coordinates are both normalized and centered, they range from -1.0 to
  1.0. The coordinates (-1.0, -1.0) correspond to the upper left corner, the
  lower right corner is located at  (1.0, 1.0) and the center is at (0, 0).
* If the coordinates are not normalized they are interpreted as numbers of pixels.

input: A 4-D float tensor of shape `[batch_size, height, width, channels]`.
size: A 1-D tensor of 2 elements containing the size of the glimpses to extract.
  The glimpse height must be specified first, following by the glimpse width.
offsets: A 2-D integer tensor of shape `[batch_size, 2]` containing the x, y
  locations of the center of each window.
glimpse: A tensor representing the glimpses `[batch_size, glimpse_height,
  glimpse_width, channels]`.
centered: indicates if the offset coordinates are centered relative to
  the image, in which case the (0, 0) offset is relative to the center of the
  input images. If false, the (0,0) offset corresponds to the upper left corner
  of the input images.
normalized: indicates if the offset coordinates are normalized.
uniform_noise: indicates if the noise should be generated using a
  uniform distribution or a gaussian distribution.
)doc");

}  // namespace tensorflow
