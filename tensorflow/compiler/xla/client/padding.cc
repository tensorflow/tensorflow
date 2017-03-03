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

#include "tensorflow/compiler/xla/client/padding.h"

#include <algorithm>

#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

std::vector<std::pair<int64, int64>> MakePadding(
    tensorflow::gtl::ArraySlice<int64> input_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding) {
  CHECK_EQ(input_dimensions.size(), window_dimensions.size());
  CHECK_EQ(input_dimensions.size(), window_strides.size());
  std::vector<std::pair<int64, int64>> low_high_padding;
  switch (padding) {
    case Padding::kValid:
      low_high_padding.resize(window_dimensions.size(), {0, 0});
      return low_high_padding;

    case Padding::kSame:
      for (tensorflow::gtl::ArraySlice<int64>::size_type i = 0;
           i < input_dimensions.size(); ++i) {
        int64 input_dimension = input_dimensions[i];
        int64 window_dimension = window_dimensions[i];
        int64 window_stride = window_strides[i];
        // We follow the same convention as in Tensorflow, such that
        // output dimension := ceil(input_dimension / window_stride).
        // See tensorflow/tensorflow/python/ops/nn.py
        // for the reference. See also tensorflow/core/kernels/ops_util.cc
        // for the part where we avoid negative padding using max(0, x).
        //
        //
        // For an odd sized window dimension 2N+1 with stride 1, the middle
        // element is always inside the base area, so we can see it as N + 1 +
        // N elements. In the example below, we have a kernel of size
        // 2*3+1=7 so that the center element is 4 with 123 to the
        // left and 567 to the right.
        //
        //  base area:           ------------------------
        //  kernel at left:   1234567
        //  kernel at right:                         1234567
        //
        // We can see visually here that we need to pad the base area
        // by 3 on each side:
        //
        //  padded base area: 000------------------------000
        //
        // For an even number 2N, there are two options:
        //
        // *** Option A
        //
        // We view 2N as (N - 1) + 1 + N, so for N=3 we have 12 to the
        // left, 3 is the center and 456 is to the right, like this:
        //
        //  base area:           ------------------------
        //  kernel at left:    123456
        //  kernel at right:                          123456
        //  padded base area:  00------------------------000
        //
        // Note how we pad by one more to the right than to the left.
        //
        // *** Option B
        //
        // We view 2N as N + 1 + (N - 1), so for N=3 we have 123 to
        // the left, 4 is the center and 56 is to the right, like
        // this:
        //
        //  base area:           ------------------------
        //  kernel at left:   123456
        //  kernel at right:                         123456
        //  padded base area: 000------------------------00
        //
        // The choice here is arbitrary. We choose option A as this is
        // what DistBelief and Tensorflow do.
        //
        // When the stride is greater than 1, the output size is smaller than
        // the input base size. The base area is padded such that the last
        // window fully fits in the padded base area, and the padding amount is
        // evenly divided between the left and the right (or 1 more on the right
        // if odd size padding is required). The example below shows the
        // required padding when the base size is 10, the kernel size is 5, and
        // the stride is 3. In this example, the output size is 4.
        //
        // base area:           ----------
        // 1'st kernel:       12345
        // 2'nd kernel:          12345
        // 3'rd kernel:             12345
        // 4'th kernel:                12345
        // padded base area:  00----------00
        int64 output_dimension =
            tensorflow::MathUtil::CeilOfRatio(input_dimension, window_stride);
        int64 padding_size =
            std::max<int64>((output_dimension - 1) * window_stride +
                                window_dimension - input_dimension,
                            0);
        low_high_padding.emplace_back(
            tensorflow::MathUtil::FloorOfRatio(padding_size, 2ll),
            tensorflow::MathUtil::CeilOfRatio(padding_size, 2ll));
      }
      break;
  }

  return low_high_padding;
}

}  // namespace xla
