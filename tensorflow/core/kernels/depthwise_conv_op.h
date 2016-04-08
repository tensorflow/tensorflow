/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

struct DepthwiseArgs {
  // Input layer dimensions
  int batch;
  int in_rows;
  int in_cols;
  int in_depth;
  int filter_rows;
  int filter_cols;
  int depth_multiplier;
  int stride;
  int pad_rows;
  int pad_cols;

  // Output layer dimensions
  int out_rows;
  int out_cols;
  int out_depth;

  DepthwiseArgs()
      : batch(0),
        in_rows(0),
        in_cols(0),
        in_depth(0),
        filter_rows(0),
        filter_cols(0),
        depth_multiplier(0),
        stride(0),
        pad_rows(0),
        pad_cols(0),
        out_rows(0),
        out_cols(0),
        out_depth(0) {}
};

}  // namespace tensorflow

namespace tensorflow {
namespace functor {

// Pads 'filter' to vector-register boundary along its inner dimension:
//   filter_inner_dim_size = in_depth * depth_multiplier
// Requires 'filter' to have the following storage order:
//   [filter_rows, filter_cols, in_depth, depth_multiplier]
// Returns zero-padded filter in 'padded_filter'.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   So we have a total of 3 * 2 = 6 filters, each of spatial size 2 x 2.
//
//   filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, u1, v1] [w1, x1, y1, z1]
//     [u2, v2, w2, x2] [y2, z2, u3, v3] [w3, x3, y3, z3]
//
//   padded_filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, 0, 0] [u1, v1, w1, x1] [y1, z1, 0, 0]
//     [u2, v2, w2, x2] [y2, z2, 0, 0] [u3, v3, w3, x3] [y3, z3, 0, 0]

template <typename T>
struct DepthwiseFilterPadOp {
  void operator()(const DepthwiseArgs& args, const T* filter,
                  T* padded_filter) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    // Calculate vectorized and scalar lengths of filter's inner dimension.
    const int64 filter_inner_dim_size = args.out_depth;
    const int64 vectorized_size =
        (filter_inner_dim_size / kPacketSize) * kPacketSize;
    const int64 scalar_size = filter_inner_dim_size - vectorized_size;
    // Calculate required padding and padded output buffer stride.
    const int64 pad_size = scalar_size > 0 ? kPacketSize - scalar_size : 0;
    const int64 padded_filter_stride = vectorized_size + kPacketSize;

    const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
    for (int64 i = 0; i < filter_spatial_size; ++i) {
      const int64 input_base = i * filter_inner_dim_size;
      const int64 output_base = i * padded_filter_stride;
      // Write vectorized length of filter's inner dimension to output.
      for (int64 j = 0; j < vectorized_size; j += kPacketSize) {
        const auto v = Eigen::internal::ploadu<Packet>(filter + input_base + j);
        Eigen::internal::pstoreu<T>(padded_filter + output_base + j, v);
      }
      // Write scalar length of filter's inner dimension to output.
      for (int64 j = 0; j < scalar_size; ++j) {
        padded_filter[output_base + vectorized_size + j] =
            filter[input_base + vectorized_size + j];
      }
      // Pad the remainder of output to vector-register boundary.
      for (int64 j = 0; j < pad_size; ++j) {
        padded_filter[output_base + vectorized_size + scalar_size + j] = 0;
      }
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_H_
