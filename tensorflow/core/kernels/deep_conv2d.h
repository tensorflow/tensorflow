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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DEEP_CONV2D_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DEEP_CONV2D_H_

#include "tensorflow/core/framework/types.h"

namespace tensorflow {

class OpKernelContext;

// DeepConv2D is a Conv2D implementation specialized for deep (i.e. large
// in_depth * out_depth product) convolutions (see deep_conv2d.cc for details).

// DeepConv2DTransform is an interface for implementing transforms for
// DeepConv2D. Implementations must specify transform matrices and
// input/output/filter shapes. DeepConv2d computes:
//
//   y = C[Ad * Bg]
//
//   C: output transform matrix
//   A: input data transform matrix
//   B: filter transform matrix
//   d: vectorized 2D data tile
//   g: vectorized 2D filter tile
//   y: vectorized 2D output tile

template <typename T>
class DeepConv2DTransform {
 public:
  virtual ~DeepConv2DTransform() {}

  virtual void GetFilterTransformMatrix(const int64 rows, const int64 cols,
                                        T* transform_matrix) const = 0;

  virtual void GetInputTransformMatrix(const int64 rows, const int64 cols,
                                       T* transform_matrix) const = 0;

  virtual void GetOutputTransformMatrix(const int64 rows, const int64 cols,
                                        T* transform_matrix) const = 0;

  struct Shape {
    Shape(int64 r, int64 c) : rows(r), cols(c) {}
    int64 rows;
    int64 cols;
  };

  virtual const Shape& filter_shape() const = 0;
  virtual const Shape& input_shape() const = 0;
  virtual const Shape& output_shape() const = 0;
};

// Conv2D arguments used by DeepConv2D implementation.
struct Conv2DArgs {
  // Input layer dimensions
  int batch;
  int in_rows;
  int in_cols;
  int in_depth;
  int filter_rows;
  int filter_cols;
  int pad_rows;
  int pad_cols;

  // Output layer dimensions
  int out_rows;
  int out_cols;
  int out_depth;

  Conv2DArgs()
      : batch(0),
        in_rows(0),
        in_cols(0),
        in_depth(0),
        filter_rows(0),
        filter_cols(0),
        pad_rows(0),
        pad_cols(0),
        out_rows(0),
        out_cols(0),
        out_depth(0) {}
};

// Returns true if convolution operation specified by function arguments
// can use DeepConv2D implementation, and false otherwise.
// May return false based on parameters, cost, or whether feature is disabled.
bool CanUseDeepConv2D(int stride_rows, int stride_cols, int filter_rows,
                      int filter_cols, int in_depth, int out_depth,
                      int out_rows, int out_cols);

namespace functor {

// Calls DeepConv2D implementation (see deep_conv2d.cc for details).
template <typename Device, typename T>
struct DeepConv2D {
  void operator()(OpKernelContext* ctx, const Conv2DArgs& args, const T* input,
                  const T* filter, T* output);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DEEP_CONV2D_H_
