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

#ifndef TENSORFLOW_CORE_KERNELS_WINOGRAD_TRANSFORM_H_
#define TENSORFLOW_CORE_KERNELS_WINOGRAD_TRANSFORM_H_

#include "tensorflow/core/kernels/deep_conv2d.h"

namespace tensorflow {

// Winograd DeepConv2DTransform implementation for 3x3 filters.
// Details:
// *) Arithmetic complexity of computations: Shmuel Winograd
// *) Fast Algorithms for Convolutional Neural Networks: Lavin, Gray

template <typename T>
class WinogradTransform : public DeepConv2DTransform<T> {
 public:
  typedef typename DeepConv2DTransform<T>::Shape Shape;

  WinogradTransform()
      : filter_shape_(3, 3), input_shape_(4, 4), output_shape_(2, 2) {}

  virtual void GetFilterTransformMatrix(const int64 rows, const int64 cols,
                                        T* transform_matrix) const;

  virtual void GetInputTransformMatrix(const int64 rows, const int64 cols,
                                       T* transform_matrix) const;

  virtual void GetOutputTransformMatrix(const int64 rows, const int64 cols,
                                        T* transform_matrix) const;

  virtual const Shape& filter_shape() const { return filter_shape_; }
  virtual const Shape& input_shape() const { return input_shape_; }
  virtual const Shape& output_shape() const { return output_shape_; }

 private:
  const Shape filter_shape_;
  const Shape input_shape_;
  const Shape output_shape_;
};

// The filter transform matrix is the kronecker product 'M * M' of the
// following matrix 'M':
//
//   [ 1    0   0   ]
//   [ 1/2  1/2 1/2 ]
//   [ 1/2 -1/2 1/2 ]
//   [ 0    0   1   ]
//
// The data layout of 'transform_matrix':
//   [input_tile_spatial_size, filter_spatial_size]
//
template <typename T>
void WinogradTransform<T>::GetFilterTransformMatrix(const int64 rows,
                                                    const int64 cols,
                                                    T* transform_matrix) const {
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  memset(transform_matrix, 0, sizeof(T) * rows * cols);

  // Sub matrix [0,0]
  transform_matrix[0 * cols + 0] = T(1.0);

  transform_matrix[1 * cols + 0] = T(0.5);
  transform_matrix[1 * cols + 1] = T(0.5);
  transform_matrix[1 * cols + 2] = T(0.5);

  transform_matrix[2 * cols + 0] = T(0.5);
  transform_matrix[2 * cols + 1] = T(-0.5);
  transform_matrix[2 * cols + 2] = T(0.5);

  transform_matrix[3 * cols + 2] = T(1.0);

  // Sub matrix [1,0]
  transform_matrix[4 * cols + 0] = T(0.5);

  transform_matrix[5 * cols + 0] = T(0.25);
  transform_matrix[5 * cols + 1] = T(0.25);
  transform_matrix[5 * cols + 2] = T(0.25);

  transform_matrix[6 * cols + 0] = T(0.25);
  transform_matrix[6 * cols + 1] = T(-0.25);
  transform_matrix[6 * cols + 2] = T(0.25);

  transform_matrix[7 * cols + 2] = T(0.5);

  // Sub matrix [1,1]
  transform_matrix[4 * cols + 3] = T(0.5);

  transform_matrix[5 * cols + 3] = T(0.25);
  transform_matrix[5 * cols + 4] = T(0.25);
  transform_matrix[5 * cols + 5] = T(0.25);

  transform_matrix[6 * cols + 3] = T(0.25);
  transform_matrix[6 * cols + 4] = T(-0.25);
  transform_matrix[6 * cols + 5] = T(0.25);

  transform_matrix[7 * cols + 5] = T(0.5);

  // Sub matrix [1,2]
  transform_matrix[4 * cols + 6] = T(0.5);

  transform_matrix[5 * cols + 6] = T(0.25);
  transform_matrix[5 * cols + 7] = T(0.25);
  transform_matrix[5 * cols + 8] = T(0.25);

  transform_matrix[6 * cols + 6] = T(0.25);
  transform_matrix[6 * cols + 7] = T(-0.25);
  transform_matrix[6 * cols + 8] = T(0.25);

  transform_matrix[7 * cols + 8] = T(0.5);

  // Sub matrix [2,0]
  transform_matrix[8 * cols + 0] = T(0.5);

  transform_matrix[9 * cols + 0] = T(0.25);
  transform_matrix[9 * cols + 1] = T(0.25);
  transform_matrix[9 * cols + 2] = T(0.25);

  transform_matrix[10 * cols + 0] = T(0.25);
  transform_matrix[10 * cols + 1] = T(-0.25);
  transform_matrix[10 * cols + 2] = T(0.25);

  transform_matrix[11 * cols + 2] = T(0.5);

  // Sub matrix [2,1]
  transform_matrix[8 * cols + 3] = T(-0.5);

  transform_matrix[9 * cols + 3] = T(-0.25);
  transform_matrix[9 * cols + 4] = T(-0.25);
  transform_matrix[9 * cols + 5] = T(-0.25);

  transform_matrix[10 * cols + 3] = T(-0.25);
  transform_matrix[10 * cols + 4] = T(0.25);
  transform_matrix[10 * cols + 5] = T(-0.25);

  transform_matrix[11 * cols + 5] = T(-0.5);

  // Sub matrix [2,2]
  transform_matrix[8 * cols + 6] = T(0.5);

  transform_matrix[9 * cols + 6] = T(0.25);
  transform_matrix[9 * cols + 7] = T(0.25);
  transform_matrix[9 * cols + 8] = T(0.25);

  transform_matrix[10 * cols + 6] = T(0.25);
  transform_matrix[10 * cols + 7] = T(-0.25);
  transform_matrix[10 * cols + 8] = T(0.25);

  transform_matrix[11 * cols + 8] = T(0.5);

  // Sub matrix [3,2]
  transform_matrix[12 * cols + 6] = T(1.0);

  transform_matrix[13 * cols + 6] = T(0.5);
  transform_matrix[13 * cols + 7] = T(0.5);
  transform_matrix[13 * cols + 8] = T(0.5);

  transform_matrix[14 * cols + 6] = T(0.5);
  transform_matrix[14 * cols + 7] = T(-0.5);
  transform_matrix[14 * cols + 8] = T(0.5);

  transform_matrix[15 * cols + 8] = T(1.0);
}

// The input transform matrix is the kronecker product 'M * M' of the
// following matrix 'M':
//
//   [1   0  -1   0]
//   [0   1   1   0]
//   [0  -1   1   0]
//   [0   1   0  -1]
//
// Data layout of 'transform_matrix':
//   [tile_spatial_size, tile_spatial_size]
//
template <typename T>
void WinogradTransform<T>::GetInputTransformMatrix(const int64 rows,
                                                   const int64 cols,
                                                   T* transform_matrix) const {
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  memset(transform_matrix, 0, sizeof(T) * rows * cols);

  // Sub matrix [0,0]
  transform_matrix[0 * cols + 0] = T(1.0);
  transform_matrix[0 * cols + 2] = T(-1.0);

  transform_matrix[1 * cols + 1] = T(1.0);
  transform_matrix[1 * cols + 2] = T(1.0);

  transform_matrix[2 * cols + 1] = T(-1.0);
  transform_matrix[2 * cols + 2] = T(1.0);

  transform_matrix[3 * cols + 1] = T(1.0);
  transform_matrix[3 * cols + 3] = T(-1.0);

  // Sub matrix [0,2]
  transform_matrix[0 * cols + 8] = T(-1.0);
  transform_matrix[0 * cols + 10] = T(1.0);

  transform_matrix[1 * cols + 9] = T(-1.0);
  transform_matrix[1 * cols + 10] = T(-1.0);

  transform_matrix[2 * cols + 9] = T(1.0);
  transform_matrix[2 * cols + 10] = T(-1.0);

  transform_matrix[3 * cols + 9] = T(-1.0);
  transform_matrix[3 * cols + 11] = T(1.0);

  // Sub matrix [1,1]
  transform_matrix[4 * cols + 4] = T(1.0);
  transform_matrix[4 * cols + 6] = T(-1.0);

  transform_matrix[5 * cols + 5] = T(1.0);
  transform_matrix[5 * cols + 6] = T(1.0);

  transform_matrix[6 * cols + 5] = T(-1.0);
  transform_matrix[6 * cols + 6] = T(1.0);

  transform_matrix[7 * cols + 5] = T(1.0);
  transform_matrix[7 * cols + 7] = T(-1.0);

  // Sub matrix [1,2]
  transform_matrix[4 * cols + 8] = T(1.0);
  transform_matrix[4 * cols + 10] = T(-1.0);

  transform_matrix[5 * cols + 9] = T(1.0);
  transform_matrix[5 * cols + 10] = T(1.0);

  transform_matrix[6 * cols + 9] = T(-1.0);
  transform_matrix[6 * cols + 10] = T(1.0);

  transform_matrix[7 * cols + 9] = T(1.0);
  transform_matrix[7 * cols + 11] = T(-1.0);

  // Sub matrix [2,1]
  transform_matrix[8 * cols + 4] = T(-1.0);
  transform_matrix[8 * cols + 6] = T(1.0);

  transform_matrix[9 * cols + 5] = T(-1.0);
  transform_matrix[9 * cols + 6] = T(-1.0);

  transform_matrix[10 * cols + 5] = T(1.0);
  transform_matrix[10 * cols + 6] = T(-1.0);

  transform_matrix[11 * cols + 5] = T(-1.0);
  transform_matrix[11 * cols + 7] = T(1.0);

  // Sub matrix [2,2]
  transform_matrix[8 * cols + 8] = T(1.0);
  transform_matrix[8 * cols + 10] = T(-1.0);

  transform_matrix[9 * cols + 9] = T(1.0);
  transform_matrix[9 * cols + 10] = T(1.0);

  transform_matrix[10 * cols + 9] = T(-1.0);
  transform_matrix[10 * cols + 10] = T(1.0);

  transform_matrix[11 * cols + 9] = T(1.0);
  transform_matrix[11 * cols + 11] = T(-1.0);

  // Sub matrix [3,1]
  transform_matrix[12 * cols + 4] = T(1.0);
  transform_matrix[12 * cols + 6] = T(-1.0);

  transform_matrix[13 * cols + 5] = T(1.0);
  transform_matrix[13 * cols + 6] = T(1.0);

  transform_matrix[14 * cols + 5] = T(-1.0);
  transform_matrix[14 * cols + 6] = T(1.0);

  transform_matrix[15 * cols + 5] = T(1.0);
  transform_matrix[15 * cols + 7] = T(-1.0);

  // Sub matrix [3,3]
  transform_matrix[12 * cols + 12] = T(-1.0);
  transform_matrix[12 * cols + 14] = T(1.0);

  transform_matrix[13 * cols + 13] = T(-1.0);
  transform_matrix[13 * cols + 14] = T(-1.0);

  transform_matrix[14 * cols + 13] = T(1.0);
  transform_matrix[14 * cols + 14] = T(-1.0);

  transform_matrix[15 * cols + 13] = T(-1.0);
  transform_matrix[15 * cols + 15] = T(1.0);
};

// The output transform matrix is the kronecker product 'M * M' of the
// following matrix 'M':
//
//   [1  1  1  0]
//   [0  1 -1 -1]
//
// Data layout of 'transform_matrix':
//   [out_tile_spatial_size, tile_spatial_size]
//
template <typename T>
void WinogradTransform<T>::GetOutputTransformMatrix(const int64 rows,
                                                    const int64 cols,
                                                    T* transform_matrix) const {
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  memset(transform_matrix, 0, sizeof(T) * rows * cols);

  // Sub matrix [0,0]
  transform_matrix[0 * cols + 0] = T(1.0);
  transform_matrix[0 * cols + 1] = T(1.0);
  transform_matrix[0 * cols + 2] = T(1.0);

  transform_matrix[1 * cols + 1] = T(1.0);
  transform_matrix[1 * cols + 2] = T(-1.0);
  transform_matrix[1 * cols + 3] = T(-1.0);

  // Sub matrix [0,1]
  transform_matrix[0 * cols + 4] = T(1.0);
  transform_matrix[0 * cols + 5] = T(1.0);
  transform_matrix[0 * cols + 6] = T(1.0);

  transform_matrix[1 * cols + 5] = T(1.0);
  transform_matrix[1 * cols + 6] = T(-1.0);
  transform_matrix[1 * cols + 7] = T(-1.0);

  // Sub matrix [0,2]
  transform_matrix[0 * cols + 8] = T(1.0);
  transform_matrix[0 * cols + 9] = T(1.0);
  transform_matrix[0 * cols + 10] = T(1.0);

  transform_matrix[1 * cols + 9] = T(1.0);
  transform_matrix[1 * cols + 10] = T(-1.0);
  transform_matrix[1 * cols + 11] = T(-1.0);

  // Sub matrix [1,1]
  transform_matrix[2 * cols + 4] = T(1.0);
  transform_matrix[2 * cols + 5] = T(1.0);
  transform_matrix[2 * cols + 6] = T(1.0);

  transform_matrix[3 * cols + 5] = T(1.0);
  transform_matrix[3 * cols + 6] = T(-1.0);
  transform_matrix[3 * cols + 7] = T(-1.0);

  // Sub matrix [1,2]
  transform_matrix[2 * cols + 8] = T(-1.0);
  transform_matrix[2 * cols + 9] = T(-1.0);
  transform_matrix[2 * cols + 10] = T(-1.0);

  transform_matrix[3 * cols + 9] = T(-1.0);
  transform_matrix[3 * cols + 10] = T(1.0);
  transform_matrix[3 * cols + 11] = T(1.0);

  // Sub matrix [1,3]
  transform_matrix[2 * cols + 12] = T(-1.0);
  transform_matrix[2 * cols + 13] = T(-1.0);
  transform_matrix[2 * cols + 14] = T(-1.0);

  transform_matrix[3 * cols + 13] = T(-1.0);
  transform_matrix[3 * cols + 14] = T(1.0);
  transform_matrix[3 * cols + 15] = T(1.0);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_WINOGRAD_TRANSFORM_H_
