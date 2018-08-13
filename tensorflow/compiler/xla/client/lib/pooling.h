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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_POOLING_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_POOLING_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace xla {

// Tensor format for reduce window operations.
class TensorFormat {
 public:
  TensorFormat(int batch_dimension, int feature_dimension,
               tensorflow::gtl::ArraySlice<int64> spatial_dimensions)
      : batch_dimension_(batch_dimension),
        feature_dimension_(feature_dimension),
        spatial_dimensions_(spatial_dimensions.begin(),
                            spatial_dimensions.end()) {}

  int batch_dimension() const { return batch_dimension_; }

  int feature_dimension() const { return feature_dimension_; }

  int spatial_dimension(int dim) const { return spatial_dimensions_[dim]; }

  int num_spatial_dims() const { return spatial_dimensions_.size(); }

 private:
  // The number of the dimension that represents the batch.
  int batch_dimension_;
  // The number of the dimension that represents the features.
  int feature_dimension_;
  // The dimension numbers for the spatial dimensions.
  tensorflow::gtl::InlinedVector<int, 4> spatial_dimensions_;
};

// Computes the max pool of 'operand'.
XlaOp MaxPool(XlaOp operand, tensorflow::gtl::ArraySlice<int64> kernel_size,
              tensorflow::gtl::ArraySlice<int64> stride, Padding padding,
              const TensorFormat& data_format);

// Computes the average pool of 'operand'.
XlaOp AvgPool(XlaOp operand, tensorflow::gtl::ArraySlice<int64> kernel_size,
              tensorflow::gtl::ArraySlice<int64> stride,
              tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
              const TensorFormat& data_format,
              const bool counts_include_padding);

// Returns the list of low and high padding elements in each spatial dimension
// for the given 'padding' specification.
std::vector<std::pair<int64, int64>> MakeSpatialPadding(
    tensorflow::gtl::ArraySlice<int64> input_size,
    tensorflow::gtl::ArraySlice<int64> kernel_size,
    tensorflow::gtl::ArraySlice<int64> stride, Padding padding,
    const TensorFormat& data_format);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_POOLING_H_
