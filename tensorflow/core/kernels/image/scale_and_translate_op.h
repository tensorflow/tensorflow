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

#ifndef TENSORFLOW_CORE_KERNELS_IMAGESCALE_AND_TRANSLATE_OP_H_
#define TENSORFLOW_CORE_KERNELS_IMAGESCALE_AND_TRANSLATE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/image/sampling_kernels.h"

namespace tensorflow {
namespace functor {

// The scale and translate op works by scaling and translating the row and
// column dimensions separately.
// When scaling and translating the rows the set of input pixels and kernel
// weights used to compute a given output pixel within a row is constant across
// rows and can thus be precomputed and reused for every row. Similarly for the
// columns. This precomputed data structure is called a 'span'.

// To compute the gradient we use the spans computed on the forward pass and
// essentially reverse them: we record for each input pixel which output
// pixels it contributes to. This means that the forward and backward passes
// use the same core algorithm, only the spans are computed differently.

// A pre-computed span of pixels along a single dimension.
// The output pixel will be the weighted sum of pixels starting from start.
struct Spans {
  // The maximum span size of any output pixel.
  int span_size;
  // int32 tensor of size [output_dim].
  Tensor starts;
  // float tensor of size [output_dim, span_size].
  // The output pixel at x is computed as:
  //   dot_product(input[starts[x]:starts[x]+span_size], weights[x]).
  Tensor weights;
};

// Gather spans in both dimensions.
// row_span_size, row_starts and row_weights correspond to the variables in
// the row Spans data structure, similarly for col_span_size etc.
// intermediate_buffer is a Tensor used to store the result of the
// resize in the column dimension and is of size:
//    [batch_size, input_height, output_width, channels]
template <typename Device, typename T>
struct GatherSpans {
  void operator()(const Device& d, int row_span_size,
                  typename TTypes<int32, 1>::ConstTensor row_starts,
                  typename TTypes<float, 1>::ConstTensor row_weights,
                  int col_span_size,
                  typename TTypes<int32, 1>::ConstTensor col_starts,
                  typename TTypes<float, 1>::ConstTensor col_weights,
                  typename TTypes<T, 4>::ConstTensor input_images,
                  typename TTypes<float, 4>::Tensor intermediate_buffer,
                  typename TTypes<float, 4>::Tensor output_images);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_IMAGESCALE_AND_TRANSLATE_OP_H_
