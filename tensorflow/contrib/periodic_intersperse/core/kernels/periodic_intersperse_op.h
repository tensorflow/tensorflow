// =============================================================================
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef TENSORFLOW_KERNELS_PERIODICINTERSPERSE_OP_H_
#define TENSORFLOW_KERNELS_PERIODICINTERSPERSE_OP_H_

#include <cmath>
#include <type_traits>
#include <vector>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

template <class IndexVecT, class IndexT>
IndexT compute_input_index(
    IndexVecT& Y, const IndexT& i, const IndexVecT& X, const int& q,
    const std::vector<tensorflow::int64>& g,
    const std::vector<tensorflow::int64>& G, IndexT& result,
    std::vector<IndexT>& output_indices,
    const typename std::decay<decltype(Y.size())>::type& rank) {
  result = 0;
  output_indices.clear();

  // un-rasterize the output index
  {
    auto last_reduced_i = i;
    auto r = rank;
    do {
      --r;
      output_indices[r] = last_reduced_i % Y[r];
      last_reduced_i = (last_reduced_i - output_indices[r]) / Y[r];
    } while (r > 0);
  }

  // rasterize the input index
  {
    IndexT last_index_factor = 1;
    for (auto r = rank - 1;; --r) {
      IndexT index = 0;
      if (r != q)
        index = output_indices[r] / g[r];
      else {
        for (int qi = 0; qi < rank; ++qi) {
          if (qi == q) continue;
          index += G[qi] * (output_indices[qi] % g[qi]);
        }
        index *= Y[q];
        index += output_indices[r];
      }
      result += last_index_factor * index;
      last_index_factor *= X[r];
      if (r == 0) break;
    }
  }

  return result;
}

template <class T, class VecT>
void main_logic(tensorflow::OpKernelContext* context, const VecT& desired_shape,
                const tensorflow::Tensor& input_tensor) {
  // input is a strided array (last index is fastest, C-ordered)
  auto input = input_tensor.flat<T>();
  const int rank = input_tensor.dims();
  const auto original_size = input.size();
  // original and target dimensions
  std::vector<tensorflow::int64> X(rank), Y(rank);
  tensorflow::int64 total_size = 1, new_sliced_size = 1;
  // factor by which X increases/decreases w.r.t. Y
  std::vector<float> f(rank);
  // helper arrays related to f
  std::vector<tensorflow::int64> g(rank), G(rank);
  // index of adjustable dimension
  int q;
  tensorflow::TensorShape output_shape;

  // requires that the rank of the input tensor and length of the desired shape
  // are equal
  OP_REQUIRES(context, rank == desired_shape.size(),
              tensorflow::errors::InvalidArgument(
                  "periodic_intersperse expects the rank of the input tensor, ",
                  rank, ", to be the same as the length of the desired shape, ",
                  desired_shape.size(), "."));

  {
    bool found = false;
    for (int i = 0; i < rank; ++i) {
      if (desired_shape(i) < 1) {
        // only one index can be adjustable
        OP_REQUIRES(context, !found,
                    tensorflow::errors::InvalidArgument(
                        "periodic_intersperse expects only "
                        "one index to be marked as adjustable."));
        q = i;
        found = true;
      } else {
        Y[i] = desired_shape(i);
        new_sliced_size *= Y[i];
      }
    }
    // at least one index needs to be adjustable
    OP_REQUIRES(context, found, tensorflow::errors::InvalidArgument(
                                    "periodic_intersperse expects at least "
                                    "one index to be marked as adjustable."));

    int count = 0;
    for (const auto dim_info : input_tensor.shape()) {
      X[count] = dim_info.size;
      total_size *= X[count];
      ++count;
    }

    Y[q] = tensorflow::int64(
        std::floor(float(total_size) / float(new_sliced_size)));

    count = 0;
    for (const auto dim_info : input_tensor.shape()) {
      f[count] = float(Y[count]) / float(X[count]);
      g[count] = tensorflow::int64(std::ceil(f[count]));
      if (count == 0)
        G[count] = 1;
      else
        G[count] = G[count - 1] * g[count - 1];
      ++count;
    }
  }

  // ensure that the new dimension is greater than zero
  OP_REQUIRES(context, Y[q] > 0,
              tensorflow::errors::InvalidArgument(
                  "periodic_intersperse found that the "
                  "adjustable dimension, ",
                  q, ", isn't greater than zero, ", Y[q], "."));
  for (int i = 0; i < rank; ++i) {
    output_shape.AddDim(Y[i]);
  }
  const auto new_size = new_sliced_size * Y[q];

  // Create an output tensor and attach it to the current context
  tensorflow::Tensor* output_tensor = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, output_shape, &output_tensor));
  auto output = output_tensor->flat<T>();

  {
    // memory is allocated for these variables outside the inner loop for
    // efficiency (yes, I know I could create a separate class scope for
    // this purpose instead)
    typename std::decay<decltype(new_size)>::type result = 0;
    std::vector<decltype(result)> output_indices(Y.size());
    const auto rank = Y.size();

    // Fill output tensor with shuffled input tensor values
    for (typename std::decay<decltype(new_size)>::type i = 0; i < new_size;
         ++i) {
      output(i) = input(
          compute_input_index(Y, i, X, q, g, G, result, output_indices, rank));
    }
  }
}

template <class T>
void create_output_tensor(tensorflow::OpKernelContext* context,
                          const tensorflow::Tensor& input_tensor,
                          const tensorflow::DataType& input_tensor_type,
                          const tensorflow::Tensor& desired_shape_tensor) {
  auto desired_shape = desired_shape_tensor.flat<T>();

  // obligatory type switch
  if (input_tensor_type == tensorflow::DataTypeToEnum<float>::value) {
    main_logic<float>(context, desired_shape, input_tensor);
  } else if (input_tensor_type == tensorflow::DataTypeToEnum<double>::value) {
    main_logic<double>(context, desired_shape, input_tensor);
  } else if (input_tensor_type ==
             tensorflow::DataTypeToEnum<tensorflow::int32>::value) {
    main_logic<tensorflow::int32>(context, desired_shape, input_tensor);
  } else if (input_tensor_type ==
             tensorflow::DataTypeToEnum<tensorflow::int64>::value) {
    main_logic<tensorflow::int64>(context, desired_shape, input_tensor);
  }
}

class PeriodicIntersperseOp : public tensorflow::OpKernel {
 public:
  explicit PeriodicIntersperseOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Grab the input tensor and desired shape
    const tensorflow::Tensor& input_tensor = context->input(0);
    const tensorflow::DataType input_tensor_type = context->input_dtype(0);
    const tensorflow::Tensor& desired_shape_tensor = context->input(1);
    const tensorflow::DataType desired_shape_tensor_type =
        context->input_dtype(1);

    // requires that the desired shape is a vector
    OP_REQUIRES(
        context,
        tensorflow::TensorShapeUtils::IsVector(desired_shape_tensor.shape()),
        tensorflow::errors::InvalidArgument(
            "periodic_intersperse expects a 1D vector for the desired shape."));

    // obligatory type switch
    if (desired_shape_tensor_type ==
        tensorflow::DataTypeToEnum<tensorflow::int32>::value) {
      create_output_tensor<tensorflow::int32>(
          context, input_tensor, input_tensor_type, desired_shape_tensor);
    } else if (desired_shape_tensor_type ==
               tensorflow::DataTypeToEnum<tensorflow::int64>::value) {
      create_output_tensor<tensorflow::int64>(
          context, input_tensor, input_tensor_type, desired_shape_tensor);
    }
  }
};

#endif  // TENSORFLOW_KERNELS_PERIODICINTERSPERSE_OP_H_
