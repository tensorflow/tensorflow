/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_BATCH_MATMUL_HELPER_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_BATCH_MATMUL_HELPER_H_
#ifdef INTEL_MKL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

struct MklBatchMatMulHelper {
  using dims = dnnl::memory::dims;
  // This method makes the rank (ndims) of input same as the output by creating
  // new axes to the input. For example, if input shape is [a, b, c, d] and
  // output shape is [e, f, g, h, i, j], then the reshaped input would have a
  // shape of [1, 1, a, b, c, d].
  void ExpandInputDimsToOutputShape(const TensorShape& input_shape,
                                    const TensorShape& output_shape,
                                    dims* reshaped_dims) {
    auto ndims_input = input_shape.dims();
    auto ndims_output = output_shape.dims();
    auto dim_offset = ndims_output - ndims_input;
    DCHECK(dim_offset > 0);
    reshaped_dims->clear();
    reshaped_dims->resize(ndims_output, 1);
    auto input_dims = input_shape.dim_sizes();
    for (int dim_idx = 0; dim_idx < ndims_input; ++dim_idx)
      reshaped_dims->at(dim_idx + dim_offset) = input_dims[dim_idx];
  }

  std::unique_ptr<MklMatMulParams> CreateMatMulParams(
      const TensorShape& lhs_shape, const TensorShape& rhs_shape,
      const TensorShape& out_shape, bool& adj_x, bool& adj_y) {
    const auto ndims_lhs = lhs_shape.dims();
    const auto ndims_rhs = rhs_shape.dims();
    const auto ndims_out = out_shape.dims();
    auto lhs_dims = TFShapeToMklDnnDims(lhs_shape);
    auto rhs_dims = TFShapeToMklDnnDims(rhs_shape);
    auto out_dims = TFShapeToMklDnnDims(out_shape);

    // DNNL matmul_primitive requires ranks of inputs and output to be same.
    // Create dnnl::memory::dims for inputs and output of same rank.
    // It is assumed here that MatMulBCast object creates output_batch_shape as
    // a conforming superset of input batch shapes, i.e., ndims_out >=
    // ndims_lhs and ndims_out >= ndims_rhs.
    if (ndims_lhs < ndims_out) {
      ExpandInputDimsToOutputShape(lhs_shape, out_shape, &lhs_dims);
    }
    if (ndims_rhs < ndims_out) {
      ExpandInputDimsToOutputShape(rhs_shape, out_shape, &rhs_dims);
    }
    using dim = dnnl::memory::dim;
    dim m;  // Number of rows in x
    dim k;  // Number of columns in x
    dim n;  // Number of columns in y
    auto lhs_strides = CalculateTFStrides(lhs_dims);
    auto rhs_strides = CalculateTFStrides(rhs_dims);
    auto out_strides = CalculateTFStrides(out_dims);

    if (adj_x) {
      int m_idx = ndims_out - 1;
      int k_idx = ndims_out - 2;
      m = lhs_dims[m_idx];
      k = lhs_dims[k_idx];
      std::swap(lhs_dims[m_idx], lhs_dims[k_idx]);
      lhs_strides[m_idx] = m;
      lhs_strides[k_idx] = 1;
    }

    if (adj_y) {
      int k_idx = ndims_out - 1;
      int n_idx = ndims_out - 2;
      k = rhs_dims[k_idx];
      n = rhs_dims[n_idx];
      std::swap(rhs_dims[k_idx], rhs_dims[n_idx]);
      rhs_strides[k_idx] = k;
      rhs_strides[n_idx] = 1;
    }

    return std::make_unique<MklMatMulParams>(
        lhs_dims, rhs_dims, out_dims, lhs_strides, rhs_strides, out_strides);
  }
};

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_BATCH_MATMUL_HELPER_H_
