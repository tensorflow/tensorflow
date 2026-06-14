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
#if defined(INTEL_MKL)

#include <memory>
#include <string>
#include <utility>

#include "dnnl.hpp"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
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
                                    dims* reshaped_dims) const {
    if (reshaped_dims == nullptr) {
      return;
    }
    const int ndims_input = input_shape.dims();
    const int ndims_output = output_shape.dims();
    const int dim_offset = ndims_output - ndims_input;
    if (dim_offset <= 0) {
      return;
    }
    reshaped_dims->clear();
    reshaped_dims->resize(ndims_output, 1);
    for (int dim_idx = 0; dim_idx < ndims_input; ++dim_idx) {
      (*reshaped_dims)[dim_idx + dim_offset] = input_shape.dim_size(dim_idx);
    }
  }

  std::unique_ptr<MklMatMulParams> CreateMatMulParams(
      const std::string& prefix, const TensorShape& lhs_shape,
      const TensorShape& rhs_shape, const TensorShape& out_shape, bool adj_x,
      bool adj_y) const {
    const int ndims_lhs = lhs_shape.dims();
    const int ndims_rhs = rhs_shape.dims();
    const int ndims_out = out_shape.dims();
    if (ndims_out < 2) {
      return nullptr;
    }
    DCHECK_GE(ndims_out, ndims_lhs);
    DCHECK_GE(ndims_out, ndims_rhs);

    dnnl::memory::dims lhs_dims = TFShapeToMklDnnDims(lhs_shape);
    dnnl::memory::dims rhs_dims = TFShapeToMklDnnDims(rhs_shape);
    dnnl::memory::dims out_dims = TFShapeToMklDnnDims(out_shape);

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
    dnnl::memory::dims lhs_strides = CalculateTFStrides(lhs_dims);
    dnnl::memory::dims rhs_strides = CalculateTFStrides(rhs_dims);
    dnnl::memory::dims out_strides = CalculateTFStrides(out_dims);

    if (ndims_lhs < ndims_out) {
      const int dim_offset = ndims_out - ndims_lhs;
      for (int i = 0; i < dim_offset; ++i) {
        lhs_strides[i] = 0;
      }
    }
    if (ndims_rhs < ndims_out) {
      const int dim_offset = ndims_out - ndims_rhs;
      for (int i = 0; i < dim_offset; ++i) {
        rhs_strides[i] = 0;
      }
    }

    if (adj_x) {
      const int m_idx = ndims_out - 1;
      const int k_idx = ndims_out - 2;
      const dnnl::memory::dim m = lhs_dims[m_idx];  // Cols of x before swap.
      std::swap(lhs_dims[m_idx], lhs_dims[k_idx]);
      lhs_strides[m_idx] = m;
      lhs_strides[k_idx] = 1;
    }

    if (adj_y) {
      const int k_idx = ndims_out - 1;
      const int n_idx = ndims_out - 2;
      const dnnl::memory::dim k = rhs_dims[k_idx];  // Cols of y before swap.
      std::swap(rhs_dims[k_idx], rhs_dims[n_idx]);
      rhs_strides[k_idx] = k;
      rhs_strides[n_idx] = 1;
    }

    return std::make_unique<MklMatMulParams>(
        prefix, std::move(lhs_dims), std::move(rhs_dims), std::move(out_dims),
        std::move(lhs_strides), std::move(rhs_strides), std::move(out_strides));
  }
};

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_BATCH_MATMUL_HELPER_H_
