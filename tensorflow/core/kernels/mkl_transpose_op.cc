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

// See docs in ../ops/array_ops.cc.

#if defined(INTEL_MKL)

#define EIGEN_USE_THREADS

#if !defined(INTEL_MKL_DNN_ONLY)
#include "mkl_trans.h"
#endif

#include "mkldnn.hpp"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/kernels/transpose_op.h"
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::stream;

namespace tensorflow {

// output = TransposeOp(T<any> input, T<int32> perm) takes a tensor
// of type T and rank N, and a permutation of 0, 1, ..., N-1. It
// shuffles the dimensions of the input tensor according to permutation.
//
// Specifically, the returned tensor output meets the following condition:
// 1) output.dims() == input.dims();
// 2) output.dim_size(i) == input.dim_size(perm[i]);
// 3) output.tensor<T, N>(i_0, i_1, ..., i_N-1) ==
//      input.tensor<T, N>(j_0, j_1, ..., j_N-1),
//    where i_s == j_{perm[s]}
//
// REQUIRES: perm is a vector of int32.
// REQUIRES: input.dims() == perm.size().
// REQUIRES: perm is a permutation.

namespace {
#if !defined(INTEL_MKL_DNN_ONLY)
template <typename T>
Status MKLTranspose2D(const char trans, const Tensor& in, Tensor* out);

// Documentation here: https://software.intel.com/en-us/node/520863
// Parameters: (ordering:row-major, operation:transpose, num_rows, num_cols,
//              alpha (for scaling), array, dist_bet_adjacent_cols/rows
//              (source), array, dist_bet_adjacent_cols/rows (dest))

#define INSTANTIATE(T, PREFIX)                                                \
  template <>                                                                 \
  Status MKLTranspose2D<T>(const char trans, const Tensor& in, Tensor* out) { \
    mkl_##PREFIX##omatcopy('R', trans, in.dim_size(0), in.dim_size(1), 1,     \
                           in.flat<T>().data(), in.dim_size(1),               \
                           out->flat<T>().data(), in.dim_size(0));            \
    return Status::OK();                                                      \
  }

INSTANTIATE(float, s)
INSTANTIATE(double, d)

#undef INSTANTIATE

template <>
Status MKLTranspose2D<complex64>(const char trans, const Tensor& in,
                                 Tensor* out) {
  const MKL_Complex8 alpha = {1.0f, 0.0f};
  mkl_comatcopy(
      'R', trans, in.dim_size(0), in.dim_size(1), alpha,
      reinterpret_cast<const MKL_Complex8*>(in.flat<complex64>().data()),
      in.dim_size(1),
      reinterpret_cast<MKL_Complex8*>(
          const_cast<complex64*>(out->flat<complex64>().data())),
      in.dim_size(0));
  return Status::OK();
}

template <>
Status MKLTranspose2D<complex128>(const char trans, const Tensor& in,
                                  Tensor* out) {
  const MKL_Complex16 alpha = {1.0, 0.0};
  mkl_zomatcopy(
      'R', trans, in.dim_size(0), in.dim_size(1), alpha,
      reinterpret_cast<const MKL_Complex16*>(in.flat<complex128>().data()),
      in.dim_size(1),
      reinterpret_cast<MKL_Complex16*>(
          const_cast<complex128*>(out->flat<complex128>().data())),
      in.dim_size(0));
  return Status::OK();
}

static const char kMKLTranspose = 'T';
static const char kMKLConjugateTranspose = 'C';

#endif  // if !defined(INTEL_MKL_DNN_ONLY)

// MKL-DNN based Transpose implementation
template <typename T>
Status MKLTransposeND(OpKernelContext* ctx, const Tensor& in, Tensor* out,
                      const gtl::ArraySlice<int32>& perm);

static inline memory::dims ReorderStrides(const memory::dims& strides,
                                          const gtl::ArraySlice<int32>& perm) {
  memory::dims reordered_strides;
  reordered_strides.resize(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    reordered_strides[perm[i]] = strides[i];
  }
  return reordered_strides;
}

// Transpose of N-dimensional tensor using MKL-DNN
template <typename T>
Status MKLTransposeND(OpKernelContext* context, const Tensor& in_tensor,
                      Tensor* out_tensor, const gtl::ArraySlice<int32>& perm) {
  try {
    engine cpu_engine = engine(ENGINE_CPU, 0);
    MklDnnData<T> in(&cpu_engine);
    MklDnnData<T> out(&cpu_engine);

    memory::dims in_dims = TFShapeToMklDnnDims(in_tensor.shape());
    memory::dims out_dims = TFShapeToMklDnnDims(out_tensor->shape());
    memory::dims in_strides = CalculateTFStrides(in_dims);
    // Reorder output strides based on permutation requested.
    memory::dims out_strides =
        ReorderStrides(CalculateTFStrides(out_dims), perm);

    in.SetUsrMem(in_dims, in_strides, &in_tensor);
    // Output dimensions are same as input dimensions. We adjust the layout
    // using strides.
    out.SetUsrMem(in_dims, out_strides, out_tensor);

    std::vector<primitive> net;
    std::shared_ptr<stream> transpose_stream;
    transpose_stream.reset(new CPU_STREAM(cpu_engine));
#ifdef ENABLE_MKLDNN_V1
    const int net_idx = 0;
    net.push_back(reorder(in.GetOpMem(), out.GetOpMem()));
    std::vector<std::unordered_map<int, memory>> net_args;
    net_args.push_back(
        {{MKLDNN_ARG_FROM, in.GetOpMem()}, {MKLDNN_ARG_TO, out.GetOpMem()}});
    net.at(net_idx).execute(*transpose_stream, net_args.at(net_idx));
#else
    net.push_back(FindOrCreateReorder<T>(in.GetUsrMem(), out.GetUsrMem()));
    transpose_stream->submit(net).wait();
#endif  // ENABLE_MKLDNN_V1

    return Status::OK();
  } catch (mkldnn::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + std::string(e.message) + ", in file " +
                       std::string(__FILE__) + ":" + std::to_string(__LINE__);
    return errors::Aborted("Operation received an exception:", error_msg);
  }
}

}  // namespace

Status MklTransposeCpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                      gtl::ArraySlice<int32> perm,
                                      Tensor* out) {
#if !defined(INTEL_MKL_DNN_ONLY)
  if (in.dims() == 2) {
    if (perm[0] == 0 && perm[1] == 1) {
      return Status::OK();
    }
    switch (in.dtype()) {
      case DT_FLOAT:
        return MKLTranspose2D<float>(kMKLTranspose, in, out);
      case DT_DOUBLE:
        return MKLTranspose2D<double>(kMKLTranspose, in, out);
      case DT_COMPLEX64:
        return MKLTranspose2D<complex64>(kMKLTranspose, in, out);
      case DT_COMPLEX128:
        return MKLTranspose2D<complex128>(kMKLTranspose, in, out);
      default:
        break;
    }
  }
#endif

  // MKL-DNN has limit on the maximum number of dimensions in a tensor.
  // Fallback to Eigen for not supported cases.
  if (in.dims() <= TENSOR_MAX_DIMS) {
    switch (in.dtype()) {
      case DT_FLOAT:
        return MKLTransposeND<float>(ctx, in, out, perm);
        break;
      case DT_BFLOAT16:
        return MKLTransposeND<bfloat16>(ctx, in, out, perm);
        break;
      // TODO(nhasabni): support other types such as INT8.
      default:
        break;
    }
  }

  // Fallback to eigen if transpose parameters not supported by MKL or MKL-DNN
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<CPUDevice>(), in, perm,
                                   out);
}

Status MklConjugateTransposeCpuOp::DoTranspose(OpKernelContext* ctx,
                                               const Tensor& in,
                                               gtl::ArraySlice<int32> perm,
                                               Tensor* out) {
#if !defined(INTEL_MKL_DNN_ONLY)
  if (in.dims() == 2 && perm[0] == 1 && perm[1] == 0) {
    // TODO(rmlarsen): By setting lda and ldb, we could use the MKL kernels
    // for any transpose that can be reduced to swapping the last two
    // dimensions in a rank-3 tensor. We can even run each outer dimension in
    // a separate thread.
    switch (in.dtype()) {
      case DT_FLOAT:
        return MKLTranspose2D<float>(kMKLTranspose, in, out);
      case DT_DOUBLE:
        return MKLTranspose2D<double>(kMKLTranspose, in, out);
      case DT_COMPLEX64:
        return MKLTranspose2D<complex64>(kMKLConjugateTranspose, in, out);
      case DT_COMPLEX128:
        return MKLTranspose2D<complex128>(kMKLConjugateTranspose, in, out);
      default:
        break;
    }
  }
#endif

  // MKL-DNN has limit on the maximum number of dimensions in a tensor.
  // Fallback to Eigen for not supported cases.
  if (in.dims() <= TENSOR_MAX_DIMS) {
    switch (in.dtype()) {
      case DT_FLOAT:
        return MKLTransposeND<float>(ctx, in, out, perm);
        break;
      case DT_BFLOAT16:
        return MKLTransposeND<bfloat16>(ctx, in, out, perm);
        break;
      // TODO(nhasabni): support other types such as INT8.
      default:
        break;
    }
  }

  // Fallback to eigen if transpose parameters not supported by MKL or MKL-DNN
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<CPUDevice>(), in,
                                            perm, out);
}

#define REGISTER(T)                                                           \
  REGISTER_KERNEL_BUILDER(Name("_MklTranspose")                               \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("perm")                             \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklTransposeCpuOp);                                 \
  REGISTER_KERNEL_BUILDER(Name("_MklConjugateTranspose")                      \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("perm")                             \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklConjugateTransposeCpuOp);

TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER

}  // namespace tensorflow

#endif  // INTEL_MKL
