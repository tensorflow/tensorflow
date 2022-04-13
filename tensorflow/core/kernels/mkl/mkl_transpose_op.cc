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

#include "dnnl.hpp"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/kernels/transpose_op.h"
#include "tensorflow/core/util/mkl_util.h"

using dnnl::stream;

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
// oneDNN based Transpose implementation
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

// Transpose of N-dimensional tensor using oneDNN
template <typename T>
Status MKLTransposeND(OpKernelContext* context, const Tensor& in_tensor,
                      Tensor* out_tensor, const gtl::ArraySlice<int32>& perm) {
  try {
    engine cpu_engine = engine(engine::kind::cpu, 0);
    MklDnnData<T> in(&cpu_engine);
    MklDnnData<T> out(&cpu_engine);

    memory::dims in_dims = TFShapeToMklDnnDims(in_tensor.shape());
    memory::dims out_dims = TFShapeToMklDnnDims(out_tensor->shape());
    memory::dims in_strides = CalculateTFStrides(in_dims);
    // Reorder output strides based on permutation requested.
    memory::dims out_strides =
        ReorderStrides(CalculateTFStrides(out_dims), perm);

    std::shared_ptr<stream> transpose_stream;
    in.SetUsrMem(in_dims, in_strides, &in_tensor);
    // Output dimensions are same as input dimensions. We adjust the layout
    // using strides.
    out.SetUsrMem(in_dims, out_strides, out_tensor);

    std::vector<primitive> net;
    auto* prim = FindOrCreateReorder<T>(in.GetUsrMem(), out.GetUsrMem());
    MklDnnThreadPool eigen_tp(context);
    transpose_stream.reset(CreateStream(&eigen_tp, prim->GetEngine()));
    in.SetUsrMemDataHandle(&in_tensor, transpose_stream);
    out.SetUsrMemDataHandle(out_tensor, transpose_stream);
    net.push_back(*(prim->GetPrimitive()));
    std::vector<MemoryArgsMap> net_args;
    net_args.push_back(
        {{DNNL_ARG_FROM, *in.GetUsrMem()}, {DNNL_ARG_TO, *out.GetUsrMem()}});
    execute_primitives(net, transpose_stream, net_args);

    return Status::OK();
  } catch (dnnl::error& e) {
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
  // oneDNN has limit on the maximum number of dimensions in a tensor.
  // Fallback to Eigen for not supported cases.
  if (in.dims() <= DNNL_MAX_NDIMS) {
    switch (in.dtype()) {
      case DT_FLOAT:
        return MKLTransposeND<float>(ctx, in, out, perm);
        break;
      case DT_BFLOAT16:
        return MKLTransposeND<bfloat16>(ctx, in, out, perm);
        break;
      // TODO(intel-tf): support other types such as INT8.
      default:
        break;
    }
  }

  // Fallback to eigen if transpose parameters not supported by oneDNN
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<CPUDevice>(), in, perm,
                                   out);
}

Status MklConjugateTransposeCpuOp::DoTranspose(OpKernelContext* ctx,
                                               const Tensor& in,
                                               gtl::ArraySlice<int32> perm,
                                               Tensor* out) {
  // oneDNN has limit on the maximum number of dimensions in a tensor.
  // Fallback to Eigen for not supported cases.
  if (in.dims() <= DNNL_MAX_NDIMS) {
    switch (in.dtype()) {
      case DT_FLOAT:
        return MKLTransposeND<float>(ctx, in, out, perm);
        break;
      case DT_BFLOAT16:
        return MKLTransposeND<bfloat16>(ctx, in, out, perm);
        break;
      // TODO(intel-tf): support other types such as INT8.
      default:
        break;
    }
  }

  // Fallback to eigen if transpose parameters not supported by oneDNN
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
