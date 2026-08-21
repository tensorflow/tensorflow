/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_KDNN_ADAPTER_H_
#define TENSORFLOW_CORE_UTIL_KDNN_ADAPTER_H_
#include "kdnn.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "kdnn_threadpool.h"
#include "kdnn_types_adapter.h"
#include "kdnn_layout_adapter.h"
#include "tensorflow/core/util/port.h"
#include "tensorflow/core/framework/tensor.h"
#include "operations/kdnn_softmax.hpp"
#include "operations/kdnn_eltwise.hpp"

namespace tensorflow {

inline void kdnnFusedGemm(OpKernelContext* ctx, const Tensor& a, const Tensor& b, Tensor* out,
                    bool fusion_relu, bool trans_x, bool trans_y) {
  int m = a.dim_size(0);
  int n = b.dim_size(trans_y ? 0 : 1);
  int k = b.dim_size(trans_y ? 1 : 0);
  const float *A = a.flat<float>().data();
  const float *B = b.flat<float>().data();
  float *C = out->flat<float>().data();
  const Tensor& bias = ctx->input(2);
  const float *Bias = bias.flat<float>().data();
  if (bias.dims() != 1 || bias.dim_size(0) != n) {
    OP_REQUIRES_OK(ctx, errors::InvalidArgument("bias must be 1-dimensional and match n",
                            bias.shape().DebugString()));
  }
  KDNN::PostOpsDataPtrs po_ptrs;
  KDNN::PostOps post_ops;
  if (fusion_relu) {
    post_ops.AppendEltwise(KDNN::ActivationFunction::RELU);
    po_ptrs.push_back(&post_ops);
  }
  // intra_op thread_pool
  thread::ThreadPool* thread_pool =
    ctx->device()
    ->tensorflow_cpu_worker_threads()
    ->workers;
  kdnn::KDNNThreadPool kdnn_tp(thread_pool);
  KDNN::Threading::ActivateThreadpool(&kdnn_tp);
  const KDNN::TensorInfo srcInfo = {{m, k}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  const KDNN::TensorInfo weightsInfo = {{k, n}, KDNN::Element::TypeT::F32, trans_y ? KDNN::Layout::BA : KDNN::Layout::AB};
  const KDNN::TensorInfo dstInfo = {{m, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  const KDNN::TensorInfo biasInfo = {{1, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  KDNN::Attributes attr;
  attr.SetPostOps(post_ops);
  KDNN::Gemm gemm(srcInfo, weightsInfo, dstInfo, biasInfo, attr);
  gemm.Run(A, B, C, Bias, po_ptrs);
  KDNN::Threading::DeactivateThreadpool();
}

template<typename T>
inline void KDNNConcatImpl(OpKernelContext* ctx,
                    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& inputs,
                    typename TTypes<T, 2>::Matrix* output) {
  KDNN::Element::TypeT kdnnType = KDNN::Element::TypeAdapter<T>::value;
  KDNN::Layout kdnnLayout = KDNN::LayoutAdapter<2, false>::value;
  OP_REQUIRES(ctx, kdnnType != KDNN::Element::TypeT::UNDEFINED,
    errors::InvalidArgument("unsupported kdnn data type"));
  OP_REQUIRES(ctx, kdnnLayout != KDNN::Layout::UNDEFINED,
    errors::InvalidArgument("unsupported kdnn layout"));
  std::vector<KDNN::TensorInfo> inputInfos;
  std::vector<const void *> input_ptrs;
  inputInfos.reserve(inputs.size());
  input_ptrs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto dim0 = inputs[i]->dimension(0);
    auto dim1 = inputs[i]->dimension(1);
    inputInfos.emplace_back(KDNN::TensorInfo{{dim0, dim1}, kdnnType, kdnnLayout});
    input_ptrs.push_back(static_cast<const void*>(inputs[i]->data()));
  }
  void* output_ptr = static_cast<void *>(output->data());
  thread::ThreadPool* thread_pool =
    ctx->device()
    ->tensorflow_cpu_worker_threads()
    ->workers;
  kdnn::KDNNThreadPool kdnn_tp(thread_pool);
  KDNN::Threading::ActivateThreadpool(&kdnn_tp);
  KDNN::TensorInfo outputInfo({output->dimension(0), output->dimension(1)}, kdnnType, kdnnLayout);
  KDNN::ConcatLayer concat(inputInfos, 1, outputInfo);
  concat.Run(input_ptrs.data(), output_ptr);
  KDNN::Threading::DeactivateThreadpool();
}

inline KDNN::TensorInfo MakeInfo(const tensorflow::Tensor* tensor, bool transposed) {
  const tensorflow::TensorShape& shape = tensor->shape();
  int dims = shape.dims();

  std::vector<int64_t> d5 = {1, 1, 1, 1, 1};
  for (int i = 0; i < dims; ++i) {
    d5[4 - i] = shape.dim_size(dims - 1 - i);
  }

  if (transposed) {
    std::swap(d5[3], d5[4]);
  }

  return KDNN::TensorInfo(
    {d5[0], d5[1], d5[2], d5[3], d5[4]},
    KDNN::Element::TypeT::F32,
    transposed ? KDNN::Layout::ABCED : KDNN::Layout::ABCDE
  );
}

inline KDNN::TensorInfo MakeOutputInfo(const KDNN::TensorInfo &tensorA, const KDNN::TensorInfo &tensorB) {
  int dims = tensorA.GetNumDims();
  std::vector<int64_t> d5 = {1, 1, 1, 1, 1};
  for (int i = 0; i < dims - 2; ++i) {
    d5[i] = std::max(tensorA.GetDims()[i], tensorB.GetDims()[i]);
  }
  d5[3] = tensorA.GetDims()[3];
  d5[4] = tensorB.GetDims()[4];
  return KDNN::TensorInfo(
    {d5[0], d5[1], d5[2], d5[3], d5[4]},
    KDNN::Element::TypeT::F32, KDNN::Layout::ABCDE
  );
}

inline void kdnnGemm(const OpKernelContext* ctx, const Tensor& a, const Tensor& b, Tensor* out,
                     bool trans_x, bool trans_y) {
  int m = a.dim_size(trans_x ? 2 : 1);
  int n = b.dim_size(trans_y ? 1 : 2);
  int k = b.dim_size(trans_y ? 2 : 1);
  const float *A = a.flat<float>().data();
  const float *B = b.flat<float>().data();
  float *C = out->flat<float>().data();
  thread::ThreadPool* thread_pool =
    ctx->device()
    ->tensorflow_cpu_worker_threads()
    ->workers;
  kdnn::KDNNThreadPool kdnn_tp(thread_pool);
  KDNN::Threading::ActivateThreadpool(&kdnn_tp);
  const KDNN::TensorInfo srcInfo = {{m, k}, KDNN::Element::TypeT::F32, trans_x ? KDNN::Layout::BA : KDNN::Layout::AB};
  const KDNN::TensorInfo weightsInfo = {{k, n}, KDNN::Element::TypeT::F32, trans_y ? KDNN::Layout::BA : KDNN::Layout::AB};
  const KDNN::TensorInfo dstInfo = {{m, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  KDNN::Gemm gemm(srcInfo, weightsInfo, dstInfo);
  gemm.Run(A, B, C);
  KDNN::Threading::DeactivateThreadpool();
}

inline void kdnnBatchGemm(const OpKernelContext* ctx, const Tensor& a, const Tensor& b, Tensor* out,
                          bool trans_x, bool trans_y) {
  const float *A = a.flat<float>().data();
  const float *B = b.flat<float>().data();
  float *C = out->flat<float>().data();
  thread::ThreadPool* thread_pool =
    ctx->device()
    ->tensorflow_cpu_worker_threads()
    ->workers;
  kdnn::KDNNThreadPool kdnn_tp(thread_pool);
  KDNN::Threading::ActivateThreadpool(&kdnn_tp);
  const KDNN::TensorInfo srcInfo = MakeInfo(&a, trans_x);
  const KDNN::TensorInfo weightsInfo = MakeInfo(&b, trans_y);
  const KDNN::TensorInfo dstInfo = MakeOutputInfo(srcInfo, weightsInfo);
  KDNN::Gemm gemm(srcInfo, weightsInfo, dstInfo);
  gemm.Run(A, B, C);
  KDNN::Threading::DeactivateThreadpool();
}

template <typename Functor>
inline void kdnnFloormodOp(OpKernelContext* ctx, const Tensor &input_0, const Tensor &input_1, Tensor *output) {
    typedef typename Functor::in_type Tin;    // Input scalar data type.
    const Tin* src = input_0.flat<Tin>().data();
    const Tin* src_1 = input_1.flat<Tin>().data();
    Tin* dst = output->flat<Tin>().data();

    KDNN::Shape tensorShape({input_0.shape().num_elements()});
    thread::ThreadPool* thread_pool =
        ctx->device()
        ->tensorflow_cpu_worker_threads()
        ->workers;
    kdnn::KDNNThreadPool kdnn_tp(thread_pool);
    KDNN::Threading::ActivateThreadpool(&kdnn_tp);

    if (std::is_same<Tin, int64_t>::value) {
        KDNN::TensorInfo inputTensorInfo(tensorShape, KDNN::Element::TypeT::S64, KDNN::Layout::A);
        KDNN::TensorInfo inputTensorInfo_1(tensorShape, KDNN::Element::TypeT::S64, KDNN::Layout::A);
        KDNN::TensorInfo outputTensorInfo(tensorShape, KDNN::Element::TypeT::S64, KDNN::Layout::A);
        KDNN::BinaryLayer layer(inputTensorInfo, inputTensorInfo_1, outputTensorInfo, KDNN::BinaryFunction::FLOORMOD);
        layer.Run(src, src_1, dst);
    } else {
        KDNN::TensorInfo inputTensorInfo(tensorShape, KDNN::Element::TypeT::F32, KDNN::Layout::A);
        KDNN::TensorInfo inputTensorInfo_1(tensorShape, KDNN::Element::TypeT::F32, KDNN::Layout::A);
        KDNN::TensorInfo outputTensorInfo(tensorShape, KDNN::Element::TypeT::F32, KDNN::Layout::A);
        KDNN::BinaryLayer layer(inputTensorInfo, inputTensorInfo_1, outputTensorInfo, KDNN::BinaryFunction::FLOORMOD);
        layer.Run(src, src_1, dst);
    }

    KDNN::Threading::DeactivateThreadpool();
    return;
}

template <typename Functor>
inline void kdnnSigmoidOp(OpKernelContext* ctx, const Tensor &input, Tensor *output)
{
    typedef typename Functor::in_type Tin;
    const Tin* src = input.flat<Tin>().data();
    Tin* dst = output->flat<Tin>().data();
    KDNN::Shape tensorShape({input.shape().num_elements()});
    KDNN::TensorInfo inputTensorInfo(tensorShape, KDNN::Element::TypeT::F32, KDNN::Layout::A);
    KDNN::TensorInfo outputTensorInfo(tensorShape, KDNN::Element::TypeT::F32, KDNN::Layout::A);

    thread::ThreadPool* thread_pool =
        ctx->device()
        ->tensorflow_cpu_worker_threads()
        ->workers;
    kdnn::KDNNThreadPool kdnn_tp(thread_pool);
    KDNN::Threading::ActivateThreadpool(&kdnn_tp);
    KDNN::ActivationLayerFWD layer(inputTensorInfo, outputTensorInfo, KDNN::ActivationFunction::SIGMOID);
    layer.Run(src, dst);
    KDNN::Threading::DeactivateThreadpool();
    return;
}

template <typename T>
inline void kdnnSoftmaxOp(OpKernelContext* ctx, const Tensor &input, Tensor *output)
{
    const T* src = input.flat_inner_dims<T>().data();
    T* dst = output->flat_inner_dims<T>().data();
    KDNN::Shape tensorShape({input.flat_inner_dims<T>().dimension(0), input.flat_inner_dims<T>().dimension(1)});
    KDNN::TensorInfo inputTensorInfo(tensorShape, KDNN::Element::TypeT::F32, KDNN::Layout::AB);
    KDNN::TensorInfo outputTensorInfo(tensorShape, KDNN::Element::TypeT::F32, KDNN::Layout::AB);

    thread::ThreadPool* thread_pool =
        ctx->device()
        ->tensorflow_cpu_worker_threads()
        ->workers;
    kdnn::KDNNThreadPool kdnn_tp(thread_pool);
    KDNN::Threading::ActivateThreadpool(&kdnn_tp);
    KDNN::SoftmaxLayerFWD layer(inputTensorInfo, outputTensorInfo, 1, KDNN::AlgorithmKind::SOFTMAX);
    layer.Run(src, dst);
    KDNN::Threading::DeactivateThreadpool();
    return;
}

}// namespace tensorflow
#endif
