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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/contrib/rnn/kernels/lstm_ops.h"

#include <memory>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#if GOOGLE_CUDA

namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

#endif  // GOOGLE_CUDA

namespace functor {
template <typename T>
void TensorCuBlasGemm<T>::operator()(OpKernelContext* ctx,
                                     perftools::gputools::Stream* stream,
                                     bool transa, bool transb, uint64 m,
                                     uint64 n, uint64 k, T alpha, const T* a,
                                     int lda, const T* b, int ldb, T beta, T* c,
                                     int ldc) {
#if GOOGLE_CUDA
  perftools::gputools::blas::Transpose trans[] = {
      perftools::gputools::blas::Transpose::kNoTranspose,
      perftools::gputools::blas::Transpose::kTranspose};

  auto a_ptr = AsDeviceMemory(a);
  auto b_ptr = AsDeviceMemory(b);
  auto c_ptr = AsDeviceMemory(c);

  bool blas_launch_status =
      stream
          ->ThenBlasGemm(trans[transa], trans[transb], m, n, k, alpha, a_ptr,
                         lda, b_ptr, ldb, beta, &c_ptr, ldc)
          .ok();
  OP_REQUIRES(ctx, blas_launch_status, errors::Aborted("CuBlasGemm failed!"));
#else
  ctx->SetStatus(errors::InvalidArgument("CuBlasGemm needs CUDA."));
#endif
}

template struct TensorCuBlasGemm<float>;
// template struct TensorCuBlasGemm<double>;
}  // end namespace functor

template <typename Device, typename T, bool USE_CUBLAS>
class LSTMBlockCellOp : public OpKernel {
 public:
  explicit LSTMBlockCellOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_clip", &cell_clip_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 cell_size = cs_prev_tensor->dim_size(1);

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                                        cs_prev_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("h_prev.dims(1) != cell_size: ",
                                        h_prev_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != input_size + cell_size: ",
                    w_tensor->dim_size(0), " vs. ", input_size + cell_size));
    OP_REQUIRES(
        ctx, w_tensor->dim_size(1) == cell_size * 4,
        errors::InvalidArgument("w.dim_size(1) != cell_size * 4: ",
                                w_tensor->dim_size(1), " vs. ", cell_size * 4));

    OP_REQUIRES(
        ctx, b_tensor->dim_size(0) == cell_size * 4,
        errors::InvalidArgument("b.dim_size(0) != cell_size * 4: ",
                                b_tensor->dim_size(0), " vs. ", cell_size * 4));

    // Allocate our output tensors.
    Tensor* i_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("i", TensorShape({batch_size, cell_size}),
                                  &i_tensor));

    Tensor* cs_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("cs", TensorShape({batch_size, cell_size}),
                                  &cs_tensor));

    Tensor* f_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("f", TensorShape({batch_size, cell_size}),
                                  &f_tensor));

    Tensor* o_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("o", TensorShape({batch_size, cell_size}),
                                  &o_tensor));

    Tensor* ci_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("ci", TensorShape({batch_size, cell_size}),
                                  &ci_tensor));

    Tensor* co_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("co", TensorShape({batch_size, cell_size}),
                                  &co_tensor));

    Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("h", TensorShape({batch_size, cell_size}),
                                  &h_tensor));

    // Allocate our temp tensors.
    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &xh_tensor));

    Tensor icfo_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                      TensorShape({batch_size, cell_size * 4}),
                                      &icfo_tensor));

    const Device& device = ctx->eigen_device<Device>();
    perftools::gputools::Stream* stream =
        std::is_same<Device, GPUDevice>::value
            ? ctx->op_device_context()->stream()
            : nullptr;

    functor::LSTMBlockCellFprop<Device, T, USE_CUBLAS>(batch_size, input_size,
                                                       cell_size)(
        ctx, stream, device, forget_bias_, cell_clip_, use_peephole_,
        x_tensor->matrix<T>(), cs_prev_tensor->matrix<T>(),
        h_prev_tensor->matrix<T>(), w_tensor->matrix<T>(), wci_tensor->vec<T>(),
        wcf_tensor->vec<T>(), wco_tensor->vec<T>(), b_tensor->vec<T>(),
        xh_tensor.matrix<T>(), i_tensor->matrix<T>(), cs_tensor->matrix<T>(),
        f_tensor->matrix<T>(), o_tensor->matrix<T>(), ci_tensor->matrix<T>(),
        co_tensor->matrix<T>(), icfo_tensor.matrix<T>(), h_tensor->matrix<T>());
  }

 private:
  float forget_bias_;
  float cell_clip_;
  bool use_peephole_;
};

#define REGISTER_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("LSTMBlockCell").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LSTMBlockCellOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
// REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                \
  template <>                                                              \
  void LSTMBlockCellFprop<GPUDevice, T, true>::operator()(                 \
      OpKernelContext* ctx, perftools::gputools::Stream* stream,           \
      const GPUDevice& d, const T forget_bias, const T cell_clip,          \
      bool use_peephole, typename TTypes<T>::ConstMatrix x,                \
      typename TTypes<T>::ConstMatrix cs_prev,                             \
      typename TTypes<T>::ConstMatrix h_prev,                              \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci, \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,  \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::Matrix xh,       \
      typename TTypes<T>::Matrix i, typename TTypes<T>::Matrix cs,         \
      typename TTypes<T>::Matrix f, typename TTypes<T>::Matrix o,          \
      typename TTypes<T>::Matrix ci, typename TTypes<T>::Matrix co,        \
      typename TTypes<T>::Matrix icfo, typename TTypes<T>::Matrix h);      \
                                                                           \
  extern template struct LSTMBlockCellFprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
// DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("LSTMBlockCell").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LSTMBlockCellOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
// REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

template <typename Device, typename T, bool USE_CUBLAS>
class LSTMBlockCellGradOp : public OpKernel {
 public:
  explicit LSTMBlockCellGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const Tensor* i_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("i", &i_tensor));

    const Tensor* cs_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs", &cs_tensor));

    const Tensor* f_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("f", &f_tensor));

    const Tensor* o_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("o", &o_tensor));

    const Tensor* ci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ci", &ci_tensor));

    const Tensor* co_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("co", &co_tensor));

    const Tensor* cs_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_grad", &cs_grad_tensor));

    const Tensor* h_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_grad", &h_grad_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 cell_size = cs_prev_tensor->dim_size(1);

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                                        cs_prev_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("h_prev.dims(1) != cell_size: ",
                                        h_prev_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != input_size + cell_size: ",
                    w_tensor->dim_size(0), " vs. ", input_size + cell_size));
    OP_REQUIRES(
        ctx, w_tensor->dim_size(1) == cell_size * 4,
        errors::InvalidArgument("w.dim_size(1) != cell_size * 4: ",
                                w_tensor->dim_size(1), " vs. ", cell_size * 4));

    OP_REQUIRES(
        ctx, b_tensor->dim_size(0) == cell_size * 4,
        errors::InvalidArgument("b.dim_size(0) != cell_size * 4: ",
                                b_tensor->dim_size(0), " vs. ", cell_size * 4));

    OP_REQUIRES(
        ctx, i_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("i.dim_size(0) != batch_size: ",
                                i_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(
        ctx, i_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("i.dim_size(1) != cell_size: ",
                                i_tensor->dim_size(1), " vs. ", cell_size));

    OP_REQUIRES(
        ctx, cs_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("cs.dim_size(0) != batch_size: ",
                                cs_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(
        ctx, cs_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("cs.dim_size(1) != cell_size: ",
                                cs_tensor->dim_size(1), " vs. ", cell_size));

    OP_REQUIRES(
        ctx, f_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("f.dim_size(0) != batch_size: ",
                                f_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(
        ctx, f_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("i.dim_size(1) != cell_size: ",
                                f_tensor->dim_size(1), " vs. ", cell_size));

    OP_REQUIRES(
        ctx, o_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("o.dim_size(0) != batch_size: ",
                                o_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(
        ctx, o_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("o.dim_size(1) != cell_size: ",
                                o_tensor->dim_size(1), " vs. ", cell_size));

    OP_REQUIRES(
        ctx, ci_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("ci.dim_size(0) != batch_size: ",
                                ci_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(
        ctx, ci_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("ci.dim_size(1) != cell_size: ",
                                ci_tensor->dim_size(1), " vs. ", cell_size));

    OP_REQUIRES(
        ctx, co_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("co.dim_size(0) != batch_size: ",
                                co_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(
        ctx, co_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("co.dim_size(1) != cell_size: ",
                                co_tensor->dim_size(1), " vs. ", cell_size));

    OP_REQUIRES(ctx, cs_grad_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "cs_grad_tensor.dims(0) != batch_size: ",
                    cs_grad_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(ctx, cs_grad_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_grad_tensor.dims(1) != cell_size: ",
                                        cs_grad_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, h_grad_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_grad_tensor.dims(0) != batch_size: ",
                                        h_grad_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_grad_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("h_grad_tensor.dims(1) != cell_size: ",
                                        h_grad_tensor->dim_size(1), " vs. ",
                                        cell_size));

    // Allocate our output tensors.
    Tensor* cs_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("cs_prev_grad",
                                        TensorShape({batch_size, cell_size}),
                                        &cs_prev_grad_tensor));

    Tensor* dicfo_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            "dicfo", TensorShape({batch_size, cell_size * 4}),
                            &dicfo_tensor));

    Tensor* wci_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wci_grad", wci_tensor->shape(),
                                             &wci_grad_tensor));

    Tensor* wcf_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wcf_grad", wcf_tensor->shape(),
                                             &wcf_grad_tensor));

    Tensor* wco_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wco_grad", wco_tensor->shape(),
                                             &wco_grad_tensor));

    // Allocate our temp tensors.
    Tensor do_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &do_tensor));

    Tensor dcs_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &dcs_tensor));

    Tensor dci_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &dci_tensor));

    Tensor df_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &df_tensor));

    Tensor di_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &di_tensor));

    const Device& device = ctx->eigen_device<Device>();
    perftools::gputools::Stream* stream =
        std::is_same<Device, GPUDevice>::value
            ? ctx->op_device_context()->stream()
            : nullptr;

    functor::TensorZero<Device, T>()(device, wci_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, wcf_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, wco_grad_tensor->flat<float>());

    functor::LSTMBlockCellBprop<Device, T, USE_CUBLAS>(batch_size, input_size,
                                                       cell_size)(
        ctx, stream, device, use_peephole_, x_tensor->matrix<T>(),
        cs_prev_tensor->matrix<T>(), h_prev_tensor->matrix<T>(),
        w_tensor->matrix<T>(), wci_tensor->vec<T>(), wcf_tensor->vec<T>(),
        wco_tensor->vec<T>(), b_tensor->vec<T>(), i_tensor->matrix<T>(),
        cs_tensor->matrix<T>(), f_tensor->matrix<T>(), o_tensor->matrix<T>(),
        ci_tensor->matrix<T>(), co_tensor->matrix<T>(),
        cs_grad_tensor->matrix<T>(), h_grad_tensor->matrix<T>(),
        do_tensor.matrix<T>(), dcs_tensor.matrix<T>(), dci_tensor.matrix<T>(),
        df_tensor.matrix<T>(), di_tensor.matrix<T>(), dicfo_tensor->matrix<T>(),
        cs_prev_grad_tensor->matrix<T>(), wci_grad_tensor->vec<T>(),
        wcf_grad_tensor->vec<T>(), wco_grad_tensor->vec<T>());
  }

 protected:
  bool use_peephole_;
};

#define REGISTER_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("LSTMBlockCellGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LSTMBlockCellGradOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
// REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                   \
  template <>                                                                 \
  void LSTMBlockCellBprop<GPUDevice, T, true>::operator()(                    \
      OpKernelContext* ctx, perftools::gputools::Stream* stream,              \
      const GPUDevice& d, bool use_peephole,                                  \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstMatrix cs_prev,                                \
      typename TTypes<T>::ConstMatrix h_prev,                                 \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,    \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,     \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::ConstMatrix i,      \
      typename TTypes<T>::ConstMatrix cs, typename TTypes<T>::ConstMatrix f,  \
      typename TTypes<T>::ConstMatrix o, typename TTypes<T>::ConstMatrix ci,  \
      typename TTypes<T>::ConstMatrix co,                                     \
      typename TTypes<T>::ConstMatrix cs_grad,                                \
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_, \
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,         \
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,           \
      typename TTypes<T>::Matrix dicfo,                                       \
      typename TTypes<T>::Matrix cs_prev_grad,                                \
      typename TTypes<T>::Vec wci_grad, typename TTypes<T>::Vec wcf_grad,     \
      typename TTypes<T>::Vec wco_grad);                                      \
                                                                              \
  extern template struct LSTMBlockCellBprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
// DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("LSTMBlockCellGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LSTMBlockCellGradOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
// REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

template <typename Device, typename T, bool USE_CUBLAS>
class BlockLSTMOp : public OpKernel {
 public:
  explicit BlockLSTMOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_len", &max_len_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_clip", &cell_clip_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* seq_len_max_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("seq_len_max", &seq_len_max_tensor));

    OpInputList x_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("x", &x_list));
    const int64 batch_size = x_list[0].dim_size(0);
    const int64 input_size = x_list[0].dim_size(1);

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));
    const int64 cell_size = b_tensor->dim_size(0) / 4;

    OpOutputList i_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("i", &i_list));

    OpOutputList cs_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("cs", &cs_list));

    OpOutputList f_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("f", &f_list));

    OpOutputList o_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("o", &o_list));

    OpOutputList ci_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("ci", &ci_list));

    OpOutputList co_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("co", &co_list));

    OpOutputList h_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("h", &h_list));

    TensorShape batch_cell_shape({batch_size, cell_size});
    for (int64 t = 0; t < max_len_; ++t) {
      Tensor* i_tensor = nullptr;
      OP_REQUIRES_OK(ctx, i_list.allocate(t, batch_cell_shape, &i_tensor));

      Tensor* cs_tensor = nullptr;
      OP_REQUIRES_OK(ctx, cs_list.allocate(t, batch_cell_shape, &cs_tensor));

      Tensor* f_tensor = nullptr;
      OP_REQUIRES_OK(ctx, f_list.allocate(t, batch_cell_shape, &f_tensor));

      Tensor* o_tensor = nullptr;
      OP_REQUIRES_OK(ctx, o_list.allocate(t, batch_cell_shape, &o_tensor));

      Tensor* ci_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ci_list.allocate(t, batch_cell_shape, &ci_tensor));

      Tensor* co_tensor = nullptr;
      OP_REQUIRES_OK(ctx, co_list.allocate(t, batch_cell_shape, &co_tensor));

      Tensor* h_tensor = nullptr;
      OP_REQUIRES_OK(ctx, h_list.allocate(t, batch_cell_shape, &h_tensor));
    }

    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &xh_tensor));

    Tensor icfo_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                      TensorShape({batch_size, cell_size * 4}),
                                      &icfo_tensor));

    const Device& device = ctx->eigen_device<Device>();
    perftools::gputools::Stream* stream =
        std::is_same<Device, GPUDevice>::value
            ? ctx->op_device_context()->stream()
            : nullptr;

    const int64 seq_len_max = seq_len_max_tensor->scalar<int64>()();
    for (int64 t = 0; t < seq_len_max; ++t) {
      const Tensor& x_tensor = x_list[t];
      const Tensor& cs_prev_tensor2 =
          t == 0 ? *cs_prev_tensor : *cs_list[t - 1];
      const Tensor& h_prev_tensor2 = t == 0 ? *h_prev_tensor : *h_list[t - 1];

      Tensor* i_tensor = i_list[t];
      Tensor* cs_tensor = cs_list[t];
      Tensor* f_tensor = f_list[t];
      Tensor* o_tensor = o_list[t];
      Tensor* ci_tensor = ci_list[t];
      Tensor* co_tensor = co_list[t];
      Tensor* h_tensor = h_list[t];

      functor::LSTMBlockCellFprop<Device, T, USE_CUBLAS>(batch_size, input_size,
                                                         cell_size)(
          ctx, stream, device, forget_bias_, cell_clip_, use_peephole_,
          x_tensor.matrix<T>(), cs_prev_tensor2.matrix<T>(),
          h_prev_tensor2.matrix<T>(), w_tensor->matrix<T>(),
          wci_tensor->vec<T>(), wcf_tensor->vec<T>(), wco_tensor->vec<T>(),
          b_tensor->vec<T>(), xh_tensor.matrix<T>(), i_tensor->matrix<T>(),
          cs_tensor->matrix<T>(), f_tensor->matrix<T>(), o_tensor->matrix<T>(),
          ci_tensor->matrix<T>(), co_tensor->matrix<T>(),
          icfo_tensor.matrix<T>(), h_tensor->matrix<T>());
    }

    for (int64 t = seq_len_max; t < max_len_; ++t) {
      Tensor* cs_tensor = cs_list[t];
      Tensor* h_tensor = h_list[t];

      functor::TensorZero<Device, T>()(device, cs_tensor->flat<float>());
      functor::TensorZero<Device, T>()(device, h_tensor->flat<float>());
    }
  }

 private:
  int64 max_len_;
  float forget_bias_;
  float cell_clip_;
  bool use_peephole_;
};

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("BlockLSTM").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BlockLSTMOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
// REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                              \
  template <>                                                            \
  void TensorZero<GPUDevice, T>::operator()(const GPUDevice& d,          \
                                            typename TTypes<T>::Flat t); \
                                                                         \
  extern template struct TensorZero<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
// DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                           \
  REGISTER_KERNEL_BUILDER(Name("BlockLSTM")              \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("seq_len_max") \
                              .TypeConstraint<T>("T"),   \
                          BlockLSTMOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
// REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

template <typename Device, typename T, bool USE_CUBLAS>
class BlockLSTMGradOp : public OpKernel {
 public:
  explicit BlockLSTMGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_len", &max_len_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* seq_len_max_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("seq_len_max", &seq_len_max_tensor));

    OpInputList x_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("x", &x_list));
    const int64 batch_size = x_list[0].dim_size(0);
    const int64 input_size = x_list[0].dim_size(1);

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));
    const int64 cell_size = w_tensor->dim_size(1) / 4;
    OP_REQUIRES(ctx, input_size + cell_size == w_tensor->dim_size(0),
                errors::InvalidArgument("w matrix rows don't match: ",
                                        input_size + cell_size, " vs. ",
                                        w_tensor->dim_size(0)));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));
    OP_REQUIRES(
        ctx, cell_size == b_tensor->dim_size(0) / 4,
        errors::InvalidArgument("w and b cell_size don't match: ", cell_size,
                                " vs. ", b_tensor->dim_size(0)));

    OpInputList i_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("i", &i_list));

    OpInputList cs_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("cs", &cs_list));

    OpInputList f_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("f", &f_list));

    OpInputList o_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("o", &o_list));

    OpInputList ci_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("ci", &ci_list));

    OpInputList co_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("co", &co_list));

    OpInputList h_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("h", &h_list));

    OpInputList cs_grad_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("cs_grad", &cs_grad_list));

    OpInputList h_grad_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("h_grad", &h_grad_list));

    OpOutputList x_grad_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("x_grad", &x_grad_list));

    Tensor* cs_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("cs_prev_grad", cs_prev_tensor->shape(),
                                        &cs_prev_grad_tensor));

    Tensor* h_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("h_prev_grad", h_prev_tensor->shape(),
                                        &h_prev_grad_tensor));

    Tensor* w_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("w_grad", w_tensor->shape(), &w_grad_tensor));

    Tensor* wci_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wci_grad", wci_tensor->shape(),
                                             &wci_grad_tensor));

    Tensor* wcf_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wcf_grad", wcf_tensor->shape(),
                                             &wcf_grad_tensor));

    Tensor* wco_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wco_grad", wco_tensor->shape(),
                                             &wco_grad_tensor));

    Tensor* b_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("b_grad", b_tensor->shape(), &b_grad_tensor));

    TensorShape batch_input_shape({batch_size, input_size});
    TensorShape batch_cell_shape({batch_size, cell_size});
    for (int64 t = 0; t < max_len_; ++t) {
      Tensor* x_grad_tensor = nullptr;
      OP_REQUIRES_OK(
          ctx, x_grad_list.allocate(t, batch_input_shape, &x_grad_tensor));
    }

    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &xh_tensor));

    Tensor xh_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           xh_tensor.shape(), &xh_grad_tensor));

    Tensor do_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &do_tensor));

    Tensor dcs_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &dcs_tensor));

    Tensor dci_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &dci_tensor));

    Tensor df_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &df_tensor));

    Tensor di_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &di_tensor));

    Tensor dicfo_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                      TensorShape({batch_size, cell_size * 4}),
                                      &dicfo_tensor));

    Tensor cs_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &cs_grad_tensor));

    Tensor h_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &h_grad_tensor));


    const Device& device = ctx->eigen_device<Device>();
    perftools::gputools::Stream* stream =
        std::is_same<Device, GPUDevice>::value
            ? ctx->op_device_context()->stream()
            : nullptr;

    functor::TensorZero<Device, T>()(device, cs_grad_tensor.flat<float>());
    functor::TensorZero<Device, T>()(device,
                                     cs_prev_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, h_grad_tensor.flat<float>());
    functor::TensorZero<Device, T>()(device, h_prev_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, w_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, wci_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, wcf_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, wco_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, b_grad_tensor->flat<float>());

    const int64 seq_len_max = seq_len_max_tensor->scalar<int64>()();
    for (int64 t = seq_len_max - 1; t >= 0; --t) {
      const Tensor& x_tensor = x_list[t];
      const Tensor& cs_prev_tensor2 = t == 0 ? *cs_prev_tensor : cs_list[t - 1];
      const Tensor& h_prev_tensor2 = t == 0 ? *h_prev_tensor : h_list[t - 1];
      const Tensor& i_tensor = i_list[t];
      const Tensor& cs_tensor = cs_list[t];
      const Tensor& f_tensor = f_list[t];
      const Tensor& o_tensor = o_list[t];
      const Tensor& ci_tensor = ci_list[t];
      const Tensor& co_tensor = co_list[t];

      // Grab previous CS grad.
      const Tensor& const_cs_prev_grad_tensor = *cs_prev_grad_tensor;
      functor::TensorAdd<Device, T>()(
          device, const_cs_prev_grad_tensor.flat<T>(),
          cs_grad_list[t].flat<T>(), cs_grad_tensor.flat<T>());

      // Combine previous h grad and h grad coming on top.
      const Tensor& const_h_prev_grad_tensor = *h_prev_grad_tensor;
      functor::TensorAdd<Device, T>()(
          device, const_h_prev_grad_tensor.flat<T>(), h_grad_list[t].flat<T>(),
          h_grad_tensor.flat<T>());

      const Tensor& const_cs_grad_tensor = cs_grad_tensor;
      const Tensor& const_h_grad_tensor = h_grad_tensor;

      Tensor* x_grad_tensor = x_grad_list[t];
      functor::BlockLSTMBprop<Device, T, USE_CUBLAS>(batch_size, input_size,
                                                     cell_size)(
          ctx, stream, device, use_peephole_, x_tensor.matrix<T>(),
          cs_prev_tensor2.matrix<T>(), h_prev_tensor2.matrix<T>(),
          w_tensor->matrix<T>(), wci_tensor->vec<T>(), wcf_tensor->vec<T>(),
          wco_tensor->vec<T>(), b_tensor->vec<T>(), xh_tensor.matrix<T>(),
          i_tensor.matrix<T>(), cs_tensor.matrix<T>(), f_tensor.matrix<T>(),
          o_tensor.matrix<T>(), ci_tensor.matrix<T>(), co_tensor.matrix<T>(),
          const_cs_grad_tensor.matrix<T>(), const_h_grad_tensor.matrix<T>(),
          do_tensor.matrix<T>(), dcs_tensor.matrix<T>(), dci_tensor.matrix<T>(),
          df_tensor.matrix<T>(), di_tensor.matrix<T>(),
          dicfo_tensor.matrix<T>(), cs_prev_grad_tensor->matrix<T>(),
          h_prev_grad_tensor->matrix<T>(), xh_grad_tensor.matrix<T>(),
          x_grad_tensor->matrix<T>(), w_grad_tensor->matrix<T>(),
          wci_grad_tensor->vec<T>(), wcf_grad_tensor->vec<T>(),
          wco_grad_tensor->vec<T>(), b_grad_tensor->vec<T>());
    }

    for (int64 t = seq_len_max; t < max_len_; ++t) {
      Tensor* x_grad_tensor = x_grad_list[t];
      functor::TensorZero<Device, T>()(device, x_grad_tensor->flat<T>());
    }
  }

 private:
  int64 max_len_;
  bool use_peephole_;
};

#define REGISTER_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("BlockLSTMGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BlockLSTMGradOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
// REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void TensorCopy<GPUDevice, T>::operator()(const GPUDevice& d,                \
                                            typename TTypes<T>::ConstFlat src, \
                                            typename TTypes<T>::Flat dst);     \
                                                                               \
  template <>                                                                  \
  void TensorAdd<GPUDevice, T>::operator()(                                    \
      const GPUDevice& d, typename TTypes<T>::ConstFlat a,                     \
      typename TTypes<T>::ConstFlat b, typename TTypes<T>::Flat c);            \
                                                                               \
  template <>                                                                  \
  void BlockLSTMBprop<GPUDevice, T, true>::operator()(                         \
      OpKernelContext* ctx, perftools::gputools::Stream* stream,               \
      const GPUDevice& d, bool use_peephole,                                   \
      typename TTypes<T>::ConstMatrix x,                                       \
      typename TTypes<T>::ConstMatrix cs_prev,                                 \
      typename TTypes<T>::ConstMatrix h_prev,                                  \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,     \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,      \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::Matrix xh,           \
      typename TTypes<T>::ConstMatrix i, typename TTypes<T>::ConstMatrix cs,   \
      typename TTypes<T>::ConstMatrix f, typename TTypes<T>::ConstMatrix o,    \
      typename TTypes<T>::ConstMatrix ci, typename TTypes<T>::ConstMatrix co,  \
      typename TTypes<T>::ConstMatrix cs_grad,                                 \
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,  \
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,          \
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,            \
      typename TTypes<T>::Matrix dicfo,                                        \
      typename TTypes<T>::Matrix cs_prev_grad,                                 \
      typename TTypes<T>::Matrix h_prev_grad,                                  \
      typename TTypes<T>::Matrix xh_grad, typename TTypes<T>::Matrix x_grad,   \
      typename TTypes<T>::Matrix w_grad, typename TTypes<T>::Vec wci_grad,     \
      typename TTypes<T>::Vec wcf_grad, typename TTypes<T>::Vec wco_grad,      \
      typename TTypes<T>::Vec b_grad);                                         \
                                                                               \
  extern template struct TensorCopy<GPUDevice, T>;                             \
  extern template struct TensorAdd<GPUDevice, T>;                              \
  extern template struct BlockLSTMBprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
// DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                           \
  REGISTER_KERNEL_BUILDER(Name("BlockLSTMGrad")          \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("seq_len_max") \
                              .TypeConstraint<T>("T"),   \
                          BlockLSTMGradOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
// REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
