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
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/contrib/rnn/kernels/gru_ops.h"
#include "tensorflow/core/framework/op_kernel.h"

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
// TODO(gitegaurav) : Refactor the matmul operation inside the kernel. Make
// similar changes in the LSTMBlockCell. Create a new file which contains matmul
// functionality. It should perform matmul operation using CuBlas when the Cuda
// support is present otherwise using eigentensors.
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
}  // end namespace functor

template <typename Device, typename T, bool USE_CUBLAS>
class GRUCellBlockOp : public OpKernel {
 public:
  explicit GRUCellBlockOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  // TODO(gitegaurav) Replace the input checks with some smarter function.
  void Compute(OpKernelContext* ctx) override {
    // Grab the input tensors.
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_ru", &w_ru_tensor));

    const Tensor* w_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_c", &w_c_tensor));

    const Tensor* b_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_ru", &b_ru_tensor));

    const Tensor* b_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_c", &b_c_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 cell_size = h_prev_tensor->dim_size(1);

    // Sanity checks for input shapes.

    // Shape of 'h' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("h_prev.dims(1) != cell_size: ",
                                        h_prev_tensor->dim_size(1), " vs. ",
                                        cell_size));

    // Shape of 'w_ru' must be [input_size+cell_size, 2*cell_size]
    OP_REQUIRES(ctx, w_ru_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_ru.dim_size(0) != input_size + cell_size: ",
                    w_ru_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(ctx, w_ru_tensor->dim_size(1) == cell_size * 2,
                errors::InvalidArgument("w_ru.dim_size(1) != cell_size * 2: ",
                                        w_ru_tensor->dim_size(1), " vs. ",
                                        cell_size * 2));

    // Shape of 'w_c' must be [input_size+cell_size, cell_size]
    OP_REQUIRES(ctx, w_c_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_c.dim_size(0) != input_size + cell_size: ",
                    w_c_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(
        ctx, w_c_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("w_c.dim_size(1) != cell_size: ",
                                w_c_tensor->dim_size(1), " vs. ", cell_size));

    // Shape of 'b_ru' must be [2*cell_size]
    OP_REQUIRES(ctx, b_ru_tensor->dim_size(0) == cell_size * 2,
                errors::InvalidArgument("b_ru.dim_size(0) != cell_size * 2: ",
                                        b_ru_tensor->dim_size(0), " vs. ",
                                        cell_size * 2));

    OP_REQUIRES(ctx, b_ru_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_ru must be 1",
                                        b_ru_tensor->dims(), " vs. 1", 1));
    // Shape of 'b_c' must be [cell_size]
    OP_REQUIRES(
        ctx, b_c_tensor->dim_size(0) == cell_size,
        errors::InvalidArgument("b_c.dim_size(0) != cell_size: ",
                                b_c_tensor->dim_size(0), " vs. ", cell_size));
    OP_REQUIRES(ctx, b_c_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_c must be 1",
                                        b_c_tensor->dims(), " vs. 1"));

    // Create output tensors.
    Tensor* r_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("r", TensorShape({batch_size, cell_size}),
                                  &r_tensor));

    Tensor* u_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("u", TensorShape({batch_size, cell_size}),
                                  &u_tensor));

    Tensor* c_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("c", TensorShape({batch_size, cell_size}),
                                  &c_tensor));

    Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("h", TensorShape({batch_size, cell_size}),
                                  &h_tensor));

    // Allocate temp tensors.
    Tensor x_h_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &x_h_prev_tensor));

    Tensor x_h_prevr_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &x_h_prevr_tensor));

    Tensor r_u_bar_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                      TensorShape({batch_size, 2 * cell_size}),
                                      &r_u_bar_tensor));

    const Device& device = ctx->eigen_device<Device>();

    perftools::gputools::Stream* stream =
        std::is_same<Device, GPUDevice>::value
            ? ctx->op_device_context()->stream()
            : nullptr;

    functor::GRUBlockCellFprop<Device, T, USE_CUBLAS>(batch_size, input_size,
                                                      cell_size)(
        ctx, stream, device, x_tensor->matrix<T>(), h_prev_tensor->matrix<T>(),
        w_ru_tensor->matrix<T>(), w_c_tensor->matrix<T>(),
        b_ru_tensor->vec<T>(), b_c_tensor->vec<T>(), r_u_bar_tensor.matrix<T>(),
        r_tensor->matrix<T>(), u_tensor->matrix<T>(), c_tensor->matrix<T>(),
        h_tensor->matrix<T>(), x_h_prev_tensor.matrix<T>(),
        x_h_prevr_tensor.matrix<T>());
  }
};

// Register the Block GRU cell kernel for CPU.
#define REGISTER_KERNEL(T)                                            \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("GRUBlockCell").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      GRUCellBlockOp<CPUDevice, T, false>);

REGISTER_KERNEL(float);
#undef REGISTER_KERNEL

template <typename Device, typename T, bool USE_CUBLAS>
class GRUBlockCellGradOp : public OpKernel {
 public:
  explicit GRUBlockCellGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Grab the input tensors.
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_ru", &w_ru_tensor));

    const Tensor* w_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_c", &w_c_tensor));

    const Tensor* b_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_ru", &b_ru_tensor));

    const Tensor* b_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_c", &b_c_tensor));

    const Tensor* r_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("r", &r_tensor));

    const Tensor* u_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("u", &u_tensor));

    const Tensor* c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("c", &c_tensor));

    const Tensor* d_h_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("d_h", &d_h_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 cell_size = h_prev_tensor->dim_size(1);

    // Sanity checks for input shapes.

    // Shape of 'h_prev' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("h_prev.dims(1) != cell_size: ",
                                        h_prev_tensor->dim_size(1), " vs. ",
                                        cell_size));

    // Shape of 'w_ru' must be [input_size+cell_size, 2*cell_size]
    OP_REQUIRES(ctx, w_ru_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_ru.dim_size(0) != input_size + cell_size: ",
                    w_ru_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(ctx, w_ru_tensor->dim_size(1) == cell_size * 2,
                errors::InvalidArgument("w_ru.dim_size(1) != cell_size * 2: ",
                                        w_ru_tensor->dim_size(1), " vs. ",
                                        cell_size * 2));

    // Shape of 'w_c' must be [input_size+cell_size, cell_size]
    OP_REQUIRES(ctx, w_c_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_c.dim_size(0) != input_size + cell_size: ",
                    w_c_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(
        ctx, w_c_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("w_c.dim_size(1) != cell_size: ",
                                w_c_tensor->dim_size(1), " vs. ", cell_size));

    // Shape of 'b_ru' must be [2*cell_size]
    OP_REQUIRES(ctx, b_ru_tensor->dim_size(0) == cell_size * 2,
                errors::InvalidArgument("b_ru.dim_size(0) != cell_size * 2: ",
                                        b_ru_tensor->dim_size(0), " vs. ",
                                        cell_size * 2));

    OP_REQUIRES(ctx, b_ru_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_ru must be 1",
                                        b_ru_tensor->dims(), " vs. 1"));

    // Shape of 'b_c' must be [cell_size]
    OP_REQUIRES(
        ctx, b_c_tensor->dim_size(0) == cell_size,
        errors::InvalidArgument("b_c.dim_size(0) != cell_size: ",
                                b_c_tensor->dim_size(0), " vs. ", cell_size));

    OP_REQUIRES(ctx, b_c_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_c must be 1 ",
                                        b_c_tensor->dims(), " vs. 1"));

    // Shape of 'r' must be [batch_size, cell_size]
    OP_REQUIRES(
        ctx, r_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("r.dims(0) != batch_size: ",
                                r_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(
        ctx, r_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("r.dims(1) != cell_size: ",
                                r_tensor->dim_size(1), " vs. ", cell_size));

    // Shape of 'u' must be [batch_size, cell_size]
    OP_REQUIRES(
        ctx, u_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("u.dims(0) != batch_size: ",
                                u_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(
        ctx, u_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("u.dims(1) != cell_size: ",
                                u_tensor->dim_size(1), " vs. ", cell_size));

    // Shape of 'c' must be [batch_size, cell_size]
    OP_REQUIRES(
        ctx, c_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("c.dims(0) != batch_size: ",
                                c_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(
        ctx, c_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("c.dims(1) != cell_size: ",
                                c_tensor->dim_size(1), " vs. ", cell_size));

    // Shape of 'd_h' must be [batch_size, cell_size]
    OP_REQUIRES(
        ctx, d_h_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument("d_h.dims(0) != batch_size: ",
                                d_h_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(
        ctx, d_h_tensor->dim_size(1) == cell_size,
        errors::InvalidArgument("d_h.dims(1) != cell_size: ",
                                d_h_tensor->dim_size(1), " vs. ", cell_size));

    // Create output tensors.
    Tensor* d_x_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("d_x", TensorShape({batch_size, input_size}),
                                  &d_x_tensor));

    Tensor* d_h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            "d_h_prev", TensorShape({batch_size, cell_size}),
                            &d_h_prev_tensor));

    Tensor* d_c_bar_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            "d_c_bar", TensorShape({batch_size, cell_size}),
                            &d_c_bar_tensor));

    Tensor* d_r_bar_u_bar_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("d_r_bar_u_bar",
                                  TensorShape({batch_size, 2 * cell_size}),
                                  &d_r_bar_u_bar_tensor));

    // Allocate temp tensors.
    Tensor d_r_bar_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &d_r_bar_tensor));

    Tensor d_u_bar_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &d_u_bar_tensor));

    Tensor d_h_prevr_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &d_h_prevr_tensor));

    Tensor d_x_component_1_h_prev_compenent_1;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &d_x_component_1_h_prev_compenent_1));

    Tensor d_x_component_2_h_prevr;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &d_x_component_2_h_prevr));

    const Device& device = ctx->eigen_device<Device>();
    perftools::gputools::Stream* stream =
        std::is_same<Device, GPUDevice>::value
            ? ctx->op_device_context()->stream()
            : nullptr;

    functor::GRUBlockCellBprop<Device, T, USE_CUBLAS>(batch_size, input_size,
                                                      cell_size)(
        ctx, stream, device, x_tensor->matrix<T>(), h_prev_tensor->matrix<T>(),
        w_ru_tensor->matrix<T>(), w_c_tensor->matrix<T>(),
        b_ru_tensor->vec<T>(), b_c_tensor->vec<T>(), r_tensor->matrix<T>(),
        u_tensor->matrix<T>(), c_tensor->matrix<T>(), d_h_tensor->matrix<T>(),
        d_x_tensor->matrix<T>(), d_h_prev_tensor->matrix<T>(),
        d_c_bar_tensor->matrix<T>(), d_r_bar_u_bar_tensor->matrix<T>(),
        d_r_bar_tensor.matrix<T>(), d_u_bar_tensor.matrix<T>(),
        d_h_prevr_tensor.matrix<T>(),
        d_x_component_1_h_prev_compenent_1.matrix<T>(),
        d_x_component_2_h_prevr.matrix<T>());
  }
};

// Register the gradient kernel for CPU.
#define REGISTER_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("GRUBlockCellGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      GRUBlockCellGradOp<CPUDevice, T, false>);

REGISTER_KERNEL(float);
#undef REGISTER_KERNEL

// GPU support.
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

// Forward declare the GPU Fprop functor.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                   \
  template <>                                                                 \
  void GRUBlockCellFprop<GPUDevice, T, true>::operator()(                     \
      OpKernelContext* ctx, perftools::gputools::Stream* stream,              \
      const GPUDevice& d, typename TTypes<T>::ConstMatrix x,                  \
      typename TTypes<T>::ConstMatrix h_prev,                                 \
      typename TTypes<T>::ConstMatrix w_ru,                                   \
      typename TTypes<T>::ConstMatrix w_c, typename TTypes<T>::ConstVec b_ru, \
      typename TTypes<T>::ConstVec b_c, typename TTypes<T>::Matrix r_u_bar,   \
      typename TTypes<T>::Matrix r, typename TTypes<T>::Matrix u,             \
      typename TTypes<T>::Matrix c, typename TTypes<T>::Matrix h,             \
      typename TTypes<T>::Matrix x_h_prev,                                    \
      typename TTypes<T>::Matrix x_h_prevr);                                  \
  extern template struct GRUBlockCellFprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

// Register the Block GRU cell kernel for GPU.
#define REGISTER_GPU_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("GRUBlockCell").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      GRUCellBlockOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
#undef REGISTER_GPU_KERNEL

// Forward declare the GPU Bprop functor.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void GRUBlockCellBprop<GPUDevice, T, true>::operator()(                      \
      OpKernelContext* ctx, perftools::gputools::Stream* stream,               \
      const GPUDevice& d, typename TTypes<T>::ConstMatrix x,                   \
      typename TTypes<T>::ConstMatrix h, typename TTypes<T>::ConstMatrix w_ru, \
      typename TTypes<T>::ConstMatrix w_c, typename TTypes<T>::ConstVec b_ru,  \
      typename TTypes<T>::ConstVec b_c, typename TTypes<T>::ConstMatrix r,     \
      typename TTypes<T>::ConstMatrix u, typename TTypes<T>::ConstMatrix c,    \
      typename TTypes<T>::ConstMatrix d_h, typename TTypes<T>::Matrix d_x,     \
      typename TTypes<T>::Matrix d_h_prev, typename TTypes<T>::Matrix d_c_bar, \
      typename TTypes<T>::Matrix d_r_bar_u_bar,                                \
      typename TTypes<T>::Matrix d_r_bar, typename TTypes<T>::Matrix d_u_bar,  \
      typename TTypes<T>::Matrix d_h_prevr,                                    \
      typename TTypes<T>::Matrix d_x_comp1_h_prev_comp1,                       \
      typename TTypes<T>::Matrix d_x_comp2_and_h_prevr);                       \
  extern template struct GRUBlockCellBprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

// Register the gradient kernel for GPU.
#define REGISTER_GPU_KERNEL(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("GRUBlockCellGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      GRUBlockCellGradOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
