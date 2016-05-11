#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/contrib/rnn/kernels/lstm_ops.h"

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/stream.h"
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
void TensorCuBlasGemm<T>::operator()(
    OpKernelContext* ctx, perftools::gputools::Stream* stream,
    bool transa, bool transb, uint64 m, uint64 n, uint64 k, T alpha,
    const T* a, int lda, const T* b, int ldb, T beta, T *c, int ldc) {
#if GOOGLE_CUDA
  perftools::gputools::blas::Transpose trans[] = {
      perftools::gputools::blas::Transpose::kNoTranspose,
      perftools::gputools::blas::Transpose::kTranspose};

  auto a_ptr = AsDeviceMemory(a);
  auto b_ptr = AsDeviceMemory(b);
  auto c_ptr = AsDeviceMemory(c);

  bool blas_launch_status = stream->ThenBlasGemm(
      trans[transa], trans[transb], m, n, k, alpha, a_ptr, lda, b_ptr, ldb,
      beta, &c_ptr, ldc).ok();
  OP_REQUIRES(ctx, blas_launch_status, errors::Aborted("CuBlasGemm failed!"));
#else
  ctx->SetStatus(errors::InvalidArgument("CuBlasGemm needs CUDA."));
#endif
}

template struct TensorCuBlasGemm<float>;
template struct TensorCuBlasGemm<double>;
}  // end namespace functor

template <typename Device, typename T, bool USE_CUBLAS>
class LSTMCellBlockOp : public OpKernel {
 public:
  explicit LSTMCellBlockOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* states_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("states_prev", &states_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 states_size = cell_size_ * 2;

    perftools::gputools::Stream* stream =
        ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, states_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("states_prev.dims(0) != batch_size: ",
                                        states_prev_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, states_prev_tensor->dim_size(1) == states_size,
                errors::InvalidArgument("states_prev.dims(1) != cell_size * 2: ",
                                        states_prev_tensor->dim_size(1),
                                        " vs. ", states_size));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size_,
        errors::InvalidArgument("w.dim_size(0) != input_size + cell_size: ",
                                w_tensor->dim_size(0), " vs. ",
                                input_size + cell_size_));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size_ * 4,
                errors::InvalidArgument("w.dim_size(1) != cell_size * 4: ",
                                        w_tensor->dim_size(1),
                                        " vs. ", cell_size_ * 4));

    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size_ * 4,
                errors::InvalidArgument("b.dim_size(0) != cell_size * 4: ",
                                        b_tensor->dim_size(0),
                                        " vs. ", cell_size_ * 4));

    // Allocate our output tensors.
    Tensor* i_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("i",
          TensorShape({batch_size, cell_size_}), &i_tensor));

    Tensor* cs_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("cs",
          TensorShape({batch_size, cell_size_}), &cs_tensor));

    Tensor* f_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("f",
          TensorShape({batch_size, cell_size_}), &f_tensor));

    Tensor* o_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("o",
          TensorShape({batch_size, cell_size_}), &o_tensor));

    Tensor* ci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("ci",
          TensorShape({batch_size, cell_size_}), &ci_tensor));

    Tensor* co_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("co",
          TensorShape({batch_size, cell_size_}), &co_tensor));

    Tensor* states_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("states",
        TensorShape({batch_size, states_size}), &states_tensor));

    Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("h",
          TensorShape({batch_size, cell_size_}), &h_tensor));

    // Allocate our temp tensors.
    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, input_size + cell_size_}), &xh_tensor));

    Tensor cs_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &cs_prev_tensor));

    Tensor h_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &h_prev_tensor));

    Tensor icfo_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_ * 4}), &icfo_tensor));

    functor::LSTMCellBlockFprop<Device, T, USE_CUBLAS>(
        batch_size, input_size, cell_size_)(
        ctx, stream, ctx->eigen_device<Device>(), forget_bias_,
        x_tensor->matrix<T>(), states_prev_tensor->matrix<T>(),
        w_tensor->matrix<T>(), b_tensor->vec<T>(), cs_prev_tensor.matrix<T>(),
        h_prev_tensor.matrix<T>(), xh_tensor.matrix<T>(),
        i_tensor->matrix<T>(), cs_tensor->matrix<T>(), f_tensor->matrix<T>(),
        o_tensor->matrix<T>(), ci_tensor->matrix<T>(), co_tensor->matrix<T>(),
        icfo_tensor.matrix<T>(), states_tensor->matrix<T>(),
        h_tensor->matrix<T>());
  }

 private:
  int64 cell_size_;
  float forget_bias_;
};

#define REGISTER_KERNEL(T)                              \
  REGISTER_KERNEL_BUILDER(Name("LSTMCellBlock")         \
                              .Device(DEVICE_CPU)       \
                              .TypeConstraint<T>("T"),  \
                          LSTMCellBlockOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                       \
  template <>                                                     \
  void TensorMemZero<GPUDevice, T>::operator()(                   \
      const GPUDevice& d, typename TTypes<T>::Vec x);             \
                                                                  \
  template <>                                                     \
  void TensorMemCopy<GPUDevice, T>::operator()(                   \
      const GPUDevice& d, typename TTypes<T>::ConstVec in,        \
      typename TTypes<T>::Vec out);                               \
                                                                  \
  template <>                                                     \
  void LSTMCellBlockFprop<GPUDevice, T, true>::operator()(        \
      OpKernelContext* ctx, perftools::gputools::Stream* stream,  \
      const GPUDevice& d, const T forget_bias,                    \
      typename TTypes<T>::ConstMatrix x,                          \
      typename TTypes<T>::ConstMatrix states_prev,                \
      typename TTypes<T>::ConstMatrix w,                          \
      typename TTypes<T>::ConstVec b,                             \
      typename TTypes<T>::Matrix cs_prev,                         \
      typename TTypes<T>::Matrix h_prev,                          \
      typename TTypes<T>::Matrix xh,                              \
      typename TTypes<T>::Matrix i,                               \
      typename TTypes<T>::Matrix cs,                              \
      typename TTypes<T>::Matrix f,                               \
      typename TTypes<T>::Matrix o,                               \
      typename TTypes<T>::Matrix ci,                              \
      typename TTypes<T>::Matrix co,                              \
      typename TTypes<T>::Matrix icfo,                            \
      typename TTypes<T>::Matrix states,                          \
      typename TTypes<T>::Matrix h);                              \
                                                                  \
  extern template struct TensorMemZero<GPUDevice, T>;             \
  extern template struct TensorMemCopy<GPUDevice, T>;             \
  extern template struct LSTMCellBlockFprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                          \
  REGISTER_KERNEL_BUILDER(Name("LSTMCellBlock")         \
                              .Device(DEVICE_GPU)       \
                              .TypeConstraint<T>("T"),  \
                          LSTMCellBlockOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

template <typename Device, typename T, bool USE_CUBLAS>
class LSTMCellBlockGradOp : public OpKernel {
 public:
  explicit LSTMCellBlockGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* states_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("states_prev", &states_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

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

    const Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h", &h_tensor));

    const Tensor* h_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_grad", &h_grad_tensor));

    const Tensor* states_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("states_grad", &states_grad_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 states_size = cell_size_ * 2;

    const Device& device = ctx->eigen_device<Device>();
    perftools::gputools::Stream* stream =
        ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, states_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("states_prev.dims(0) != batch_size: ",
                                        states_prev_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, states_prev_tensor->dim_size(1) == states_size,
        errors::InvalidArgument("states_prev.dims(1) != cell_size * 2: ",
                                states_prev_tensor->dim_size(1),
                                " vs. ", states_size));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size_,
        errors::InvalidArgument("w.dim_size(0) != input_size + cell_size: ",
                                w_tensor->dim_size(0), " vs. ",
                                input_size + cell_size_));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size_ * 4,
                errors::InvalidArgument("w.dim_size(1) != cell_size * 4: ",
                                        w_tensor->dim_size(1),
                                        " vs. ", cell_size_ * 4));

    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size_ * 4,
                errors::InvalidArgument("b.dim_size(0) != cell_size * 4: ",
                                        b_tensor->dim_size(0),
                                        " vs. ", cell_size_ * 4));

    OP_REQUIRES(ctx, i_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("i.dim_size(0) != batch_size: ",
                                        i_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, i_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("i.dim_size(1) != cell_size_: ",
                                        i_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, cs_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs.dim_size(0) != batch_size: ",
                                        cs_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, cs_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("cs.dim_size(1) != cell_size_: ",
                                        cs_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, f_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("f.dim_size(0) != batch_size: ",
                                        f_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, f_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("i.dim_size(1) != cell_size_: ",
                                        f_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, o_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("o.dim_size(0) != batch_size: ",
                                        o_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, o_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("o.dim_size(1) != cell_size_: ",
                                        o_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, ci_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("ci.dim_size(0) != batch_size: ",
                                        ci_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, ci_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("ci.dim_size(1) != cell_size_: ",
                                        ci_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, co_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("co.dim_size(0) != batch_size: ",
                                        co_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, co_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("co.dim_size(1) != cell_size_: ",
                                        co_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, h_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h.dim_size(0) != batch_size: ",
                                        h_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, h_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("h.dim_size(1) != cell_size_: ",
                                        h_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, h_grad_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_grad_tensor.dims(0) != batch_size: ",
                                        h_grad_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, h_grad_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("h_grad_tensor.dims(1) != state_size: ",
                                        h_grad_tensor->dim_size(1),
                                        " vs. ", cell_size_));

    OP_REQUIRES(ctx, states_grad_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("states_grad.dims(0) != batch_size: ",
                                        states_grad_tensor->dim_size(0),
                                        " vs. ", batch_size));
    OP_REQUIRES(ctx, states_grad_tensor->dim_size(1) == states_size,
        errors::InvalidArgument("states_grad.dims(1) != cell_size * 2: ",
                                states_grad_tensor->dim_size(1),
                                " vs. ", states_size));

    // Allocate our output tensors.
    Tensor* x_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("x_grad",
        TensorShape({batch_size, input_size}), &x_grad_tensor));

    Tensor* states_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("states_prev_grad",
        TensorShape({batch_size, states_size}), &states_prev_grad_tensor));

    Tensor* dicfo_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("dicfo",
        TensorShape({batch_size, cell_size_ * 4}), &dicfo_tensor));

    Tensor* xh_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("xh",
        TensorShape({batch_size, input_size + cell_size_}), &xh_tensor));

    // Allocate our temp tensors.
    Tensor cs_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &cs_prev_tensor));

    Tensor states_c_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &states_c_grad_tensor));

    Tensor states_h_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &states_h_grad_tensor));

    Tensor xh_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, input_size + cell_size_}), &xh_grad_tensor));

    Tensor dh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &dh_tensor));

    Tensor do_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &do_tensor));

    Tensor dcs_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &dcs_tensor));

    Tensor dci_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &dci_tensor));

    Tensor df_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &df_tensor));

    Tensor di_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size_}), &di_tensor));

    Tensor w_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({0, 0}), &w_grad_tensor));

    Tensor b_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({0}), &b_grad_tensor));

    functor::LSTMCellBlockBprop<Device, T, USE_CUBLAS>(
        batch_size, input_size, cell_size_)(
        ctx, stream, device, true, x_tensor->matrix<T>(),
        states_prev_tensor->matrix<T>(), w_tensor->matrix<T>(),
        b_tensor->vec<T>(), i_tensor->matrix<T>(), cs_tensor->matrix<T>(),
        f_tensor->matrix<T>(), o_tensor->matrix<T>(), ci_tensor->matrix<T>(),
        co_tensor->matrix<T>(), h_tensor->matrix<T>(),
        states_grad_tensor->matrix<T>(), h_grad_tensor->matrix<T>(),
        cs_prev_tensor.matrix<T>(), states_c_grad_tensor.matrix<T>(),
        states_h_grad_tensor.matrix<T>(), xh_tensor->matrix<T>(),
        xh_grad_tensor.matrix<T>(), x_grad_tensor->matrix<T>(),
        dh_tensor.matrix<T>(), do_tensor.matrix<T>(), dcs_tensor.matrix<T>(),
        dci_tensor.matrix<T>(), df_tensor.matrix<T>(), di_tensor.matrix<T>(),
        dicfo_tensor->matrix<T>(), states_prev_grad_tensor->matrix<T>(),
        w_grad_tensor.matrix<T>(), b_grad_tensor.vec<T>());
  }

 protected:
  int64 cell_size_;
};

#define REGISTER_KERNEL(T)                              \
  REGISTER_KERNEL_BUILDER(Name("LSTMCellBlockGrad")     \
                              .Device(DEVICE_CPU)       \
                              .TypeConstraint<T>("T"),  \
                          LSTMCellBlockGradOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                       \
  template <>                                                                     \
  void LSTMCellBlockBprop<GPUDevice, T, true>::operator()(                        \
      OpKernelContext* ctx, perftools::gputools::Stream* stream,                  \
      const GPUDevice& d, bool parallel_dw, typename TTypes<T>::ConstMatrix x,    \
      typename TTypes<T>::ConstMatrix states_prev,                                \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec b,          \
      typename TTypes<T>::ConstMatrix i, typename TTypes<T>::ConstMatrix cs,      \
      typename TTypes<T>::ConstMatrix f, typename TTypes<T>::ConstMatrix o,       \
      typename TTypes<T>::ConstMatrix ci, typename TTypes<T>::ConstMatrix co,     \
      typename TTypes<T>::ConstMatrix h,                                          \
      typename TTypes<T>::ConstMatrix states_grad,                                \
      typename TTypes<T>::ConstMatrix h_grad,                                     \
      typename TTypes<T>::Matrix cs_prev,                                         \
      typename TTypes<T>::Matrix states_c_grad,                                   \
      typename TTypes<T>::Matrix states_h_grad,                                   \
      typename TTypes<T>::Matrix xh, typename TTypes<T>::Matrix xh_grad,          \
      typename TTypes<T>::Matrix x_grad, typename TTypes<T>::Matrix dh,           \
      typename TTypes<T>::Matrix do_, typename TTypes<T>::Matrix dcs,             \
      typename TTypes<T>::Matrix dci, typename TTypes<T>::Matrix df,              \
      typename TTypes<T>::Matrix di, typename TTypes<T>::Matrix dicfo,            \
      typename TTypes<T>::Matrix states_prev_grad,                                \
      typename TTypes<T>::Matrix w_grad, typename TTypes<T>::Vec b_grad);         \
                                                                                  \
  extern template struct LSTMCellBlockBprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU_KERNEL(T)                          \
  REGISTER_KERNEL_BUILDER(Name("LSTMCellBlockGrad")     \
                              .Device(DEVICE_GPU)       \
                              .TypeConstraint<T>("T"),  \
                          LSTMCellBlockGradOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
