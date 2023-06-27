/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/platform/prefetch.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


namespace functor {
template <typename Device, typename T>
struct SelectScalarHandler;
}  // namespace functor

template <typename Device, typename T>
class SelectOp : public OpKernel {
 public:
  explicit SelectOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* cond = &ctx->input(0);
    const Tensor* then = &ctx->input(1);
    const Tensor* else_ = &ctx->input(2);

    if (TensorShapeUtils::IsScalar(cond->shape())) {
      ComputeScalar(ctx, cond, then, else_);
      return;
    }

    bool broadcasting = (TensorShapeUtils::IsVector(cond->shape()) &&
                         !TensorShapeUtils::IsVector(then->shape()));

    if (broadcasting) {
      ComputeBroadcasting(ctx, cond, then, else_);
    } else {
      ComputeElementwise(ctx, cond, then, else_);
    }
  }

 protected:
  void ComputeBroadcasting(OpKernelContext* ctx, const Tensor* cond,
                           const Tensor* then, const Tensor* else_) {
    // Preliminary validation of sizes.
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(cond->shape()),
        errors::InvalidArgument("'cond' must be a vector, but saw shape: ",
                                cond->shape().DebugString()));
    OP_REQUIRES(
        ctx,
        FastBoundsCheck(cond->NumElements(),
                        std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("cond vector larger than ",
                                std::numeric_limits<Eigen::DenseIndex>::max()));
    OP_REQUIRES(
        ctx,
        FastBoundsCheck(then->flat_outer_dims<T>().dimension(1),
                        std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("flat outer dims dim 1 size >= ",
                                std::numeric_limits<Eigen::DenseIndex>::max()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(then->shape()),
                errors::InvalidArgument(
                    "'then' must be at least a vector, but saw shape: ",
                    then->shape().DebugString()));
    OP_REQUIRES(
        ctx, then->shape().dim_size(0) == cond->NumElements(),
        errors::InvalidArgument(
            "Number of batches of 'then' must match size of 'cond', but saw: ",
            then->shape().dim_size(0), " vs. ", cond->NumElements()));
    OP_REQUIRES(
        ctx, then->shape().IsSameSize(else_->shape()),
        errors::InvalidArgument(
            "'then' and 'else' must have the same size.  but received: ",
            then->shape().DebugString(), " vs. ",
            else_->shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {"t", "e"}, "output", then->shape(), &output));
    if (output->NumElements() > 0) {
      functor::BatchSelectFunctor<Device, T> func;
      func(ctx->eigen_device<Device>(), output->flat_outer_dims<T>(),
           cond->vec<bool>(), then->flat_outer_dims<T>(),
           else_->flat_outer_dims<T>());
    }
  }

  void ComputeElementwise(OpKernelContext* ctx, const Tensor* cond,
                          const Tensor* then, const Tensor* else_) {
    if (!ctx->ValidateInputsAreSameShape(this)) return;
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {"t", "e"}, "output", then->shape(), &output));
    if (output->NumElements() > 0) {
      functor::SelectFunctor<Device, T> func;
      func(ctx->eigen_device<Device>(), output->flat<T>(), cond->flat<bool>(),
           then->flat<T>(), else_->flat<T>());
    }
  }

  void ComputeScalar(OpKernelContext* ctx, const Tensor* cond,
                     const Tensor* then, const Tensor* else_) {
    OP_REQUIRES(
        ctx, then->shape().IsSameSize(else_->shape()),
        errors::InvalidArgument(
            "'then' and 'else' must have the same size.  but received: ",
            then->shape().DebugString(), " vs. ",
            else_->shape().DebugString()));

    functor::SelectScalarHandler<Device, T> handler;
    handler(ctx, cond, then, else_);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SelectOp);
};
template <typename Device, typename T>
class SelectV2Op : public OpKernel {
 public:
  explicit SelectV2Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* cond = &ctx->input(0);
    const Tensor* then = &ctx->input(1);
    const Tensor* else_ = &ctx->input(2);

    // The `cond`, `then`, and `else` are broadcastable (bcast.IsValid()),
    // This matches the behavior of numpy.
    BCastList<3> bcast({cond->shape().dim_sizes(), then->shape().dim_sizes(),
                        else_->shape().dim_sizes()},
                       false);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "condition ", cond->shape().DebugString(), ", then ",
                    then->shape().DebugString(), ", and else ",
                    else_->shape().DebugString(), " must be broadcastable"));

    // Broadcast `cond`, `then` and `else` to combined shape,
    // in order to obtain the reshape.
    BCast cond_bcast(bcast.output_shape(), cond->shape().dim_sizes(), false);
    BCast then_bcast(bcast.output_shape(), then->shape().dim_sizes(), false);
    BCast else_bcast(bcast.output_shape(), else_->shape().dim_sizes(), false);
    OP_REQUIRES(
        ctx,
        cond_bcast.IsValid() && then_bcast.IsValid() && else_bcast.IsValid(),
        errors::InvalidArgument("condition ", cond->shape().DebugString(),
                                ", then ", then->shape().DebugString(),
                                ", and else ", else_->shape().DebugString(),
                                " must be broadcastable"));

    // Combined shape should be the final shape.
    OP_REQUIRES(
        ctx,
        cond_bcast.output_shape() == bcast.output_shape() &&
            then_bcast.output_shape() == bcast.output_shape() &&
            else_bcast.output_shape() == bcast.output_shape(),
        errors::InvalidArgument("condition ", cond->shape().DebugString(),
                                ", then ", then->shape().DebugString(),
                                ", and else ", else_->shape().DebugString(),
                                " must be broadcastable to the same shape"));

    Tensor* output = nullptr;
    const TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {"t", "e"}, "output", output_shape, &output));

    if (output->NumElements() == 0) {
      return;
    }

#define HANDLE_DIM(NDIMS)                                            \
  {                                                                  \
    functor::BCastSelectFunctor<Device, T, NDIMS> func;              \
    func(ctx->eigen_device<Device>(),                                \
         output->shaped<T, NDIMS>(bcast.result_shape()),             \
         cond->template shaped<bool, NDIMS>(cond_bcast.y_reshape()), \
         then->template shaped<T, NDIMS>(then_bcast.y_reshape()),    \
         else_->template shaped<T, NDIMS>(else_bcast.y_reshape()),   \
         BCast::ToIndexArray<NDIMS>(cond_bcast.y_bcast()),           \
         BCast::ToIndexArray<NDIMS>(then_bcast.y_bcast()),           \
         BCast::ToIndexArray<NDIMS>(else_bcast.y_bcast()));          \
  }

    const int ndims = static_cast<int>(bcast.result_shape().size());
    switch (ndims) {
      case 1:
        HANDLE_DIM(1);
        break;
      case 2:
        HANDLE_DIM(2);
        break;
      case 3:
        HANDLE_DIM(3);
        break;
      case 4:
        HANDLE_DIM(4);
        break;
      case 5:
        HANDLE_DIM(5);
        break;
      case 6:
        HANDLE_DIM(6);
        break;
      case 7:
        HANDLE_DIM(7);
        break;
      case 8:
        HANDLE_DIM(8);
        break;
      default:
        ctx->SetStatus(errors::Unimplemented(
            "Broadcast between ", ctx->input(0).shape().DebugString(), " and ",
            ctx->input(1).shape().DebugString(), " is not supported yet."));
        break;
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SelectV2Op);
};

#define REGISTER_SELECT(type)                                        \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Select").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      SelectOp<CPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SelectV2").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SelectV2Op<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_SELECT);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)

// Registration of the GPU implementations.
#define REGISTER_SELECT_GPU(type)                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Select").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      SelectOp<GPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SelectV2").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SelectV2Op<GPUDevice, type>);

REGISTER_SELECT_GPU(bool);
REGISTER_SELECT_GPU(Eigen::half);
REGISTER_SELECT_GPU(float);
REGISTER_SELECT_GPU(double);
REGISTER_SELECT_GPU(int32);
REGISTER_SELECT_GPU(int64);
REGISTER_SELECT_GPU(complex64);
REGISTER_SELECT_GPU(complex128);

#undef REGISTER_SELECT_GPU

#else

#define REGISTER_SELECT_GPU(type)                                  \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Select").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SelectOp<GPUDevice, type>);

REGISTER_SELECT_GPU(bool);
REGISTER_SELECT_GPU(Eigen::half);
REGISTER_SELECT_GPU(float);
REGISTER_SELECT_GPU(double);
REGISTER_SELECT_GPU(int32);
REGISTER_SELECT_GPU(int64_t);
REGISTER_SELECT_GPU(complex64);
REGISTER_SELECT_GPU(complex128);

#undef REGISTER_SELECT_GPU
#endif

REGISTER_KERNEL_BUILDER(
    Name("Select").Device(DEVICE_GPU).TypeConstraint<bfloat16>("T"),
    SelectOp<GPUDevice, bfloat16>);
REGISTER_KERNEL_BUILDER(
    Name("SelectV2").Device(DEVICE_GPU).TypeConstraint<bfloat16>("T"),
    SelectV2Op<GPUDevice, bfloat16>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


namespace functor {

// CPU Specializations of Select functors.
template <typename Device, typename T>
struct SelectFunctorBase {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    Assign(d, out, cond_flat.select(then_flat, else_flat));
  }
};

template <typename T>
struct SelectFunctor<CPUDevice, T> : SelectFunctorBase<CPUDevice, T> {};

template <typename Device, typename T>
struct SelectScalarHandler {
  void operator()(OpKernelContext* ctx, const Tensor* cond, const Tensor* then,
                  const Tensor* else_) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {"t", "e"}, "output", then->shape(), &output));

    if (output->NumElements() > 0) {
      functor::SelectScalarFunctor<Device, T> func;
      TTypes<bool>::ConstScalar cond_scalar = cond->scalar<bool>();
      func(ctx->eigen_device<Device>(), output->flat<T>(), cond_scalar,
           then->flat<T>(), else_->flat<T>());
    }
  }
};

// Specialization for CPU device. Forward input to output depending on the
// `cond` value.
// TODO(sjhwang): Consider specializing for GPUDevice as well by using
// GPUDevice::memcpyDeviceToHost() to fetch bool value.
template <typename T>
struct SelectScalarHandler<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, const Tensor* cond, const Tensor* then,
                  const Tensor* else_) {
    if (cond->scalar<bool>()()) {
      OP_REQUIRES_OK(ctx, ctx->set_output("output", *then));
    } else {
      OP_REQUIRES_OK(ctx, ctx->set_output("output", *else_));
    }
  }
};


template <typename Device, typename T>
struct BatchSelectFunctorBase {
  void operator()(const Device& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims) {
    const Eigen::DenseIndex batch = cond_vec.size();
    const Eigen::DenseIndex all_but_batch = then_flat_outer_dims.dimension(1);

    Eigen::IndexList<Eigen::type2index<1>, Eigen::DenseIndex> broadcast_dims;
    broadcast_dims.set(1, all_but_batch);
    Eigen::IndexList<Eigen::DenseIndex, Eigen::type2index<1> > reshape_dims;
    reshape_dims.set(0, batch);

    Assign(d, output_flat_outer_dims,
           cond_vec.reshape(reshape_dims)
               .broadcast(broadcast_dims)
               .select(then_flat_outer_dims, else_flat_outer_dims));
  }
};

// A fast implementation on CPU, using loop to get rid of broadcasting.
template <typename T>
struct BatchSelectFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims) {
    const size_t batch = cond_vec.size();
    const size_t batch_size = then_flat_outer_dims.size() / batch;
    T* output = output_flat_outer_dims.data();
    const bool* c = cond_vec.data();
    const T* t = then_flat_outer_dims.data();
    const T* e = else_flat_outer_dims.data();

    auto work = [batch_size, output, c, t, e](int64_t start, int64_t end) {
      for (size_t i = start; i < end; ++i) {
        size_t offset = i * batch_size;
        port::prefetch<port::PREFETCH_HINT_NTA>(
            reinterpret_cast<const void*>(&t[offset + batch_size]));
        port::prefetch<port::PREFETCH_HINT_NTA>(
            reinterpret_cast<const void*>(&e[offset + batch_size]));
        port::prefetch<port::PREFETCH_HINT_NTA>(
            reinterpret_cast<const void*>(&c[i + 1]));
        if (c[i]) {
          for (size_t j = 0; j < batch_size; ++j) {
            output[offset + j] = t[offset + j];
          }
        } else {
          for (size_t j = 0; j < batch_size; ++j) {
            output[offset + j] = e[offset + j];
          }
        }
      }
    };
    auto cost = Eigen::TensorOpCost(sizeof(T) * batch_size * 2,  // ld bytes
                                    sizeof(T) * batch_size,      // st bytes
                                    batch_size);  // compute cycles
    d.parallelFor(batch, cost, work);
  }
};

template <typename Device, typename T, int NDIMS>
struct BCastSelectFunctorBase {
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS>::Tensor output_tensor,
                  typename TTypes<bool, NDIMS>::ConstTensor cond_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor then_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor else_tensor,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> cond_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> then_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> else_bcast) {
    output_tensor.device(d) = cond_tensor.broadcast(cond_bcast)
                                  .select(then_tensor.broadcast(then_bcast),
                                          else_tensor.broadcast(else_bcast));
  }
};

template <typename T, int NDIMS>
struct BCastSelectFunctor<CPUDevice, T, NDIMS>
    : BCastSelectFunctorBase<CPUDevice, T, NDIMS> {};


}  // namespace functor

}  // namespace tensorflow
