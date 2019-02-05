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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/platform/prefetch.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace functor {
template <typename Device, typename T>
struct SelectScalarHandler;
}  // namespace functor

template <typename Device, typename T>
class SelectOp : public OpKernel {
 public:
  explicit SelectOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* cond;
    const Tensor* then;
    const Tensor* else_;
    OP_REQUIRES_OK(ctx, ctx->input("condition", &cond));
    OP_REQUIRES_OK(ctx, ctx->input("t", &then));
    OP_REQUIRES_OK(ctx, ctx->input("e", &else_));

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

#define REGISTER_SELECT(type)                                      \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Select").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SelectOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_SELECT);

#if GOOGLE_CUDA

// Registration of the GPU implementations.
#define REGISTER_SELECT_GPU(type)                                  \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Select").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SelectOp<GPUDevice, type>);

REGISTER_SELECT_GPU(bool);
REGISTER_SELECT_GPU(Eigen::half);
REGISTER_SELECT_GPU(float);
REGISTER_SELECT_GPU(double);
REGISTER_SELECT_GPU(int32);
REGISTER_SELECT_GPU(int64);
REGISTER_SELECT_GPU(complex64);
REGISTER_SELECT_GPU(complex128);

#undef REGISTER_SELECT_GPU

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
// Registration of the SYCL implementations.
#define REGISTER_SELECT_SYCL(type)                                  \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Select").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      SelectOp<SYCLDevice, type>);

REGISTER_SELECT_SYCL(float);
REGISTER_SELECT_SYCL(double);
REGISTER_SELECT_SYCL(int32);
REGISTER_SELECT_SYCL(int64);
#undef REGISTER_SELECT_SYCL
#endif  // TENSORFLOW_USE_SYCL

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
#ifdef TENSORFLOW_USE_SYCL
template <typename T>
struct SelectFunctor<SYCLDevice, T> : SelectFunctorBase<SYCLDevice, T> {};
#endif  // TENSORFLOW_USE_SYCL

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

// Specilization for CPU device. Forward input to output depending on the `cond`
// value.
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

#ifdef TENSORFLOW_USE_SYCL
template <typename Device, typename T>
struct SelectScalarFunctorBase {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  TTypes<bool>::ConstScalar cond,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    out.device(d) = cond() ? then_flat : else_flat;
  }
};

template <typename T>
struct SelectScalarFunctor<SYCLDevice, T>
    : SelectScalarFunctorBase<SYCLDevice, T> {};
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
struct BatchSelectFunctorBase {
  void operator()(const Device& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims) {
    const Eigen::DenseIndex batch = cond_vec.size();
    const Eigen::DenseIndex all_but_batch = then_flat_outer_dims.dimension(1);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<Eigen::DenseIndex, 2> broadcast_dims{{1, all_but_batch}};
    Eigen::Tensor<Eigen::DenseIndex, 2>::Dimensions reshape_dims{{batch, 1}};
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::DenseIndex> broadcast_dims;
    broadcast_dims.set(1, all_but_batch);
    Eigen::IndexList<Eigen::DenseIndex, Eigen::type2index<1> > reshape_dims;
    reshape_dims.set(0, batch);
#endif

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

    auto work = [batch_size, output, c, t, e](int64 start, int64 end) {
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

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
struct BatchSelectFunctor<SYCLDevice, T>
    : BatchSelectFunctorBase<SYCLDevice, T> {};
#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor

}  // namespace tensorflow
