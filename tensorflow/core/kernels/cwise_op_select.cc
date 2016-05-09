/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

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
        ctx, FastBoundsCheck(cond->NumElements(),
                             std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("cond vector larger than ",
                                std::numeric_limits<Eigen::DenseIndex>::max()));
    OP_REQUIRES(
        ctx, FastBoundsCheck(then->flat_outer_dims<T>().dimension(1),
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
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, then->shape(), &output));
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
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, then->shape(), &output));
    if (output->NumElements() > 0) {
      functor::SelectFunctor<Device, T> func;
      func(ctx->eigen_device<Device>(), output->flat<T>(), cond->flat<bool>(),
           then->flat<T>(), else_->flat<T>());
    }
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

REGISTER_SELECT_GPU(Eigen::half);
REGISTER_SELECT_GPU(float);
REGISTER_SELECT_GPU(double);
REGISTER_SELECT_GPU(int32);
REGISTER_SELECT_GPU(int64);
REGISTER_SELECT_GPU(complex64);

#undef REGISTER_SELECT_GPU

#endif  // GOOGLE_CUDA

namespace functor {

// CPU Specializations of Select functors.
template <typename T>
struct SelectFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    Assign(d, out, cond_flat.select(then_flat, else_flat));
  }
};

template <typename T>
struct BatchSelectFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims) {
    const Eigen::DenseIndex batch = cond_vec.size();
    const Eigen::DenseIndex all_but_batch = then_flat_outer_dims.dimension(1);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<Eigen::DenseIndex, 2> broadcast_dims{{ 1, all_but_batch }};
    Eigen::Tensor<Eigen::DenseIndex, 2>::Dimensions reshape_dims{{ batch, 1 }};
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

}  // namespace functor

}  // namespace tensorflow
