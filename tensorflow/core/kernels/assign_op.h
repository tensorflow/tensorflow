#ifndef TENSORFLOW_KERNELS_ASSIGN_OP_H_
#define TENSORFLOW_KERNELS_ASSIGN_OP_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

// TODO(jeff): Get rid of use_exclusive_lock_ option

// Computes *input[0] = input[1]
class AssignOp : public OpKernel {
 public:
  explicit AssignOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_shape", &validate_shape_));
    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
  }

  void Compute(OpKernelContext* context) override {
    Tensor rhs = context->input(1);

    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    // If the left hand side is not initialized, or the shape of the
    // right-hand side is different than the left hand side, we need
    // to allocate a new tensor.
    {
      mutex_lock l(*context->input_ref_mutex(0));

      Tensor old_lhs = context->mutable_input(0, true);

      if (validate_shape_) {
        OP_REQUIRES(
            context, old_lhs.shape().IsSameSize(rhs.shape()),
            errors::InvalidArgument(
                "Assign requires shapes of both tensors to match. lhs shape= ",
                old_lhs.shape().ShortDebugString(), " rhs shape= ",
                rhs.shape().ShortDebugString()));
      }

      const bool same_shape = old_lhs.shape().IsSameSize(rhs.shape());
      if (!old_lhs.IsInitialized() || !same_shape) {
        // Create new tensor whose shape matches the right hand side
        // and copy then hand off to lhs.
        // We can't always know how this value will be used downstream,
        // so make conservative assumptions in specifying the memory
        // allocation attributes.
        AllocatorAttributes attr;
        attr.set_gpu_compatible(true);
        PersistentTensor copy;
        Tensor* copyTensor = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_persistent(old_lhs.dtype(), rhs.shape(),
                                                  &copy, &copyTensor, attr));
        Copy(context, copyTensor, rhs);
        context->replace_ref_input(0, *copyTensor, true);
        return;
      }

      // The tensor has already been initialized and the right hand side
      // matches the left hand side's shape.
      if (use_exclusive_lock_) {
        Copy(context, &old_lhs, rhs);
        return;
      }
    }

    // The tensor has already been initialized and the right hand side
    // matches the left hand side's shape. We have been told to do the
    // copy outside the lock.
    Tensor old_unlocked_lhs = context->mutable_input(0, false);
    Copy(context, &old_unlocked_lhs, rhs);
  }

  virtual void Copy(OpKernelContext* context, Tensor* lhs,
                    const Tensor& rhs) = 0;

  bool use_exclusive_lock_;
  bool validate_shape_;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_ASSIGN_OP_H_
