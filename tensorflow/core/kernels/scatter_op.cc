// See docs in ../ops/state_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

enum class UpdateOp { ASSIGN, ADD, SUB };

template <class T, typename Index, UpdateOp op>
class ScatterUpdateOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here.  Should we have the framework do some sort of
  //   integer promotion automatically, or should that be something
  //   that users have to do explicitly with a conversion operator
  //   in the graph?
  explicit ScatterUpdateOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* c) override {
    if (use_exclusive_lock_) {
      // Hold mutex while we apply updates
      mutex_lock l(*c->input_ref_mutex(0));
      DoCompute(c);
    } else {
      DoCompute(c);
    }
  }

 private:
  bool use_exclusive_lock_;

  // Check whether updates.shape = indices.shape + params.shape[1:]
  static bool ValidShapes(const Tensor& params, const Tensor& updates,
                          const Tensor& indices) {
    if (updates.dims() != indices.dims() + params.dims() - 1) return false;
    for (int d = 0; d < indices.dims(); d++) {
      if (updates.dim_size(d) != indices.dim_size(d)) {
        return false;
      }
    }
    for (int d = 1; d < params.dims(); d++) {
      if (params.dim_size(d) != updates.dim_size(d - 1 + indices.dims())) {
        return false;
      }
    }
    return true;
  }

  void DoCompute(OpKernelContext* c) {
    Tensor Tparams = c->mutable_input(0, use_exclusive_lock_);
    OP_REQUIRES(c, Tparams.IsInitialized(),
                errors::FailedPrecondition("Null ref for params"));
    const Tensor& Tindices = c->input(1);
    const Tensor& Tupdates = c->input(2);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(Tparams.shape()),
        errors::InvalidArgument("params must be at least 1-D, got shape ",
                                Tparams.shape().ShortDebugString()));
    OP_REQUIRES(
        c, ValidShapes(Tparams, Tupdates, Tindices),
        errors::InvalidArgument(
            "Must have updates.shape = indices.shape + params.shape[1:], got ",
            "updates.shape ", Tupdates.shape().ShortDebugString(),
            ", indices.shape ", Tindices.shape().ShortDebugString(),
            ", params.shape ", Tparams.shape().ShortDebugString()));
    const Index N = Tindices.NumElements();

    // We always return the input ref.
    c->forward_ref_input_to_ref_output(0, 0);

    if (N > 0) {
      const Index first_dim_size = Tparams.dim_size(0);
      // Validate all the indices are in range
      auto Tindices_vec = Tindices.flat<Index>();
      for (Index i = 0; i < N; i++) {
        const Index index = Tindices_vec(i);
        OP_REQUIRES(c, index >= 0 && index < first_dim_size,
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
      }
      auto Tparams_flat = Tparams.flat_outer_dims<T>();
      auto Tupdates_flat =
          Tupdates.shaped<T, 2>({N, Tupdates.NumElements() / N});
      for (Index i = 0; i < N; i++) {
        // Copy last Ndim-1 dimensions of Tupdates[i] to
        // Tparams[Tindices[i]]
        switch (op) {
          case UpdateOp::ASSIGN: {
            Tparams_flat.template chip<0>(Tindices_vec(i)) =
                Tupdates_flat.template chip<0>(i);
            break;
          }
          case UpdateOp::ADD: {
            Tparams_flat.template chip<0>(Tindices_vec(i)) +=
                Tupdates_flat.template chip<0>(i);
            break;
          }
          case UpdateOp::SUB: {
            Tparams_flat.template chip<0>(Tindices_vec(i)) -=
                Tupdates_flat.template chip<0>(i);
            break;
          }
        }
      }
    }
  }
};

#define REGISTER_SCATTER_UPDATE(type, index_type)  \
  REGISTER_KERNEL_BUILDER(                         \
      Name("ScatterUpdate")                        \
          .Device(DEVICE_CPU)                      \
          .TypeConstraint<type>("T")               \
          .TypeConstraint<index_type>("Tindices"), \
      ScatterUpdateOp<type, index_type, UpdateOp::ASSIGN>);

#define REGISTER_SCATTER_UPDATE_INT32(type) REGISTER_SCATTER_UPDATE(type, int32)
#define REGISTER_SCATTER_UPDATE_INT64(type) REGISTER_SCATTER_UPDATE(type, int64)

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_UPDATE_INT32);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_UPDATE_INT64);

#undef REGISTER_SCATTER_UPDATE_INT64
#undef REGISTER_SCATTER_UPDATE_INT32
#undef REGISTER_SCATTER_UPDATE

#define REGISTER_SCATTER_ADD(type, index_type)                         \
  REGISTER_KERNEL_BUILDER(Name("ScatterAdd")                           \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          ScatterUpdateOp<type, index_type, UpdateOp::ADD>);

#define REGISTER_SCATTER_ADD_INT32(type) REGISTER_SCATTER_ADD(type, int32)
#define REGISTER_SCATTER_ADD_INT64(type) REGISTER_SCATTER_ADD(type, int64)

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ADD_INT32);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ADD_INT64);

#undef REGISTER_SCATTER_ADD_INT32
#undef REGISTER_SCATTER_ADD_INT64
#undef REGISTER_SCATTER_ADD

#define REGISTER_SCATTER_SUB(type, index_type)                         \
  REGISTER_KERNEL_BUILDER(Name("ScatterSub")                           \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          ScatterUpdateOp<type, index_type, UpdateOp::SUB>);

#define REGISTER_SCATTER_SUB_INT32(type) REGISTER_SCATTER_SUB(type, int32)
#define REGISTER_SCATTER_SUB_INT64(type) REGISTER_SCATTER_SUB(type, int64)

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_SUB_INT32);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_SUB_INT64);

#undef REGISTER_SCATTER_SUB_INT64
#undef REGISTER_SCATTER_SUB_INT32
#undef REGISTER_SCATTER_SUB

}  // namespace tensorflow
