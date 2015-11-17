#include <string>
#include <unordered_set>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {
template <typename T>
class ListDiffOp : public OpKernel {
 public:
  explicit ListDiffOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt, DT_INT32}));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& y = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(x.shape()),
                errors::InvalidArgument("x should be a 1D vector."));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(y.shape()),
                errors::InvalidArgument("y should be a 1D vector."));

    std::unordered_set<T> y_set;
    const auto Ty = y.vec<T>();
    const int y_size = Ty.size();
    y_set.reserve(y_size);
    for (int i = 0; i < y_size; ++i) {
      y_set.insert(Ty(i));
    }

    // Compute the size of the output.
    const auto Tx = x.vec<T>();
    const int x_size = Tx.size();

    int out_size = 0;
    for (int i = 0; i < x_size; ++i) {
      if (y_set.count(Tx(i)) == 0) {
        ++out_size;
      }
    }

    // Allocate and populate outputs.
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {out_size}, &out));
    auto Tout = out->vec<T>();

    Tensor* indices = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {out_size}, &indices));
    auto Tindices = indices->vec<int32>();

    for (int i = 0, p = 0; i < x_size; ++i) {
      if (y_set.count(Tx(i)) == 0) {
        Tout(p) = Tx(i);
        Tindices(p) = i;
        p++;
      }
    }
  }
};

#define REGISTER_LISTDIFF(type)                                      \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ListDiff").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ListDiffOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_LISTDIFF);
REGISTER_LISTDIFF(string);
#undef REGISTER_LISTDIFF

}  // namespace tensorflow
