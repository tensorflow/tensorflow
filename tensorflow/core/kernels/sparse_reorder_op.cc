#define EIGEN_USE_THREADS

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

template <typename T>
class SparseReorderOp : public OpKernel {
 public:
  explicit SparseReorderOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_ind = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_ind.shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape",
                    input_ind.shape().DebugString()));

    const Tensor& input_val = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_val.shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape",
                    input_val.shape().DebugString()));

    const Tensor& input_shape_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape",
                    input_shape_in.shape().DebugString()));

    const TensorShape input_shape(input_shape_in.vec<int64>());

    gtl::InlinedVector<int64, 8> std_order(input_shape.dims());
    std::iota(std_order.begin(), std_order.end(), 0);

    // Check if the sparse tensor is already ordered correctly
    sparse::SparseTensor input_sp(input_ind, input_val, input_shape, std_order);

    if (input_sp.IndicesValid()) {
      context->set_output(0, input_sp.indices());
      context->set_output(1, input_sp.values());
    } else {
      // Deep-copy the input Tensors, then reorder in-place
      sparse::SparseTensor reordered_sp(tensor::DeepCopy(input_ind),
                                        tensor::DeepCopy(input_val),
                                        input_shape);
      reordered_sp.Reorder<T>(std_order);
      context->set_output(0, reordered_sp.indices());
      context->set_output(1, reordered_sp.values());
    }
  }
};

#define REGISTER_KERNELS(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseReorder").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseReorderOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
}  // namespace tensorflow
