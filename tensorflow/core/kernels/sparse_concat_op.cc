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
class SparseConcatOp : public OpKernel {
 public:
  explicit SparseConcatOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("concat_dim", &concat_dim_));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList inds;
    OP_REQUIRES_OK(context, context->input_list("indices", &inds));
    const int N = inds.size();
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsMatrix(inds[i].shape()),
                  errors::InvalidArgument(
                      "Input indices should be a matrix but received shape ",
                      inds[i].shape().DebugString(), " at position ", i));
    }

    OpInputList vals;
    OP_REQUIRES_OK(context, context->input_list("values", &vals));
    OP_REQUIRES(context, vals.size() == N,
                errors::InvalidArgument("Expected ", N, " input values, got ",
                                        vals.size()));
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsVector(vals[i].shape()),
                  errors::InvalidArgument(
                      "Input values should be a vector but received shape ",
                      vals[i].shape().DebugString(), " at position ", i));
    }

    OpInputList shapes;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes));
    OP_REQUIRES(context, shapes.size() == N,
                errors::InvalidArgument("Expected ", N, " input shapes, got ",
                                        shapes.size()));
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsVector(shapes[i].shape()),
                  errors::InvalidArgument(
                      "Input shapes should be a vector but received shape ",
                      shapes[i].shape().DebugString(), " at position ", i));
    }

    const TensorShape input_shape(shapes[0].vec<int64>());
    OP_REQUIRES(
        context, concat_dim_ >= 0 && concat_dim_ < input_shape.dims(),
        errors::InvalidArgument("Concat dimension must be between 0 and rank (",
                                input_shape.dims(), "), got ", concat_dim_));
    for (int i = 1; i < N; ++i) {
      const TensorShape current_shape(shapes[i].vec<int64>());
      OP_REQUIRES(context, current_shape.dims() == input_shape.dims(),
                  errors::InvalidArgument(
                      "Ranks of all input tensors must match: expected ",
                      input_shape.dims(), " but got ", current_shape.dims(),
                      " at position ", i));
      for (int j = 0; j < input_shape.dims(); ++j) {
        if (j != concat_dim_) {
          OP_REQUIRES(
              context, input_shape.dim_size(j) == current_shape.dim_size(j),
              errors::InvalidArgument(
                  "Input shapes must match: expected ", input_shape.dim_size(j),
                  " for dimension ", j, " but got ", current_shape.dim_size(j),
                  " at position ", i));
        }
      }
    }

    // The input and output sparse tensors are assumed to be ordered along
    // increasing dimension number. But in order for concat to work properly,
    // order[0] must be concat_dim. So we will reorder the inputs to the
    // concat ordering, concatenate, then reorder back to the standard order.
    // We make a deep copy of the input tensors to ensure that the in-place
    // reorder doesn't create race conditions for other ops that may be
    // concurrently reading the indices and values tensors.

    gtl::InlinedVector<int64, 8> std_order(input_shape.dims());
    std::iota(std_order.begin(), std_order.end(), 0);

    std::vector<int64> concat_order;
    concat_order.reserve(input_shape.dims());
    concat_order.push_back(concat_dim_);
    for (int j = 0; j < input_shape.dims(); ++j) {
      if (j != concat_dim_) {
        concat_order.push_back(j);
      }
    }

    std::vector<sparse::SparseTensor> sp_inputs;
    for (int i = 0; i < N; ++i) {
      const TensorShape current_shape(shapes[i].vec<int64>());
      sp_inputs.emplace_back(tensor::DeepCopy(inds[i]),
                             tensor::DeepCopy(vals[i]), current_shape,
                             std_order);
      sp_inputs[i].Reorder<T>(concat_order);
    }

    sparse::SparseTensor concat = sparse::SparseTensor::Concat<T>(sp_inputs);
    concat.Reorder<T>(std_order);

    context->set_output(0, concat.indices());
    context->set_output(1, concat.values());

    Tensor* output_shape_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                2, TensorShape({concat.shape().dims()}),
                                &output_shape_out));
    auto output_shape = output_shape_out->vec<int64>();
    for (int j = 0; j < concat.shape().dims(); ++j) {
      output_shape(j) = concat.shape().dim_size(j);
    }
  }

 private:
  int concat_dim_;
};

#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SparseConcat").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseConcatOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
}  // namespace tensorflow
