// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include <limits>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/gtl/edit_distance.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

Status ValidateShapes(OpKernelContext* ctx, const Tensor& hypothesis_indices,
                      const Tensor& hypothesis_values,
                      const Tensor& hypothesis_shape,
                      const Tensor& truth_indices, const Tensor& truth_values,
                      const Tensor& truth_shape) {
  if (!TensorShapeUtils::IsMatrix(hypothesis_indices.shape()))
    return errors::InvalidArgument(
        "hypothesis_indices should be a matrix, but got shape: ",
        hypothesis_indices.shape().DebugString());
  if (!TensorShapeUtils::IsMatrix(truth_indices.shape()))
    return errors::InvalidArgument(
        "truth_indices should be a matrix, but got shape: ",
        truth_indices.shape().DebugString());
  if (!TensorShapeUtils::IsVector(hypothesis_values.shape()))
    return errors::InvalidArgument(
        "hypothesis_values should be a vector, but got shape: ",
        hypothesis_values.shape().DebugString());
  if (!TensorShapeUtils::IsVector(truth_values.shape()))
    return errors::InvalidArgument(
        "truth_values should be a vector, but got shape: ",
        truth_values.shape().DebugString());
  if (!TensorShapeUtils::IsVector(hypothesis_shape.shape()))
    return errors::InvalidArgument(
        "hypothesis_shape should be a vector, but got shape: ",
        hypothesis_shape.shape().DebugString());
  if (!TensorShapeUtils::IsVector(truth_shape.shape()))
    return errors::InvalidArgument(
        "truth_shape should be a vector, but got shape: ",
        truth_shape.shape().DebugString());
  if (hypothesis_shape.NumElements() != hypothesis_indices.dim_size(1))
    return errors::InvalidArgument(
        "Expected hypothesis_shape.NumElements == "
        "#cols(hypothesis_indices), their shapes are: ",
        hypothesis_shape.shape().DebugString(), " and ",
        hypothesis_indices.shape().DebugString());
  if (truth_shape.NumElements() < 2)
    return errors::InvalidArgument(
        "Input SparseTensors must have rank at least 2, but truth_shape "
        "rank is: ",
        truth_shape.NumElements());
  if (truth_shape.NumElements() != truth_indices.dim_size(1))
    return errors::InvalidArgument(
        "Expected truth_shape.NumElements == "
        "#cols(truth_indices), their shapes are: ",
        truth_shape.shape().DebugString(), " and ",
        truth_indices.shape().DebugString());
  if (truth_shape.NumElements() != hypothesis_shape.NumElements())
    return errors::InvalidArgument(
        "Expected truth and hypothesis to have matching ranks, but "
        "their shapes are: ",
        truth_shape.shape().DebugString(), " and ",
        hypothesis_shape.shape().DebugString());

  return Status::OK();
}

}  // namespace

template <typename T>
class EditDistanceOp : public OpKernel {
 public:
  explicit EditDistanceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("normalize", &normalize_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* hypothesis_indices;
    const Tensor* hypothesis_values;
    const Tensor* hypothesis_shape;
    const Tensor* truth_indices;
    const Tensor* truth_values;
    const Tensor* truth_shape;
    OP_REQUIRES_OK(ctx, ctx->input("hypothesis_indices", &hypothesis_indices));
    OP_REQUIRES_OK(ctx, ctx->input("hypothesis_values", &hypothesis_values));
    OP_REQUIRES_OK(ctx, ctx->input("hypothesis_shape", &hypothesis_shape));
    OP_REQUIRES_OK(ctx, ctx->input("truth_indices", &truth_indices));
    OP_REQUIRES_OK(ctx, ctx->input("truth_values", &truth_values));
    OP_REQUIRES_OK(ctx, ctx->input("truth_shape", &truth_shape));

    OP_REQUIRES_OK(
        ctx, ValidateShapes(ctx, *hypothesis_indices, *hypothesis_values,
                            *hypothesis_shape, *truth_indices, *truth_values,
                            *truth_shape));

    TensorShape hypothesis_st_shape = TensorShapeUtils::MakeShape(
        hypothesis_shape->vec<int64>().data(), hypothesis_shape->NumElements());
    TensorShape truth_st_shape = TensorShapeUtils::MakeShape(
        truth_shape->vec<int64>().data(), truth_shape->NumElements());

    // Assume indices are sorted in row-major order.
    std::vector<int64> sorted_order(truth_st_shape.dims());
    std::iota(sorted_order.begin(), sorted_order.end(), 0);

    sparse::SparseTensor hypothesis(*hypothesis_indices, *hypothesis_values,
                                    hypothesis_st_shape, sorted_order);
    sparse::SparseTensor truth(*truth_indices, *truth_values, truth_st_shape,
                               sorted_order);

    // Group dims 0, 1, ..., RANK - 1.  The very last dim is assumed
    // to store the variable length sequences.
    std::vector<int64> group_dims(truth_st_shape.dims() - 1);
    std::iota(group_dims.begin(), group_dims.end(), 0);

    TensorShape output_shape;
    for (int d = 0; d < group_dims.size(); ++d) {
      output_shape.AddDim(std::max(hypothesis_st_shape.dim_size(d),
                                   truth_st_shape.dim_size(d)));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", output_shape, &output));
    auto output_t = output->flat<float>();
    output_t.setZero();

    std::vector<int64> output_strides(output_shape.dims());
    output_strides[output_shape.dims() - 1] = 1;
    for (int d = output_shape.dims() - 2; d >= 0; --d) {
      output_strides[d] = output_strides[d + 1] * output_shape.dim_size(d + 1);
    }

    auto hypothesis_grouper = hypothesis.group(group_dims);
    auto truth_grouper = truth.group(group_dims);

    auto hypothesis_iter = hypothesis_grouper.begin();
    auto truth_iter = truth_grouper.begin();

    auto cmp = std::equal_to<T>();

    while (hypothesis_iter != hypothesis_grouper.end() &&
           truth_iter != truth_grouper.end()) {
      sparse::Group truth_i = *truth_iter;
      sparse::Group hypothesis_j = *hypothesis_iter;
      std::vector<int64> g_truth = truth_i.group();
      std::vector<int64> g_hypothesis = hypothesis_j.group();
      auto truth_seq = truth_i.values<T>();
      auto hypothesis_seq = hypothesis_j.values<T>();

      if (g_truth == g_hypothesis) {
        auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                      output_strides.begin(), 0);
        output_t(loc) =
            gtl::LevenshteinDistance<T>(truth_seq, hypothesis_seq, cmp);
        if (normalize_) output_t(loc) /= truth_seq.size();

        ++hypothesis_iter;
        ++truth_iter;
      } else if (g_truth > g_hypothesis) {  // missing truth @ this hypothesis
        auto loc = std::inner_product(g_hypothesis.begin(), g_hypothesis.end(),
                                      output_strides.begin(), 0);
        output_t(loc) = hypothesis_seq.size();
        if (normalize_) output_t(loc) /= 0.0;
        ++hypothesis_iter;
      } else {  // missing hypothesis @ this truth
        auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                      output_strides.begin(), 0);
        output_t(loc) = (normalize_) ? 1.0 : truth_seq.size();
        ++truth_iter;
      }
    }
    while (hypothesis_iter != hypothesis_grouper.end()) {  // missing truths
      sparse::Group hypothesis_j = *hypothesis_iter;
      std::vector<int64> g_hypothesis = hypothesis_j.group();
      auto hypothesis_seq = hypothesis_j.values<T>();
      auto loc = std::inner_product(g_hypothesis.begin(), g_hypothesis.end(),
                                    output_strides.begin(), 0);
      output_t(loc) = hypothesis_seq.size();
      if (normalize_) output_t(loc) /= 0.0;
      ++hypothesis_iter;
    }
    while (truth_iter != truth_grouper.end()) {  // missing hypotheses
      sparse::Group truth_i = *truth_iter;
      std::vector<int64> g_truth = truth_i.group();
      auto truth_seq = truth_i.values<T>();
      auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                    output_strides.begin(), 0);
      output_t(loc) = (normalize_) ? 1.0 : truth_seq.size();
      ++truth_iter;
    }
  }

 private:
  bool normalize_;

  TF_DISALLOW_COPY_AND_ASSIGN(EditDistanceOp);
};

#define REGISTER_CPU_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("EditDistance").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      EditDistanceOp<T>);

TF_CALL_ALL_TYPES(REGISTER_CPU_KERNEL);

#undef REGISTER_CPU_KERNEL

}  // end namespace tensorflow
