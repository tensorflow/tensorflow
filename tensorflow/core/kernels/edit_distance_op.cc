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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include <limits>

#include <vector>
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/edit_distance.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

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
  if (hypothesis_values.NumElements() != hypothesis_indices.dim_size(0))
    return errors::InvalidArgument(
        "Expected hypothesis_values.NumElements == "
        "#rows(hypothesis_indices), their shapes are: ",
        hypothesis_values.shape().DebugString(), " and ",
        hypothesis_indices.shape().DebugString());
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
  if (truth_values.NumElements() != truth_indices.dim_size(0))
    return errors::InvalidArgument(
        "Expected truth_values.NumElements == "
        "#rows(truth_indices), their shapes are: ",
        truth_values.shape().DebugString(), " and ",
        truth_indices.shape().DebugString());
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

  return OkStatus();
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

    TensorShape hypothesis_st_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeUtils::MakeShape(
                       hypothesis_shape->vec<int64_t>().data(),
                       hypothesis_shape->NumElements(), &hypothesis_st_shape));
    TensorShape truth_st_shape;
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                            truth_shape->vec<int64_t>().data(),
                            truth_shape->NumElements(), &truth_st_shape));

    // Assume indices are sorted in row-major order.
    std::vector<int64_t> sorted_order(truth_st_shape.dims());
    std::iota(sorted_order.begin(), sorted_order.end(), 0);

    sparse::SparseTensor hypothesis;
    OP_REQUIRES_OK(ctx, sparse::SparseTensor::Create(
                            *hypothesis_indices, *hypothesis_values,
                            hypothesis_st_shape, sorted_order, &hypothesis));

    sparse::SparseTensor truth;
    OP_REQUIRES_OK(ctx, sparse::SparseTensor::Create(
                            *truth_indices, *truth_values, truth_st_shape,
                            sorted_order, &truth));

    // Group dims 0, 1, ..., RANK - 1.  The very last dim is assumed
    // to store the variable length sequences.
    std::vector<int64_t> group_dims(truth_st_shape.dims() - 1);
    std::iota(group_dims.begin(), group_dims.end(), 0);

    TensorShape output_shape;
    for (int d = 0; d < static_cast<int>(group_dims.size()); ++d) {
      OP_REQUIRES_OK(ctx, output_shape.AddDimWithStatus(
                              std::max(hypothesis_st_shape.dim_size(d),
                                       truth_st_shape.dim_size(d))));
    }
    const auto output_elements = output_shape.num_elements();
    OP_REQUIRES(
        ctx, output_elements > 0,
        errors::InvalidArgument("Got output shape ", output_shape.DebugString(),
                                " which has 0 elements"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", output_shape, &output));
    auto output_t = output->flat<float>();
    output_t.setZero();

    std::vector<int64_t> output_strides(output_shape.dims());
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
      std::vector<int64_t> g_truth = truth_i.group();
      std::vector<int64_t> g_hypothesis = hypothesis_j.group();
      auto truth_seq = truth_i.values<T>();
      auto hypothesis_seq = hypothesis_j.values<T>();

      if (g_truth == g_hypothesis) {
        auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                      output_strides.begin(), int64_t{0});
        OP_REQUIRES(
            ctx, 0 <= loc && loc < output_elements,
            errors::Internal("Got an inner product ", loc,
                             " which would require writing to outside of "
                             "the buffer for the output tensor (max elements ",
                             output_elements, ")"));
        output_t(loc) =
            gtl::LevenshteinDistance<T>(truth_seq, hypothesis_seq, cmp);
        if (normalize_) output_t(loc) /= truth_seq.size();

        ++hypothesis_iter;
        ++truth_iter;
      } else if (g_truth > g_hypothesis) {  // zero-length truth
        auto loc = std::inner_product(g_hypothesis.begin(), g_hypothesis.end(),
                                      output_strides.begin(), int64_t{0});
        OP_REQUIRES(
            ctx, 0 <= loc && loc < output_elements,
            errors::Internal("Got an inner product ", loc,
                             " which would require writing to outside of "
                             "the buffer for the output tensor (max elements ",
                             output_elements, ")"));
        output_t(loc) = hypothesis_seq.size();
        if (normalize_ && output_t(loc) != 0.0f) {
          output_t(loc) = std::numeric_limits<float>::infinity();
        }
        ++hypothesis_iter;
      } else {  // zero-length hypothesis
        auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                      output_strides.begin(), int64_t{0});
        OP_REQUIRES(
            ctx, 0 <= loc && loc < output_elements,
            errors::Internal("Got an inner product ", loc,
                             " which would require writing to outside of "
                             "the buffer for the output tensor (max elements ",
                             output_elements, ")"));
        output_t(loc) = (normalize_) ? 1.0 : truth_seq.size();
        ++truth_iter;
      }
    }
    while (hypothesis_iter != hypothesis_grouper.end()) {  // zero-length truths
      sparse::Group hypothesis_j = *hypothesis_iter;
      std::vector<int64_t> g_hypothesis = hypothesis_j.group();
      auto hypothesis_seq = hypothesis_j.values<T>();
      auto loc = std::inner_product(g_hypothesis.begin(), g_hypothesis.end(),
                                    output_strides.begin(), int64_t{0});
      OP_REQUIRES(
          ctx, 0 <= loc && loc < output_elements,
          errors::Internal("Got an inner product ", loc,
                           " which would require writing to outside of the "
                           "buffer for the output tensor (max elements ",
                           output_elements, ")"));
      output_t(loc) = hypothesis_seq.size();
      if (normalize_ && output_t(loc) != 0.0f) {
        output_t(loc) = std::numeric_limits<float>::infinity();
      }
      ++hypothesis_iter;
    }
    while (truth_iter != truth_grouper.end()) {  // missing hypotheses
      sparse::Group truth_i = *truth_iter;
      std::vector<int64_t> g_truth = truth_i.group();
      auto truth_seq = truth_i.values<T>();
      auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                    output_strides.begin(), int64_t{0});
      OP_REQUIRES(
          ctx, 0 <= loc && loc < output_elements,
          errors::Internal("Got an inner product ", loc,
                           " which would require writing to outside of the "
                           "buffer for the output tensor (max elements ",
                           output_elements, ")"));
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

TF_CALL_POD_STRING_TYPES(REGISTER_CPU_KERNEL);

#undef REGISTER_CPU_KERNEL

}  // end namespace tensorflow
