/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/ragged_tensor_variant.h"
#include "tensorflow/core/kernels/ragged_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/tensor_ops_util.h"

namespace tensorflow {
namespace {

template <typename VALUE_TYPE>
absl::Status UnbatchDenseZerothDim(
    const RaggedTensorVariant& batched_ragged,
    std::vector<RaggedTensorVariant>* ragged_components) {
  Tensor batched_values = batched_ragged.values();
  TensorShape values_shape = batched_values.shape();
  if (values_shape.dims() < 1) {
    return errors::InvalidArgument("Can't unbatch rank-0 tensor.");
  }
  auto num_components = values_shape.dim_size(0);
  values_shape.RemoveDim(0);
  auto num_values = values_shape.num_elements();

  ragged_components->resize(num_components);
  const auto& batched_flat = batched_values.flat<VALUE_TYPE>();

  for (auto i = decltype(num_components){}; i < num_components; i++) {
    (*ragged_components)[i].set_values(
        Tensor(DataTypeToEnum<VALUE_TYPE>::value, values_shape));
    auto ragged_component_values_flat =
        (*ragged_components)[i].mutable_values()->flat<VALUE_TYPE>();
    for (auto j = decltype(num_values){}; j < num_values; j++) {
      ragged_component_values_flat(j) = batched_flat(j + i * num_values);
    }
  }

  return absl::OkStatus();
}

template <typename VALUE_TYPE, typename SPLIT_TYPE>
absl::Status UnbatchRaggedZerothDim(
    const RaggedTensorVariant& batched_ragged,
    std::vector<RaggedTensorVariant>* ragged_components) {
  // Set up the component Ragged Tensors.
  int ragged_rank = batched_ragged.ragged_rank();
  if (ragged_rank == 0) {
    return UnbatchDenseZerothDim<VALUE_TYPE>(batched_ragged, ragged_components);
  }

  auto batched_splits_top_vec = batched_ragged.splits(0).vec<SPLIT_TYPE>();
  auto num_components = batched_splits_top_vec.size() - 1;

  if (num_components < 0) {
    return errors::Internal("Invalid split argument.");
  }

  int num_splits = ragged_rank - 1;
  ragged_components->resize(num_components);
  for (RaggedTensorVariant& ragged_component : *ragged_components) {
    ragged_component.mutable_nested_splits()->reserve(num_splits);
  }
  const auto& batched_flat = batched_ragged.values().flat<VALUE_TYPE>();
  auto num_inner_elems = batched_ragged.values().NumElements();
  if (batched_ragged.values().dim_size(0) > 1) {
    num_inner_elems /= batched_ragged.values().dim_size(0);
  }
  TensorShape values_shape = batched_ragged.values().shape();

  // Corner case: ragged_rank == 1, e.g. [[1, 2, 3], [4, 5]]
  if (num_splits == 0) {
    for (auto i = decltype(num_components){}; i < num_components; i++) {
      auto start = batched_splits_top_vec(i);
      auto limit = batched_splits_top_vec(i + 1);
      auto num_values = limit - start;
      values_shape.set_dim(0, num_values);
      (*ragged_components)[i].set_values(
          Tensor(DataTypeToEnum<VALUE_TYPE>::value, values_shape));
      auto ragged_component_values_flat =
          (*ragged_components)[i].mutable_values()->template flat<VALUE_TYPE>();
      for (auto j = decltype(num_values * num_inner_elems){};
           j < num_values * num_inner_elems; j++) {
        ragged_component_values_flat(j) =
            batched_flat(j + start * num_inner_elems);
      }
    }
    return absl::OkStatus();
  }

  // Unbatch nested splits.
  std::vector<typename TTypes<SPLIT_TYPE>::ConstVec> batched_splits_vec;
  batched_splits_vec.reserve(ragged_rank);
  for (int i = 0; i < ragged_rank; i++) {
    batched_splits_vec.push_back(batched_ragged.splits(i).vec<SPLIT_TYPE>());
  }
  std::vector<SPLIT_TYPE> index(num_splits, 1);
  std::vector<SPLIT_TYPE> ragged_component_values_size(num_components, 0);
  for (auto i = decltype(num_components){}; i < num_components; i++) {
    std::vector<typename TTypes<SPLIT_TYPE>::Vec> ragged_component_splits_vec;
    ragged_component_splits_vec.reserve(num_splits);
    SPLIT_TYPE split_size = -1;
    for (int j = 0; j < num_splits; j++) {
      if (j == 0) {
        split_size =
            batched_splits_top_vec(i + 1) - batched_splits_top_vec(i) + 1;
      } else {
        // Update split size based on previous split.
        SPLIT_TYPE last_index = ragged_component_splits_vec[j - 1].size() - 1;
        split_size = ragged_component_splits_vec[j - 1](last_index) + 1;
      }
      (*ragged_components)[i].append_splits(
          Tensor(DataTypeToEnum<SPLIT_TYPE>::value, TensorShape({split_size})));
      ragged_component_splits_vec.push_back((*ragged_components)[i]
                                                .mutable_splits(j)
                                                ->template vec<SPLIT_TYPE>());
      SPLIT_TYPE last_split_value = batched_splits_vec[j + 1](index[j] - 1);
      ragged_component_splits_vec[j](0) = 0;
      for (SPLIT_TYPE k = 1; k < split_size; k++, index[j]++) {
        ragged_component_splits_vec[j](k) =
            batched_splits_vec[j + 1](index[j]) - last_split_value;
      }
    }
    SPLIT_TYPE last_split_size =
        ragged_component_splits_vec[num_splits - 1].size();
    ragged_component_values_size[i] =
        ragged_component_splits_vec[num_splits - 1](last_split_size - 1);
  }

  // Unbatch values.
  int64_t value_index = 0;
  for (auto i = decltype(num_components){}; i < num_components; i++) {
    SPLIT_TYPE num_values = ragged_component_values_size[i];
    values_shape.set_dim(0, num_values);
    (*ragged_components)[i].set_values(
        Tensor(DataTypeToEnum<VALUE_TYPE>::value, values_shape));
    auto ragged_component_values_flat =
        (*ragged_components)[i].mutable_values()->template flat<VALUE_TYPE>();
    for (int64_t j = 0; j < num_values * num_inner_elems; j++, value_index++) {
      ragged_component_values_flat(j) = batched_flat(value_index);
    }
  }

  return absl::OkStatus();
}
}  // namespace

template <typename VALUE_TYPE, typename SPLIT_TYPE>
class RaggedTensorToVariantOp : public OpKernel {
 public:
  explicit RaggedTensorToVariantOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("batched_input", &batched_input_));
  }

  void Compute(OpKernelContext* context) override {
    // Read ragged_splits inputs.
    OpInputList ragged_nested_splits_in;
    OP_REQUIRES_OK(context, context->input_list("rt_nested_splits",
                                                &ragged_nested_splits_in));
    const int ragged_nested_splits_len = ragged_nested_splits_in.size();
    RaggedTensorVariant batched_ragged_input;
    // Read ragged_values input.
    batched_ragged_input.set_values(context->input(ragged_nested_splits_len));
    batched_ragged_input.mutable_nested_splits()->reserve(
        ragged_nested_splits_len);

    // Validate nested_row_splits.
    for (int i = ragged_nested_splits_len - 1; i >= 0; --i) {
      SPLIT_TYPE nvals;
      if (i == ragged_nested_splits_len - 1) {
        OP_REQUIRES(context, batched_ragged_input.values().dims() >= 1,
                    errors::InvalidArgument(
                        "Requires flat_values to have rank>=1 when "
                        "nested_row_splits is not empty, but is 0."));
        nvals = batched_ragged_input.values().dim_size(0);
      } else {
        nvals = ragged_nested_splits_in[i + 1].dim_size(0) - 1;
      }

      OP_REQUIRES_OK(context, RaggedTensorVerifySplits<SPLIT_TYPE>(
                                  ragged_nested_splits_in[i], true, nvals));
    }

    for (int i = 0; i < ragged_nested_splits_len; i++) {
      batched_ragged_input.append_splits(ragged_nested_splits_in[i]);
    }

    if (!batched_input_) {
      // Encode as a Scalar Variant Tensor.
      Tensor* encoded_scalar;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                       &encoded_scalar));
      encoded_scalar->scalar<Variant>()() = std::move(batched_ragged_input);
      return;
    }

    // Unbatch the Ragged Tensor and encode the components.
    std::vector<RaggedTensorVariant> unbatched_ragged_input;
    OP_REQUIRES_OK(context, UnbatchRaggedZerothDim<VALUE_TYPE, SPLIT_TYPE>(
                                batched_ragged_input, &unbatched_ragged_input));

    // Bundle the encoded scalar Variant Tensors into a rank-1 Variant Tensor.
    Tensor* encoded_vector;

    // output_size will be used for calling TensorShape(int64_t ...). We
    // cannot use `auto` type here, or there will be a narrowing error.
    int64_t output_size = unbatched_ragged_input.size();
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({output_size}),
                                            &encoded_vector));
    auto encoded_vector_t = encoded_vector->vec<Variant>();
    for (auto i = decltype(output_size){}; i < output_size; i++) {
      encoded_vector_t(i) = unbatched_ragged_input[i];
    }
  }

 private:
  bool batched_input_;
};

template <typename VALUE_TYPE, typename SPLIT_TYPE>
class RaggedTensorToVariantGradientOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    // Read inputs.
    Tensor encoded_variant = context->input(0);
    Tensor row_splits = context->input(1);
    auto flat_row_splits = row_splits.flat<SPLIT_TYPE>();
    TensorShape dense_values_shape;
    OP_REQUIRES_OK(context,
                   TensorShapeUtils::MakeShape(context->input(2).vec<int32>(),
                                               &dense_values_shape));

    // Validate row_splits.
    // Note rank of the row_splits can be 0. Besides, the number of ragged
    // values corresponding to the outermost splits are unknown when calculating
    // the gradient so we don't check the last element of `row_splits`
    if (row_splits.dims()) {
      OP_REQUIRES_OK(
          context, RaggedTensorVerifySplits<SPLIT_TYPE>(row_splits, false, 0));
    }

    const auto& flat_variants = encoded_variant.flat<Variant>();

    // Get a Tensor containing the flat_values for each variant.
    std::vector<Tensor> values;
    for (int i = 0; i < flat_variants.size(); ++i) {
      if (const auto* encoded = flat_variants(i).get<RaggedTensorVariant>()) {
        values.push_back(encoded->values());
      } else {
        // Missing value: this happens if only some of the variant values
        // generated by ragged_tensor_to_variant impacted the value that we're
        // calculating the gradient for.  In this case, we will see a
        // default-constructed variant; so treat it as a zero tensor with the
        // appropriate shape.
        const auto value_dtype = DataTypeToEnum<VALUE_TYPE>::v();
        auto piece_size = flat_row_splits(i + 1) - flat_row_splits(i);
        TensorShape zeros_shape = dense_values_shape;
        zeros_shape.set_dim(0, piece_size);
        Tensor zero(value_dtype, zeros_shape);
        zero.flat<VALUE_TYPE>().setZero();
        values.push_back(zero);
      }
    }

    if (values.size() == 1) {
      // Just one flat_value tensor: return as-is.
      context->set_output(0, values[0]);
    } else {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, dense_values_shape, &out));
      // ConcatCPU assumes non-empty output.
      if (dense_values_shape.num_elements() == 0) return;
      // Multiple flat_values tensors: concatenate them together.
      using Piece = typename TTypes<VALUE_TYPE, 2>::Matrix;
      using ConstPiece = typename TTypes<VALUE_TYPE, 2>::ConstMatrix;
      std::vector<std::unique_ptr<ConstPiece>> pieces;
      pieces.reserve(values.size());
      for (const Tensor& t : values) {
        // ConcatCPU assumes non-empty inputs.
        if (t.NumElements() == 0) continue;
        pieces.emplace_back(
            new ConstPiece(t.shaped<VALUE_TYPE, 2>({1, t.NumElements()})));
      }
      Piece out_flat =
          out->shaped<VALUE_TYPE, 2>({1, dense_values_shape.num_elements()});
      ConcatCPU<VALUE_TYPE>(context->device(), pieces, &out_flat);
    }
  }
};

#define REGISTER_KERNELS_WITH_SPLIT_TYPE(value_type, split_type)            \
  REGISTER_KERNEL_BUILDER(Name("RaggedTensorToVariant")                     \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<value_type>("Tvalues")        \
                              .TypeConstraint<split_type>("Tsplits"),       \
                          RaggedTensorToVariantOp<value_type, split_type>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("RaggedTensorToVariantGradient")                                 \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<value_type>("Tvalues")                            \
          .TypeConstraint<split_type>("Tsplits"),                           \
      RaggedTensorToVariantGradientOp<value_type, split_type>);

#define REGISTER_KERNELS(value_type)                  \
  REGISTER_KERNELS_WITH_SPLIT_TYPE(value_type, int32) \
  REGISTER_KERNELS_WITH_SPLIT_TYPE(value_type, int64_t)
TF_CALL_POD_TYPES(REGISTER_KERNELS);
TF_CALL_tstring(REGISTER_KERNELS);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
TF_CALL_quint16(REGISTER_KERNELS);
TF_CALL_qint16(REGISTER_KERNELS);
#undef REGISTER_KERNELS
#undef REGISTER_KERNELS_WITH_SPLIT_TYPE
}  // namespace tensorflow
