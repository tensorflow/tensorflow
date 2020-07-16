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
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

struct RaggedTensor {
  Tensor values;
  std::vector<Tensor> nested_splits;
};

Status RaggedToVariant(const RaggedTensor& ragged, Tensor* encoded_list) {
  // Encode as a rank-1 Variant Tensor.
  int ragged_rank = ragged.nested_splits.size();
  *encoded_list = Tensor(DT_VARIANT, TensorShape({ragged_rank + 1}));
  auto encoded_vec = encoded_list->vec<Variant>();
  for (int i = 0; i < ragged_rank; i++) {
    encoded_vec(i) = ragged.nested_splits[i];
  }
  encoded_vec(ragged_rank) = ragged.values;
  return Status::OK();
}

template <typename VALUE_TYPE, typename SPLIT_TYPE>
Status UnbatchRaggedZerothDim(const RaggedTensor& batched_ragged,
                              std::vector<RaggedTensor>* ragged_components) {
  // Set up the component Ragged Tensors.
  int ragged_rank = batched_ragged.nested_splits.size();
  auto batched_splits_top_vec =
      batched_ragged.nested_splits[0].vec<SPLIT_TYPE>();
  int num_components = batched_splits_top_vec.size() - 1;
  int num_splits = ragged_rank - 1;
  ragged_components->resize(num_components);
  for (RaggedTensor ragged_component : *ragged_components) {
    ragged_component.nested_splits.reserve(num_splits);
  }
  const auto& batched_flat = batched_ragged.values.flat<VALUE_TYPE>();
  int num_inner_elems = batched_ragged.values.NumElements();
  if (batched_ragged.values.dim_size(0) > 1) {
    num_inner_elems /= batched_ragged.values.dim_size(0);
  }
  TensorShape values_shape = batched_ragged.values.shape();

  // Corner case: ragged_rank == 1, e.g. [[1, 2, 3], [4, 5]]
  if (num_splits == 0) {
    for (int i = 0; i < num_components; i++) {
      int start = batched_splits_top_vec(i);
      int limit = batched_splits_top_vec(i + 1);
      int num_values = limit - start;
      values_shape.set_dim(0, num_values);
      (*ragged_components)[i].values =
          Tensor(DataTypeToEnum<VALUE_TYPE>::value, values_shape);
      auto ragged_component_values_flat =
          (*ragged_components)[i].values.flat<VALUE_TYPE>();
      for (int j = 0; j < num_values * num_inner_elems; j++) {
        ragged_component_values_flat(j) =
            batched_flat(j + start * num_inner_elems);
      }
    }
    return Status::OK();
  }

  // Unbatch nested splits.
  std::vector<typename TTypes<SPLIT_TYPE>::ConstVec> batched_splits_vec;
  batched_splits_vec.reserve(ragged_rank);
  for (int i = 0; i < ragged_rank; i++) {
    batched_splits_vec.push_back(
        batched_ragged.nested_splits[i].vec<SPLIT_TYPE>());
  }
  std::vector<int> index(num_splits, 1);
  std::vector<int> ragged_component_values_size(num_components, 0);
  for (int i = 0; i < num_components; i++) {
    std::vector<typename TTypes<SPLIT_TYPE>::Vec> ragged_component_splits_vec;
    ragged_component_splits_vec.reserve(num_splits);
    int split_size = -1;
    for (int j = 0; j < num_splits; j++) {
      if (j == 0) {
        split_size =
            batched_splits_top_vec(i + 1) - batched_splits_top_vec(i) + 1;
      } else {
        // Update split size based on previous split.
        int last_index = ragged_component_splits_vec[j - 1].size() - 1;
        split_size = ragged_component_splits_vec[j - 1](last_index) + 1;
      }
      (*ragged_components)[i].nested_splits.push_back(
          Tensor(DataTypeToEnum<SPLIT_TYPE>::value, TensorShape({split_size})));
      ragged_component_splits_vec.push_back(
          (*ragged_components)[i].nested_splits[j].vec<SPLIT_TYPE>());
      SPLIT_TYPE last_split_value = batched_splits_vec[j + 1](index[j] - 1);
      ragged_component_splits_vec[j](0) = 0;
      for (int k = 1; k < split_size; k++, index[j]++) {
        ragged_component_splits_vec[j](k) =
            batched_splits_vec[j + 1](index[j]) - last_split_value;
      }
    }
    int last_split_size = ragged_component_splits_vec[num_splits - 1].size();
    ragged_component_values_size[i] =
        ragged_component_splits_vec[num_splits - 1](last_split_size - 1);
  }

  // Unbatch values.
  int value_index = 0;
  for (int i = 0; i < num_components; i++) {
    int num_values = ragged_component_values_size[i];
    values_shape.set_dim(0, num_values);
    (*ragged_components)[i].values =
        Tensor(DataTypeToEnum<VALUE_TYPE>::value, values_shape);
    auto ragged_component_values_flat =
        (*ragged_components)[i].values.flat<VALUE_TYPE>();
    for (int j = 0; j < num_values * num_inner_elems; j++, value_index++) {
      ragged_component_values_flat(j) = batched_flat(value_index);
    }
  }

  return Status::OK();
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
    RaggedTensor batched_ragged_input;
    // Read ragged_values input.
    batched_ragged_input.values = context->input(ragged_nested_splits_len);
    batched_ragged_input.nested_splits.reserve(ragged_nested_splits_len);
    for (int i = 0; i < ragged_nested_splits_len; i++) {
      batched_ragged_input.nested_splits.push_back(ragged_nested_splits_in[i]);
    }

    if (!batched_input_) {
      // Encode the input as is.
      Tensor encoded_list;
      OP_REQUIRES_OK(context,
                     RaggedToVariant(batched_ragged_input, &encoded_list));
      // Encode as a Scalar Variant Tensor.
      Tensor* encoded_scalar;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                       &encoded_scalar));
      encoded_scalar->scalar<Variant>()() = std::move(encoded_list);
      return;
    }

    // Unbatch the Ragged Tensor and encode the components.
    std::vector<RaggedTensor> ragged_components;
    OP_REQUIRES_OK(context, UnbatchRaggedZerothDim<VALUE_TYPE, SPLIT_TYPE>(
                                batched_ragged_input, &ragged_components));
    std::vector<Tensor> encoded_components(ragged_components.size());
    for (int i = 0; i < ragged_components.size(); i++) {
      OP_REQUIRES_OK(context, RaggedToVariant(ragged_components[i],
                                              &encoded_components[i]));
    }

    // Bundle the encoded scalar Variant Tensors into a rank-1 Variant Tensor.
    Tensor* encoded_ragged;
    int output_size = ragged_components.size();
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({output_size}),
                                            &encoded_ragged));
    auto encoded_ragged_vec = encoded_ragged->vec<Variant>();
    for (int i = 0; i < output_size; i++) {
      encoded_ragged_vec(i) = encoded_components[i];
    }
  }

 private:
  bool batched_input_;
};

#define REGISTER_KERNELS_WITH_SPLIT_TYPE(value_type, split_type)      \
  REGISTER_KERNEL_BUILDER(Name("RaggedTensorToVariant")               \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<value_type>("Tvalues")  \
                              .TypeConstraint<split_type>("Tsplits"), \
                          RaggedTensorToVariantOp<value_type, split_type>);
#define REGISTER_KERNELS(value_type)                  \
  REGISTER_KERNELS_WITH_SPLIT_TYPE(value_type, int32) \
  REGISTER_KERNELS_WITH_SPLIT_TYPE(value_type, int64)
TF_CALL_POD_TYPES(REGISTER_KERNELS);
TF_CALL_tstring(REGISTER_KERNELS);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
TF_CALL_quint16(REGISTER_KERNELS);
TF_CALL_qint16(REGISTER_KERNELS);
#undef REGISTER_KERNELS
#undef REGISTER_KERNELS_WITH_SPLIT_TYPE
}  // namespace tensorflow
