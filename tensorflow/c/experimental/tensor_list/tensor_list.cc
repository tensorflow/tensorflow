/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/tensor_list/tensor_list.h"

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/tensor_list.h"

void TF_GetTensorListFromTensor(const TF_Tensor* tensor, int index,
                                TF_TensorList** list, TF_Status* status) {
  ::tensorflow::Tensor cc_tensor;
  ::tensorflow::Status s = ::tensorflow::TF_TensorToTensor(tensor, &cc_tensor);
  TF_SetStatus(status, TF_OK, "");
  if (!s.ok()) {
    *list = nullptr;
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return;
  }

  if (index >= cc_tensor.NumElements()) {
    *list = nullptr;
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
    return;
  }

  ::tensorflow::TensorList* tensor_list = nullptr;
  if (cc_tensor.NumElements() == 1) {
    tensor_list = cc_tensor.scalar<::tensorflow::Variant>()()
                      .get<::tensorflow::TensorList>();
  } else {
    // there's maybe multiple variants in the Tensor, such as batch input
    auto values = cc_tensor.flat<::tensorflow::Variant>();
    tensor_list = values(index).get<::tensorflow::TensorList>();
  }

  // status has been set to TF_OK
  *list = reinterpret_cast<TF_TensorList*>(tensor_list);
}

void TF_SetTensorListToTensor(TF_TensorList* list, int index, TF_Tensor* tensor,
                              TF_Status* status) {
  ::tensorflow::Tensor cc_tensor;
  ::tensorflow::Status s = ::tensorflow::TF_TensorToTensor(tensor, &cc_tensor);
  TF_SetStatus(status, TF_OK, "");
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return;
  }

  if (index >= cc_tensor.NumElements()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
    return;
  }

  tensorflow::TensorList* tensor_list =
      reinterpret_cast<::tensorflow::TensorList*>(list);
  if (cc_tensor.NumElements() == 1) {
    cc_tensor.scalar<::tensorflow::Variant>()() = *tensor_list;
  } else {
    // flat the tensor to a vector despite the tensor's shape and there should
    // be no nullptr
    auto values = cc_tensor.flat<::tensorflow::Variant>();
    values(index) = *tensor_list;
  }
}

TF_DataType TF_TensorListGetDataType(const TF_TensorList* list) {
  auto cc_list = reinterpret_cast<const ::tensorflow::TensorList*>(list);
  return static_cast<TF_DataType>(cc_list->element_dtype);
}

int TF_TensorListNumDims(const TF_TensorList* list) {
  auto cc_list = reinterpret_cast<const ::tensorflow::TensorList*>(list);
  return cc_list->element_shape.dims();
}

int64_t TF_TensorListDim(const TF_TensorList* list, int dim_index) {
  auto cc_list = reinterpret_cast<const ::tensorflow::TensorList*>(list);
  return cc_list->element_shape.dim_size(dim_index);
}

void TF_TensorListCopy(const TF_TensorList* from, TF_TensorList* to) {
  auto cc_from = reinterpret_cast<const ::tensorflow::TensorList*>(from);
  auto cc_to = reinterpret_cast<::tensorflow::TensorList*>(to);
  // the `to` has been allocated from user and the deep copy will return a new
  // tensor list
  *cc_to = std::move(cc_from->Copy());
}

TF_TensorList* TF_NewTensorList() {
  auto cc_list = new ::tensorflow::TensorList;
  return reinterpret_cast<TF_TensorList*>(cc_list);
}

void TF_DeleteTensorList(TF_TensorList* list) {
  auto cc_list = reinterpret_cast<::tensorflow::TensorList*>(list);
  if (cc_list) {
    delete cc_list;
  }
}

int TF_TensorListSize(const TF_TensorList* list) {
  auto cc_list = reinterpret_cast<const ::tensorflow::TensorList*>(list);
  return cc_list->tensors().size();
}

void TF_TensorListSetDataType(TF_TensorList* list, TF_DataType dtype) {
  auto cc_list = reinterpret_cast<::tensorflow::TensorList*>(list);
  cc_list->element_dtype = static_cast<::tensorflow::DataType>(dtype);
}

void TF_TensorListSetShape(TF_TensorList* list, int num_dims, int64_t* dims) {
  auto cc_list = reinterpret_cast<::tensorflow::TensorList*>(list);

  if (dims) {
    std::vector<::tensorflow::int64> d;
    for (int i = 0; i < num_dims; i++) {
      d.push_back(dims[i]);
    }
    ::tensorflow::PartialTensorShape::MakePartialShape(
        d.data(), d.size(), &(cc_list->element_shape));
  } else {
    // if the dims is null, the shape will be an empty.
    cc_list->element_shape = ::tensorflow::PartialTensorShape();
  }
}

void TF_TensorListGetTensor(const TF_TensorList* list, int index,
                            TF_Tensor** tensor, TF_Status* status) {
  const tensorflow::TensorList* cc_list =
      reinterpret_cast<const ::tensorflow::TensorList*>(list);
  TF_SetStatus(status, TF_OK, "");
  if (index < cc_list->tensors().size()) {
    const tensorflow::Tensor& cc_tensor = cc_list->tensors()[index];
    tensorflow::Status s;
    TF_Tensor* tf_tensor = TF_TensorFromTensor(cc_tensor, &s);
    *tensor = tf_tensor;
    ::tensorflow::Set_TF_Status_from_Status(status, s);
  } else {
    // If out of bound, set the tensor to nullptr and return.
    *tensor = nullptr;
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
  }
}

void TF_TensorListSetTensor(TF_TensorList* list, int index, TF_Tensor* tensor,
                            TF_Status* status) {
  tensorflow::TensorList* tensor_list =
      reinterpret_cast<::tensorflow::TensorList*>(list);

  TF_SetStatus(status, TF_OK, "");
  if (list == nullptr || tensor == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "input argument is invalid");
  }

  // If out of bound, return directly
  if (index >= tensor_list->tensors().size()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
    return;
  }

  ::tensorflow::Tensor cc_tensor;
  ::tensorflow::Status s = ::tensorflow::TF_TensorToTensor(tensor, &cc_tensor);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return;
  }

  tensor_list->tensors()[index] = cc_tensor;
}

void TF_TensorListPop(TF_TensorList* list) {
  tensorflow::TensorList* tensor_list =
      reinterpret_cast<::tensorflow::TensorList*>(list);

  if (tensor_list->tensors().size() > 0) {
    tensor_list->tensors().pop_back();
  }

  // if the list has no tensor, it will do nothing
}

void TF_TensorListPush(TF_TensorList* list, TF_Tensor* tensor) {
  tensorflow::TensorList* tensor_list =
      reinterpret_cast<::tensorflow::TensorList*>(list);
  ::tensorflow::Tensor cc_tensor;
  ::tensorflow::Status s = ::tensorflow::TF_TensorToTensor(tensor, &cc_tensor);
  tensor_list->tensors().push_back(cc_tensor);
}

namespace tensorflow {
static Status ForwardInputOrCreateNewList(OpKernelContext* c, int32 input_index,
                                          int32 output_index,
                                          const TensorList& input_list,
                                          TensorList** output_list) {
  // Attempt to forward the input tensor to the output if possible.
  std::unique_ptr<Tensor> maybe_output = c->forward_input(
      input_index, output_index, DT_VARIANT, TensorShape{},
      c->input_memory_type(input_index), AllocatorAttributes());
  Tensor* output_tensor;
  if (maybe_output != nullptr && maybe_output->dtype() == DT_VARIANT &&
      maybe_output->NumElements() == 1) {
    output_tensor = maybe_output.get();
    TensorList* tmp_out = output_tensor->scalar<Variant>()().get<TensorList>();
    if (tmp_out == nullptr) {
      return errors::InvalidArgument(
          "Expected input ", input_index, " to be a TensorList but saw ",
          output_tensor->scalar<Variant>()().TypeName());
    }
    if (tmp_out->RefCountIsOne()) {
      // Woohoo, forwarding succeeded!
      c->set_output(output_index, *output_tensor);
      *output_list = tmp_out;
      return Status::OK();
    }
  }

  // If forwarding is not possible allocate a new output tensor and copy
  // the `input_list` to it.
  AllocatorAttributes attr;
  attr.set_on_host(true);
  TF_RETURN_IF_ERROR(
      c->allocate_output(output_index, {}, &output_tensor, attr));
  output_tensor->scalar<Variant>()() = input_list.Copy();

  *output_list = output_tensor->scalar<Variant>()().get<TensorList>();
  return Status::OK();
}
}  // namespace tensorflow

void TF_ForwardInputOrCreateNewList(TF_OpKernelContext* context,
                                    int input_index, int output_index,
                                    TF_TensorList* input_list,
                                    TF_TensorList** output_list,
                                    TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(context);
  ::tensorflow::TensorList* cc_input_list =
      reinterpret_cast<::tensorflow::TensorList*>(input_list);
  ::tensorflow::TensorList* cc_output_list;

  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Status s = ::tensorflow::ForwardInputOrCreateNewList(
      cc_ctx, input_index, output_index, *cc_input_list, &cc_output_list);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return;
  }

  *output_list = reinterpret_cast<TF_TensorList*>(cc_output_list);
}
