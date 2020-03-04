/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/operation_interface.h"

#include "absl/container/fixed_array.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/tensor_handle_interface.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

OperationInterface::OperationInterface(TFE_Context* ctx)
    : operation_(ctx->context) {}

const string& OperationInterface::DeviceName() const {
  absl::variant<Device*, CustomDevice*> variant_device =
      (operation_.Device() == kVariantDeviceNull)
          ? operation_.EagerContext().HostCPU()
          : operation_.Device();
  return absl::visit([](auto* d) -> const string& { return d->name(); },
                     variant_device);
}

Status OperationInterface::SetDeviceName(const char* name) {
  return operation_.SetDeviceName(name);
}

Status OperationInterface::SetAttrString(const char* attr_name,
                                         const char* data, size_t length) {
  operation_.MutableAttrs()->Set(attr_name, StringPiece(data, length));
  return Status::OK();
}

Status OperationInterface::SetAttrInt(const char* attr_name, int64_t value) {
  operation_.MutableAttrs()->Set(attr_name, static_cast<int64>(value));
  return Status::OK();
}

Status OperationInterface::SetAttrFloat(const char* attr_name, float value) {
  operation_.MutableAttrs()->Set(attr_name, value);
  return Status::OK();
}

Status OperationInterface::SetAttrBool(const char* attr_name, bool value) {
  operation_.MutableAttrs()->Set(attr_name, value);
  return Status::OK();
}

Status OperationInterface::SetAttrType(const char* attr_name,
                                       TF_DataType value) {
  operation_.MutableAttrs()->Set(attr_name, static_cast<DataType>(value));
  return Status::OK();
}

Status OperationInterface::SetAttrShape(const char* attr_name,
                                        const int64_t* dims,
                                        const int num_dims) {
  if (num_dims > TensorShape::MaxDimensions()) {
    return errors::InvalidArgument("Value specified for `", attr_name, "` has ",
                                   num_dims,
                                   " dimensions which is over the limit of ",
                                   TensorShape::MaxDimensions(), ".");
  }

  TensorShapeProto proto;
  if (num_dims < 0) {
    proto.set_unknown_rank(true);
  } else {
    for (int d = 0; d < num_dims; ++d) {
      proto.add_dim()->set_size(dims[d]);
    }
  }

  operation_.MutableAttrs()->Set(attr_name, proto);

  return Status::OK();
}

Status OperationInterface::SetAttrFunction(
    const char* attr_name,
    const std::unique_ptr<AbstractOperationInterface>& value) {
  AttrValue attr_value;
  NameAttrList* func = attr_value.mutable_func();
  func->set_name(value->Name());
  OperationInterface* value_operation =
      tensorflow::down_cast<OperationInterface*>(value.get());
  value_operation->operation_.Attrs().FillAttrValueMap(func->mutable_attr());
  operation_.MutableAttrs()->Set(attr_name, attr_value);
  return Status::OK();
}

Status OperationInterface::SetAttrFunctionName(const char* attr_name,
                                               const char* data,
                                               size_t length) {
  AttrValue attr_value;
  NameAttrList* func = attr_value.mutable_func();
  func->set_name(data, length);
  operation_.MutableAttrs()->Set(attr_name, attr_value);
  return Status::OK();
}

Status OperationInterface::SetAttrTensor(const char* attr_name,
                                         TF_Tensor* tensor) {
  Tensor t;
  TF_RETURN_IF_ERROR(TF_TensorToTensor(tensor, &t));
  operation_.MutableAttrs()->Set(attr_name, t);
  return Status::OK();
}

Status OperationInterface::SetAttrStringList(const char* attr_name,
                                             const void* const* values,
                                             const size_t* lengths,
                                             int num_values) {
  std::vector<StringPiece> v(num_values);
  for (int i = 0; i < num_values; ++i) {
    v[i] = StringPiece(static_cast<const char*>(values[i]), lengths[i]);
  }
  operation_.MutableAttrs()->Set(attr_name, v);

  return Status::OK();
}

Status OperationInterface::SetAttrFloatList(const char* attr_name,
                                            const float* values,
                                            int num_values) {
  operation_.MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<const float>(values, num_values));
  return Status::OK();
}

Status OperationInterface::SetAttrIntList(const char* attr_name,
                                          const int64_t* values,
                                          int num_values) {
  operation_.MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<const int64>(
                     reinterpret_cast<const int64*>(values), num_values));
  return Status::OK();
}

Status OperationInterface::SetAttrTypeList(const char* attr_name,
                                           const TF_DataType* values,
                                           int num_values) {
  operation_.MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<const DataType>(
                     reinterpret_cast<const DataType*>(values), num_values));
  return Status::OK();
}

Status OperationInterface::SetAttrBoolList(const char* attr_name,
                                           const unsigned char* values,
                                           int num_values) {
  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  operation_.MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<const bool>(b.get(), num_values));
  return Status::OK();
}

Status OperationInterface::SetAttrShapeList(const char* attr_name,
                                            const int64_t** dims,
                                            const int* num_dims,
                                            int num_values) {
  std::unique_ptr<TensorShapeProto[]> proto(new TensorShapeProto[num_values]);
  for (int i = 0; i < num_values; ++i) {
    const auto num_dims_i = num_dims[i];

    if (num_dims_i > TensorShape::MaxDimensions()) {
      return errors::InvalidArgument(
          strings::StrCat("Value specified for `", attr_name, "` has ",
                          num_dims_i, " dimensions which is over the limit of ",
                          TensorShape::MaxDimensions(), "."));
    }
    if (num_dims_i < 0) {
      proto[i].set_unknown_rank(true);
    } else {
      const int64_t* dims_i = dims[i];
      auto proto_i = &proto[i];
      for (int d = 0; d < num_dims_i; ++d) {
        proto_i->add_dim()->set_size(dims_i[d]);
      }
    }
  }
  operation_.MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<TensorShapeProto>(proto.get(), num_values));
  return Status::OK();
}

Status OperationInterface::SetAttrFunctionList(const char* attr_name,
                                               const TFE_Op** value,
                                               int num_values) {
  std::unique_ptr<NameAttrList[]> funcs(new NameAttrList[num_values]);
  for (int i = 0; i < num_values; i++) {
    auto value_operation =
        tensorflow::down_cast<OperationInterface*>(value[i]->operation.get());
    funcs[i].set_name(value_operation->operation_.Name());
    value_operation->operation_.Attrs().FillAttrValueMap(
        funcs[i].mutable_attr());
  }
  operation_.MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<const NameAttrList>(funcs.get(), num_values));
  return Status::OK();
}

const OpDef* OperationInterface::GetOpDef(Status* status) {
  const tensorflow::OpDef* op_def = operation_.OpDef();
  if (op_def) return op_def;
  *status = OpDefForOp(Name(), &op_def);
  return op_def;
}

Status OperationInterface::InputLength(const char* input_name, int* length) {
  Status status;
  const tensorflow::OpDef* op_def = GetOpDef(&status);
  if (!status.ok()) {
    return status;
  }
  AttrValueMap attrs;
  operation_.Attrs().FillAttrValueMap(&attrs);
  NameRangeMap name_ranges;
  TF_RETURN_IF_ERROR(
      NameRangesForNode(AttrSlice(&attrs), *op_def, &name_ranges, nullptr));
  auto iter = name_ranges.find(input_name);
  if (iter == name_ranges.end()) {
    return errors::InvalidArgument("Input '", input_name, "' not found");
  }
  *length = iter->second.second - iter->second.first;
  return Status::OK();
}

Status OperationInterface::OutputLength(const char* output_name, int* length) {
  Status status;
  const tensorflow::OpDef* op_def = GetOpDef(&status);
  if (!status.ok()) {
    return status;
  }
  AttrValueMap attrs;
  operation_.Attrs().FillAttrValueMap(&attrs);
  NameRangeMap name_ranges;
  TF_RETURN_IF_ERROR(
      NameRangesForNode(AttrSlice(&attrs), *op_def, nullptr, &name_ranges));
  auto iter = name_ranges.find(output_name);
  if (iter == name_ranges.end()) {
    return errors::InvalidArgument("Output '", output_name, "' not found");
  }
  *length = iter->second.second - iter->second.first;
  return Status::OK();
}

Status OperationInterface::AddInput(
    const std::unique_ptr<AbstractTensorHandleInterface>& input) {
  TensorHandle* h =
      tensorflow::down_cast<TensorHandleInterface*>(input.get())->Handle();
  operation_.AddInput(h);
  return operation_.MaybeInferSingleInputAttrs(h);
}

Status OperationInterface::AddInputList(
    const absl::FixedArray<std::unique_ptr<AbstractTensorHandleInterface>>&
        inputs) {
  for (auto& input : inputs) {
    TensorHandle* h =
        tensorflow::down_cast<TensorHandleInterface*>(input.get())->Handle();
    operation_.AddInput(h);
  }
  return operation_.InferInputListAttrs(inputs.size());
}

Status OperationInterface::Execute(
    absl::FixedArray<std::unique_ptr<AbstractTensorHandleInterface>>* retvals,
    int* num_retvals) {
  absl::FixedArray<tensorflow::TensorHandle*> handle_retvals(*num_retvals);
  TF_RETURN_IF_ERROR(
      EagerExecute(&operation_, handle_retvals.data(), num_retvals));
  for (int i = 0; i < *num_retvals; ++i) {
    retvals->at(i).reset(
        new tensorflow::TensorHandleInterface(handle_retvals[i]));
  }
  return Status::OK();
}

Status OperationInterface::SetCancellationManager(
    TFE_CancellationManager* cancellation_manager) {
  operation_.SetCancellationManager(
      &cancellation_manager->cancellation_manager);
  return Status::OK();
}

Status OperationInterface::SetUseXla(bool enable) {
  operation_.SetUseXla(enable);
  return Status::OK();
}

}  // namespace tensorflow
