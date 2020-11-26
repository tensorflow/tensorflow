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
#include "tensorflow/c/experimental/op_handler/wrapper_operation.h"

namespace tensorflow {
WrapperOperation::WrapperOperation(AbstractOperation* parent_op,
                                   AbstractOperationKind kind)
    : AbstractOperation(kind), parent_op_(parent_op) {
  // TODO(b/172003047): Consider making AbstractOperation RefCounted.
  // parent_op_->Ref();
}
void WrapperOperation::Release() {
  parent_op_->Release();
  // TODO(b/172003047): Consider making AbstractOperation RefCounted.
  delete this;
}

Status WrapperOperation::Reset(const char* op, const char* raw_device_name) {
  return parent_op_->Reset(op, raw_device_name);
}
const string& WrapperOperation::Name() const { return parent_op_->Name(); }
const string& WrapperOperation::DeviceName() const {
  return parent_op_->DeviceName();
}
Status WrapperOperation::SetDeviceName(const char* name) {
  return parent_op_->SetDeviceName(name);
}
Status WrapperOperation::AddInput(AbstractTensorHandle* input) {
  return parent_op_->AddInput(input);
}
Status WrapperOperation::AddInputList(
    absl::Span<AbstractTensorHandle* const> inputs) {
  return parent_op_->AddInputList(inputs);
}
Status WrapperOperation::SetAttrString(const char* attr_name, const char* data,
                                       size_t length) {
  return parent_op_->SetAttrString(attr_name, data, length);
}
Status WrapperOperation::SetAttrInt(const char* attr_name, int64_t value) {
  return parent_op_->SetAttrInt(attr_name, value);
}
Status WrapperOperation::SetAttrFloat(const char* attr_name, float value) {
  return parent_op_->SetAttrFloat(attr_name, value);
}
Status WrapperOperation::SetAttrBool(const char* attr_name, bool value) {
  return parent_op_->SetAttrBool(attr_name, value);
}
Status WrapperOperation::SetAttrType(const char* attr_name, DataType value) {
  return parent_op_->SetAttrType(attr_name, value);
}
Status WrapperOperation::SetAttrShape(const char* attr_name,
                                      const int64_t* dims, const int num_dims) {
  return parent_op_->SetAttrShape(attr_name, dims, num_dims);
}
Status WrapperOperation::SetAttrFunction(const char* attr_name,
                                         const AbstractOperation* value) {
  return parent_op_->SetAttrFunction(attr_name, value);
}
Status WrapperOperation::SetAttrFunctionName(const char* attr_name,
                                             const char* value, size_t length) {
  return parent_op_->SetAttrFunctionName(attr_name, value, length);
}
Status WrapperOperation::SetAttrTensor(const char* attr_name,
                                       AbstractTensorInterface* tensor) {
  return parent_op_->SetAttrTensor(attr_name, tensor);
}
Status WrapperOperation::SetAttrStringList(const char* attr_name,
                                           const void* const* values,
                                           const size_t* lengths,
                                           int num_values) {
  return parent_op_->SetAttrStringList(attr_name, values, lengths, num_values);
}
Status WrapperOperation::SetAttrFloatList(const char* attr_name,
                                          const float* values, int num_values) {
  return parent_op_->SetAttrFloatList(attr_name, values, num_values);
}
Status WrapperOperation::SetAttrIntList(const char* attr_name,
                                        const int64_t* values, int num_values) {
  return parent_op_->SetAttrIntList(attr_name, values, num_values);
}
Status WrapperOperation::SetAttrTypeList(const char* attr_name,
                                         const DataType* values,
                                         int num_values) {
  return parent_op_->SetAttrTypeList(attr_name, values, num_values);
}
Status WrapperOperation::SetAttrBoolList(const char* attr_name,
                                         const unsigned char* values,
                                         int num_values) {
  return parent_op_->SetAttrBoolList(attr_name, values, num_values);
}
Status WrapperOperation::SetAttrShapeList(const char* attr_name,
                                          const int64_t** dims,
                                          const int* num_dims, int num_values) {
  return parent_op_->SetAttrShapeList(attr_name, dims, num_dims, num_values);
}
Status WrapperOperation::SetAttrFunctionList(
    const char* attr_name, absl::Span<const AbstractOperation*> values) {
  return parent_op_->SetAttrFunctionList(attr_name, values);
}
AbstractOperation* WrapperOperation::GetBackingOperation() {
  return parent_op_;
}
Status WrapperOperation::Execute(absl::Span<AbstractTensorHandle*> retvals,
                                 int* num_retvals) {
  return parent_op_->Execute(retvals, num_retvals);
}

}  // namespace tensorflow
