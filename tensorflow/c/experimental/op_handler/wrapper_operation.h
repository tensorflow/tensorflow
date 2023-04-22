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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OP_HANDLER_WRAPPER_OPERATION_H_
#define TENSORFLOW_C_EXPERIMENTAL_OP_HANDLER_WRAPPER_OPERATION_H_

#include "tensorflow/c/eager/abstract_operation.h"

namespace tensorflow {

// Forwards all of the AbstractOperation's methods to its wrapped operation.
//
// Useful as a base class to default to forwarding while adding some
// customization.
class WrapperOperation : public AbstractOperation {
 public:
  explicit WrapperOperation(AbstractOperation*, AbstractOperationKind kind);
  void Release() override;
  Status Reset(const char* op, const char* raw_device_name) override;
  const string& Name() const override;
  const string& DeviceName() const override;
  Status SetDeviceName(const char* name) override;
  Status AddInput(AbstractTensorHandle* input) override;
  Status AddInputList(absl::Span<AbstractTensorHandle* const> inputs) override;
  Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                 int* num_retvals) override;
  Status SetAttrString(const char* attr_name, const char* data,
                       size_t length) override;
  Status SetAttrInt(const char* attr_name, int64_t value) override;
  Status SetAttrFloat(const char* attr_name, float value) override;
  Status SetAttrBool(const char* attr_name, bool value) override;
  Status SetAttrType(const char* attr_name, DataType value) override;
  Status SetAttrShape(const char* attr_name, const int64_t* dims,
                      const int num_dims) override;
  Status SetAttrFunction(const char* attr_name,
                         const AbstractOperation* value) override;
  Status SetAttrFunctionName(const char* attr_name, const char* value,
                             size_t length) override;
  Status SetAttrTensor(const char* attr_name,
                       AbstractTensorInterface* tensor) override;
  Status SetAttrStringList(const char* attr_name, const void* const* values,
                           const size_t* lengths, int num_values) override;
  Status SetAttrFloatList(const char* attr_name, const float* values,
                          int num_values) override;
  Status SetAttrIntList(const char* attr_name, const int64_t* values,
                        int num_values) override;
  Status SetAttrTypeList(const char* attr_name, const DataType* values,
                         int num_values) override;
  Status SetAttrBoolList(const char* attr_name, const unsigned char* values,
                         int num_values) override;
  Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                          const int* num_dims, int num_values) override;
  Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperation*> values) override;
  AbstractOperation* GetBackingOperation();

 private:
  AbstractOperation* parent_op_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_C_EXPERIMENTAL_OP_HANDLER_WRAPPER_OPERATION_H_
