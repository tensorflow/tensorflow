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
#ifndef TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_TAPE_TAPE_OPERATION_H_
#define TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_TAPE_TAPE_OPERATION_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gradients {
class TapeOperation : public AbstractOperation {
 public:
  explicit TapeOperation(AbstractOperation*, Tape*, const GradientRegistry&);
  void Release() override;
  absl::Status Reset(const char* op, const char* raw_device_name) override;
  const string& Name() const override;
  const string& DeviceName() const override;
  absl::Status SetDeviceName(const char* name) override;
  absl::Status AddInput(AbstractTensorHandle* input) override;
  absl::Status AddInputList(
      absl::Span<AbstractTensorHandle* const> inputs) override;
  absl::Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                       int* num_retvals) override;
  absl::Status SetAttrString(const char* attr_name, const char* data,
                             size_t length) override;
  absl::Status SetAttrInt(const char* attr_name, int64_t value) override;
  absl::Status SetAttrFloat(const char* attr_name, float value) override;
  absl::Status SetAttrBool(const char* attr_name, bool value) override;
  absl::Status SetAttrType(const char* attr_name, DataType value) override;
  absl::Status SetAttrShape(const char* attr_name, const int64_t* dims,
                            const int num_dims) override;
  absl::Status SetAttrFunction(const char* attr_name,
                               const AbstractOperation* value) override;
  absl::Status SetAttrFunctionName(const char* attr_name, const char* value,
                                   size_t length) override;
  absl::Status SetAttrTensor(const char* attr_name,
                             AbstractTensorInterface* tensor) override;
  absl::Status SetAttrStringList(const char* attr_name,
                                 const void* const* values,
                                 const size_t* lengths,
                                 int num_values) override;
  absl::Status SetAttrFloatList(const char* attr_name, const float* values,
                                int num_values) override;
  absl::Status SetAttrIntList(const char* attr_name, const int64_t* values,
                              int num_values) override;
  absl::Status SetAttrTypeList(const char* attr_name, const DataType* values,
                               int num_values) override;
  absl::Status SetAttrBoolList(const char* attr_name,
                               const unsigned char* values,
                               int num_values) override;
  absl::Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                                const int* num_dims, int num_values) override;
  absl::Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperation*> values) override;
  AbstractOperation* GetBackingOperation();
  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
    return ptr->getKind() == kTape;
  }
  ~TapeOperation() override;

 private:
  AbstractOperation* parent_op_;
  ForwardOperation forward_op_;
  Tape* tape_;
  const GradientRegistry& registry_;
};

}  // namespace gradients
}  // namespace tensorflow
#endif  // TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_TAPE_TAPE_OPERATION_H_
