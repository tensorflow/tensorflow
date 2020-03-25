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
#ifndef TENSORFLOW_C_EAGER_OPERATION_INTERFACE_H_
#define TENSORFLOW_C_EAGER_OPERATION_INTERFACE_H_

#include <memory>

#include "absl/container/fixed_array.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/tensor_handle_interface.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"

// Abstract interface to an operation.
class AbstractOperationInterface {
 public:
  virtual ~AbstractOperationInterface() {}

  virtual void Clear() = 0;
  virtual tensorflow::Status Reset(const char* op,
                                   const char* raw_device_name) = 0;

  virtual const tensorflow::string& Name() const = 0;
  virtual const tensorflow::string& DeviceName() const = 0;
  virtual tensorflow::Status SetDeviceName(const char* name) = 0;

  virtual tensorflow::Status AddInput(
      const std::unique_ptr<AbstractTensorHandleInterface>& input) = 0;
  virtual tensorflow::Status AddInputList(
      const absl::FixedArray<std::unique_ptr<AbstractTensorHandleInterface>>&
          inputs) = 0;
  virtual tensorflow::Status Execute(
      absl::FixedArray<std::unique_ptr<AbstractTensorHandleInterface>>* retvals,
      int* num_retvals) = 0;
  virtual const tensorflow::OpDef* OpDef() const = 0;

  virtual tensorflow::Status SetAttrString(const char* attr_name,
                                           const char* data, size_t length) = 0;
  virtual tensorflow::Status SetAttrInt(const char* attr_name,
                                        int64_t value) = 0;
  virtual tensorflow::Status SetAttrFloat(const char* attr_name,
                                          float value) = 0;
  virtual tensorflow::Status SetAttrBool(const char* attr_name, bool value) = 0;
  virtual tensorflow::Status SetAttrType(const char* attr_name,
                                         TF_DataType value) = 0;
  virtual tensorflow::Status SetAttrShape(const char* attr_name,
                                          const int64_t* dims,
                                          const int num_dims) = 0;
  virtual tensorflow::Status SetAttrFunction(
      const char* attr_name,
      const std::unique_ptr<AbstractOperationInterface>& value) = 0;
  virtual tensorflow::Status SetAttrFunctionName(const char* attr_name,
                                                 const char* value,
                                                 size_t length) = 0;
  virtual tensorflow::Status SetAttrTensor(const char* attr_name,
                                           TF_Tensor* tensor) = 0;
  virtual tensorflow::Status SetAttrStringList(const char* attr_name,
                                               const void* const* values,
                                               const size_t* lengths,
                                               int num_values) = 0;
  virtual tensorflow::Status SetAttrFloatList(const char* attr_name,
                                              const float* values,
                                              int num_values) = 0;
  virtual tensorflow::Status SetAttrIntList(const char* attr_name,
                                            const int64_t* values,
                                            int num_values) = 0;
  virtual tensorflow::Status SetAttrTypeList(const char* attr_name,
                                             const TF_DataType* values,
                                             int num_values) = 0;
  virtual tensorflow::Status SetAttrBoolList(const char* attr_name,
                                             const unsigned char* values,
                                             int num_values) = 0;
  virtual tensorflow::Status SetAttrShapeList(const char* attr_name,
                                              const int64_t** dims,
                                              const int* num_dims,
                                              int num_values) = 0;
  virtual tensorflow::Status SetAttrFunctionList(const char* attr_name,
                                                 const TFE_Op** value,
                                                 int num_values) = 0;

  virtual tensorflow::Status InputLength(const char* input_name,
                                         int* length) = 0;
  virtual tensorflow::Status OutputLength(const char* output_name,
                                          int* length) = 0;

  // Experimental
  virtual tensorflow::Status SetUseXla(bool enable) {
    return tensorflow::errors::Unimplemented("SetUseXla not implemented");
  }
  virtual tensorflow::Status SetCancellationManager(
      TFE_CancellationManager* cancellation_manager) {
    return tensorflow::errors::Unimplemented(
        "SetCancellationManager not implemented");
  }
};

namespace tensorflow {

class OpDef;

class OperationInterface : public AbstractOperationInterface {
 public:
  explicit OperationInterface(TFE_Context* ctx);
  ~OperationInterface() override{};

  void Clear() override { operation_.Clear(); }
  Status Reset(const char* op, const char* raw_device_name) override {
    return operation_.Reset(op, raw_device_name, false, nullptr);
  }

  const string& Name() const override { return operation_.Name(); }
  const string& DeviceName() const override;
  Status SetDeviceName(const char* name) override;

  Status AddInput(
      const std::unique_ptr<AbstractTensorHandleInterface>& input) override;
  Status AddInputList(
      const absl::FixedArray<std::unique_ptr<AbstractTensorHandleInterface>>&
          inputs) override;
  Status Execute(
      absl::FixedArray<std::unique_ptr<AbstractTensorHandleInterface>>* retvals,
      int* num_retvals) override;
  const tensorflow::OpDef* OpDef() const override {
    return operation_.OpDef();
  };

  Status SetAttrString(const char* attr_name, const char* data,
                       size_t length) override;
  Status SetAttrInt(const char* attr_name, int64_t value) override;
  Status SetAttrFloat(const char* attr_name, float value) override;
  Status SetAttrBool(const char* attr_name, bool value) override;
  Status SetAttrType(const char* attr_name, TF_DataType value) override;
  Status SetAttrShape(const char* attr_name, const int64_t* dims,
                      const int num_dims) override;
  Status SetAttrFunction(
      const char* attr_name,
      const std::unique_ptr<AbstractOperationInterface>& value) override;
  Status SetAttrFunctionName(const char* attr_name, const char* data,
                             size_t length) override;
  Status SetAttrTensor(const char* attr_name, TF_Tensor* tensor) override;
  Status SetAttrStringList(const char* attr_name, const void* const* values,
                           const size_t* lengths, int num_values) override;
  Status SetAttrFloatList(const char* attr_name, const float* values,
                          int num_values) override;
  Status SetAttrIntList(const char* attr_name, const int64_t* values,
                        int num_values) override;
  Status SetAttrTypeList(const char* attr_name, const TF_DataType* values,
                         int num_values) override;
  Status SetAttrBoolList(const char* attr_name, const unsigned char* values,
                         int num_values) override;
  Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                          const int* num_dims, int num_values) override;
  Status SetAttrFunctionList(const char* attr_name, const TFE_Op** value,
                             int num_values) override;

  Status InputLength(const char* input_name, int* length) override;
  Status OutputLength(const char* output_name, int* length) override;

  Status SetUseXla(bool enable) override;
  Status SetCancellationManager(
      TFE_CancellationManager* cancellation_manager) override;

  // TODO(gjn): Remove once TFE_InferShapes is removed
  const tensorflow::AttrBuilder& Attrs() const { return operation_.Attrs(); }
  tensorflow::AttrBuilder* MutableAttrs() { return operation_.MutableAttrs(); }

  const TensorHandle* GetInput(int i) const { return operation_.Inputs()[i]; }

 private:
  const tensorflow::OpDef* GetOpDef(Status* status);
  EagerOperation operation_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_OPERATION_INTERFACE_H_
