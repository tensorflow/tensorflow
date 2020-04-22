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

#include "absl/types/span.h"
#include "tensorflow/c/eager/tensor_handle_interface.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"

struct TFE_Op;

namespace tensorflow {

// Abstract interface to an operation.
class AbstractOperationInterface {
 public:
  // Release any underlying resources, including the interface object.
  //
  // WARNING: The destructor of this class is marked as protected to disallow
  // clients from directly destroying this object since it may manage it's own
  // lifetime through ref counting. Thus this must be allocated on the heap and
  // clients MUST call Release() in order to destroy an instance of this class.
  virtual void Release() = 0;

  virtual void Clear() = 0;
  virtual Status Reset(const char* op, const char* raw_device_name) = 0;

  virtual const string& Name() const = 0;

  // Returns the operation's device name.
  //
  // The value returned may be different from the one set by SetDeviceName, but
  // it will be compatible with it: the name will be updated by device placement
  // logic to refer to the specific device chosen.
  //
  // Example: If one calls `op->SetDeviceName("/device:GPU")`, the value
  // returned by DeviceName should be "/device:GPU:*" until a particular GPU is
  // chosen for the operation by the device placement logic in the
  // executor. After that, the value returned by DeviceName will be a full
  // device name such as "/job:localhost/replica:0/task:0/device:GPU:1".
  virtual const string& DeviceName() const = 0;

  // Sets the operation device name.
  //
  // The given `name` must be parseable by DeviceNameUtils::ParseFullName, and
  // the result will be used as a constraint for device placement. See the
  // documentation for DeviceName for more details.
  //
  // The value will override the previous value - that is, no "merging" of
  // existing and given constraints will be performed.
  virtual Status SetDeviceName(const char* name) = 0;

  virtual Status AddInput(AbstractTensorHandleInterface* input) = 0;
  virtual Status AddInputList(
      absl::Span<AbstractTensorHandleInterface*> inputs) = 0;
  virtual Status Execute(absl::Span<AbstractTensorHandleInterface*> retvals,
                         int* num_retvals) = 0;
  virtual const tensorflow::OpDef* OpDef() const = 0;

  virtual Status SetAttrString(const char* attr_name, const char* data,
                               size_t length) = 0;
  virtual Status SetAttrInt(const char* attr_name, int64_t value) = 0;
  virtual Status SetAttrFloat(const char* attr_name, float value) = 0;
  virtual Status SetAttrBool(const char* attr_name, bool value) = 0;
  virtual Status SetAttrType(const char* attr_name, DataType value) = 0;
  virtual Status SetAttrShape(const char* attr_name, const int64_t* dims,
                              const int num_dims) = 0;
  virtual Status SetAttrFunction(const char* attr_name,
                                 const AbstractOperationInterface* value) = 0;
  virtual Status SetAttrFunctionName(const char* attr_name, const char* value,
                                     size_t length) = 0;
  virtual Status SetAttrTensor(const char* attr_name,
                               AbstractTensorInterface* tensor) = 0;
  virtual Status SetAttrStringList(const char* attr_name,
                                   const void* const* values,
                                   const size_t* lengths, int num_values) = 0;
  virtual Status SetAttrFloatList(const char* attr_name, const float* values,
                                  int num_values) = 0;
  virtual Status SetAttrIntList(const char* attr_name, const int64_t* values,
                                int num_values) = 0;
  virtual Status SetAttrTypeList(const char* attr_name, const DataType* values,
                                 int num_values) = 0;
  virtual Status SetAttrBoolList(const char* attr_name,
                                 const unsigned char* values,
                                 int num_values) = 0;
  virtual Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                                  const int* num_dims, int num_values) = 0;
  virtual Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperationInterface*> values) = 0;

  virtual Status InputLength(const char* input_name, int* length) = 0;
  virtual Status OutputLength(const char* output_name, int* length) = 0;

  // Experimental
  virtual Status SetUseXla(bool enable) = 0;

 protected:
  virtual ~AbstractOperationInterface() {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_OPERATION_INTERFACE_H_
