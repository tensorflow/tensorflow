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
#ifndef TENSORFLOW_C_EAGER_ABSTRACT_OPERATION_H_
#define TENSORFLOW_C_EAGER_ABSTRACT_OPERATION_H_

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Abstract interface to an operation.
// This interface allows building and executing an operation in either
// tracing or immediate execution mode.
class AbstractOperation {
 protected:
  enum AbstractOperationKind {
    kGraph,
    kMlir,
    kEager,
    kTfrt,
    kTape,
    kOpHandler
  };
  explicit AbstractOperation(AbstractOperationKind kind) : kind_(kind) {}
  virtual ~AbstractOperation() {}

 public:
  AbstractOperationKind getKind() const { return kind_; }

  // Release any underlying resources, including the interface object.
  //
  // WARNING: The destructor of this class is marked as protected to disallow
  // clients from directly destroying this object since it may manage it's own
  // lifetime through ref counting. Thus this must be allocated on the heap and
  // clients MUST call Release() in order to destroy an instance of this class.
  virtual void Release() = 0;

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

  virtual Status AddInput(AbstractTensorHandle* input) = 0;
  virtual Status AddInputList(
      absl::Span<AbstractTensorHandle* const> inputs) = 0;
  virtual Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                         int* num_retvals) = 0;

  virtual Status SetAttrString(const char* attr_name, const char* data,
                               size_t length) = 0;
  virtual Status SetAttrInt(const char* attr_name, int64_t value) = 0;
  virtual Status SetAttrFloat(const char* attr_name, float value) = 0;
  virtual Status SetAttrBool(const char* attr_name, bool value) = 0;
  virtual Status SetAttrType(const char* attr_name, DataType value) = 0;
  virtual Status SetAttrShape(const char* attr_name, const int64_t* dims,
                              const int num_dims) = 0;
  virtual Status SetAttrShape(const char* attr_name,
                              const PartialTensorShape shape);
  virtual Status SetAttrFunction(const char* attr_name,
                                 const AbstractOperation* value) = 0;
  virtual Status SetAttrFunctionName(const char* attr_name, const char* value,
                                     size_t length) = 0;
  virtual Status SetAttrTensor(const char* attr_name,
                               AbstractTensorInterface* tensor) = 0;
  virtual Status SetAttrStringList(const char* attr_name,
                                   const void* const* values,
                                   const size_t* lengths, int num_values) = 0;
  virtual Status SetAttrStringList(const char* attr_name,
                                   absl::Span<string const> values);
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
      const char* attr_name, absl::Span<const AbstractOperation*> values) = 0;

 private:
  const AbstractOperationKind kind_;
};

// TODO(b/193656009): Defining these in a cc file causes linker errors with
// fastbuild.
inline Status AbstractOperation::SetAttrShape(const char* attr_name,
                                              const PartialTensorShape shape) {
  return SetAttrShape(attr_name, shape.dim_sizes().data(), shape.dims());
}

inline Status AbstractOperation::SetAttrStringList(
    const char* attr_name, absl::Span<string const> values) {
  std::vector<const char*> raw_strs;
  std::vector<size_t> lengths;
  raw_strs.reserve(values.size());
  lengths.reserve(values.size());
  for (const auto& s : values) {
    raw_strs.emplace_back(s.data());
    lengths.emplace_back(s.size());
  }
  return SetAttrStringList(attr_name,
                           reinterpret_cast<const void**>(raw_strs.data()),
                           lengths.data(), values.size());
}

namespace internal {
struct AbstractOperationDeleter {
  void operator()(AbstractOperation* p) const {
    if (p != nullptr) {
      p->Release();
    }
  }
};
}  // namespace internal

using AbstractOperationPtr =
    std::unique_ptr<AbstractOperation, internal::AbstractOperationDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_ABSTRACT_OPERATION_H_
