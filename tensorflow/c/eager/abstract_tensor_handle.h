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
#ifndef TENSORFLOW_C_EAGER_ABSTRACT_TENSOR_HANDLE_H_
#define TENSORFLOW_C_EAGER_ABSTRACT_TENSOR_HANDLE_H_

#include <memory>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Abstract interface to a Tensor handle in either tracing or immediate
// execution mode.
class AbstractTensorHandle : public core::RefCounted {
 protected:
  enum AbstractTensorHandleKind { kGraph, kMlir, kEager, kTfrt, kCustomDevice };
  explicit AbstractTensorHandle(AbstractTensorHandleKind kind) : kind_(kind) {}
  ~AbstractTensorHandle() override {}

 public:
  // Returns tensor dtype.
  virtual tensorflow::DataType DataType() const = 0;

  // Returns the status of the tensor handle. If it is a tfrt::TensorHandle,
  // the tensor handle can be an error and return non-OK status.
  virtual absl::Status TensorHandleStatus() const;

  // Returns tensor shape. If tensor has unknown rank, shape remains untouched.
  virtual absl::Status Shape(tensorflow::PartialTensorShape* shape) const = 0;

  // Returns tensor (full) type.
  // While there is no immediate plan to deprecate dtype and shape in favor
  // of only using full type type information, this is a future possibility.
  //
  // Note that map_dtype_to_child_of_tensor() from core/framework/types.h
  // can be used to set a FullTypeDef based on dtype in a derived class if
  // appropriate.
  virtual tensorflow::FullTypeDef FullType() const = 0;

  // The default debug string includes a shape, dtype and FullType.
  // Implementations are free to override it with something more informative.
  virtual std::string DebugString() const;

  AbstractTensorHandleKind getKind() const { return kind_; }

 private:
  const AbstractTensorHandleKind kind_;
};

namespace internal {
struct AbstractTensorHandleDeleter {
  void operator()(AbstractTensorHandle* p) const {
    if (p != nullptr) {
      p->Unref();
    }
  }
};
}  // namespace internal

// TODO(b/185908092): Make AbstractTensorHandlePtr an IntrusivePtr.
using AbstractTensorHandlePtr =
    std::unique_ptr<AbstractTensorHandle,
                    internal::AbstractTensorHandleDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_ABSTRACT_TENSOR_HANDLE_H_
