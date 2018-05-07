/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_

#include "tensorflow/c/eager/runtime.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

namespace tensorflow {
class EagerOperation {
 public:
  // t is NULL iff the EagerOperation corresponds to a TensorFlow function
  // instead of a primitive operation.
  EagerOperation(tensorflow::EagerContext* ctx, const char* op,
                 const tensorflow::AttrTypeMap* t)
      : ctx_(ctx), name_(op), attrs_(op), attr_types_(t), device_(nullptr) {}

  ~EagerOperation() {
    for (tensorflow::TensorHandle* h : inputs_) {
      h->Unref();
    }
  }

  bool is_function() const { return attr_types_ == nullptr; }

  tensorflow::EagerContext* EagerContext() { return ctx_; }

  tensorflow::AttrBuilder* MutableAttrs() { return &attrs_; }
  const tensorflow::AttrBuilder& Attrs() const { return attrs_; }

  const tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 4>& Inputs()
      const {
    return inputs_;
  }
  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 4>*
  MutableInputs() {
    return &inputs_;
  }
  void AddInput(tensorflow::TensorHandle* h);

  const tensorflow::string& Name() const { return name_; }
  const tensorflow::AttrTypeMap* AttrTypes() const { return attr_types_; }

  tensorflow::Device* Device() const { return device_; }
  tensorflow::Status SetDevice(const char* device);
  void SetDevice(tensorflow::Device* device) { device_ = device; }

  void SetUseXla(bool use_xla) { use_xla_ = use_xla; }

 private:
  tensorflow::EagerContext* ctx_;  // Must outlive the EagerOperation.
  const tensorflow::string name_;
  tensorflow::AttrBuilder attrs_;
  const tensorflow::AttrTypeMap* attr_types_;
  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 4> inputs_;
  tensorflow::Device* device_;
  bool use_xla_ = false;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_
