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
namespace tensorflow {

// Abstract interface to a Tensor handle in either tracing or immediate
// execution mode.
class AbstractTensorHandle {
 protected:
  enum AbstractTensorHandleKind { kGraph, kMlir, kEager, kTfrt };
  explicit AbstractTensorHandle(AbstractTensorHandleKind kind) : kind_(kind) {}
  virtual ~AbstractTensorHandle() {}

 public:
  AbstractTensorHandleKind getKind() const { return kind_; }

  // Release any underlying resources, including the interface object.
  //
  // WARNING: The destructor of this class is marked as protected to disallow
  // clients from directly destroying this object since it may manage it's own
  // lifetime through ref counting. Thus this must be allocated on the heap and
  // clients MUST call Release() in order to destroy an instance of this class.
  virtual void Release() = 0;

 private:
  const AbstractTensorHandleKind kind_;
};

namespace internal {
struct AbstractTensorHandleDeleter {
  void operator()(AbstractTensorHandle* p) const {
    if (p != nullptr) {
      p->Release();
    }
  }
};
}  // namespace internal

using AbstractTensorHandlePtr =
    std::unique_ptr<AbstractTensorHandle,
                    internal::AbstractTensorHandleDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_ABSTRACT_TENSOR_HANDLE_H_
