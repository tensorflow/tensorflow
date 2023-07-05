/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_HOLDER_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_HOLDER_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"

namespace tensorflow {

// TensorHolder is used to hold some tensors temporarily until the object is
// destructured. This prevents the tensor from being destroyed prematurely.
class TensorHolder {
 public:
  ~TensorHolder() {
    mutex_lock l(lock_);
    VLOG(3) << "Tensors to delete: " << tensors_.size();
    for (auto& ref : tensors_) {
      ref->Unref();
    }
  }

  void AddTensor(const Tensor& tensor) {
    mutex_lock l(lock_);
    tensors_.push_back(absl::make_unique<TensorReference>(tensor));
  }

 private:
  mutex lock_;
  std::vector<std::unique_ptr<TensorReference>> tensors_ TF_GUARDED_BY(lock_);
};

}  // namespace tensorflow

#endif
