/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_DATA_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_DATA_H_

#include <utility>
#include <variant>

#include "absl/types/variant.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Local Tensor Handle: Handle to a Tensor present on the local host.
class LocalTensorHandleData {
 public:
  LocalTensorHandleData() : ctrl_(absl::in_place_type<BlockingControl>) {}
  explicit LocalTensorHandleData(tensorflow::Tensor&& t)
      : tensor_(std::move(t)),
        forwarding_protection_tensor_(tensor_),
        ctrl_(absl::in_place_type<NonBlockingControl>) {}

  // A local tensor handle should be able to satisfy all of these requests.
  absl::Status Tensor(const tensorflow::Tensor** t) const;
  absl::Status TensorValue(tensorflow::TensorValue* t);
  absl::Status Shape(TensorShape* shape) const;
  absl::Status NumDims(int* num_dims) const;
  absl::Status Dim(int dim_index, int64_t* dim) const;
  absl::Status NumElements(int64_t* num_elements) const;
  absl::Status Unprotect();

  bool IsReady() const {
    return std::visit([](auto& data) { return data.IsReady(); }, ctrl_);
  }

  absl::Status WaitReady(const char* caller) const {
    return std::visit([caller](auto& data) { return data.WaitReady(caller); },
                      ctrl_);
  }
  void Poison(absl::Status status) {
    return std::visit([status](auto& data) { data.Poison(status); }, ctrl_);
  }
  absl::Status IsPoisoned() const {
    return std::visit([](auto& data) { return data.IsPoisoned(); }, ctrl_);
  }

  absl::Status SetTensor(tensorflow::Tensor&& t);

  string DebugString() const;

 private:
  tensorflow::Tensor tensor_;
  // TensorHandle has its own reference counting which is distinct from the
  // backing Tensor. As a result, if the Tensor reference count is 1 while
  // executing an op, the TensorBuffer could be reused for the output. We avoid
  // this behavior maintaining another reference count with the
  // forwarding_protection_tensor_ Tensor. When Unprotect() is called, we
  // release this Tensor to allow forwarding.
  tensorflow::Tensor forwarding_protection_tensor_;

  // We distinguish between ready and empty tensors with the ctrl_ variant.
  // which contains 2 implementations of the waiting logic. The
  // NonBlockingControl is a simple no-op class whereas the BlockingControl
  // actually uses a mutex. By using a variant we avoid the overhead of
  // constructing and destructing the mutex for ready local tensors.
  class NonBlockingControl {
   public:
    bool IsReady() const { return true; }
    absl::Status WaitReady(const char* caller) const {
      return absl::OkStatus();
    }
    void Poison(absl::Status status) {}
    absl::Status IsPoisoned() const { return absl::OkStatus(); }
  };

  class BlockingControl {
   public:
    bool IsReady() const {
      tf_shared_lock l(mu_);
      return is_ready_;
    }
    void SetReady();
    absl::Status WaitReady(const char* caller) const;
    void Poison(absl::Status status);
    absl::Status IsPoisoned() const {
      tf_shared_lock l(mu_);
      return is_poisoned_;
    }

   private:
    mutable mutex mu_;
    bool is_ready_ TF_GUARDED_BY(mu_);
    absl::Status is_poisoned_ TF_GUARDED_BY(mu_);
  };

  std::variant<NonBlockingControl, BlockingControl> ctrl_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_DATA_H_
