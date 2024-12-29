/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_READER_OP_KERNEL_H_
#define TENSORFLOW_CORE_FRAMEWORK_READER_OP_KERNEL_H_

#include <functional>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/reader_interface.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// NOTE: This is now a very thin layer over ResourceOpKernel.
// TODO(sjhwang): Remove dependencies to this class, then delete this.

// Implementation for ops providing a Reader.
class ReaderOpKernel : public ResourceOpKernel<ReaderInterface> {
 public:
  using ResourceOpKernel::ResourceOpKernel;

  // Must be called by descendants before the first call to Compute() (typically
  // called during construction).  factory must return a ReaderInterface
  // descendant allocated with new that ReaderOpKernel will take ownership of.
  void SetReaderFactory(std::function<ReaderInterface*()> factory)
      TF_LOCKS_EXCLUDED(mu_) {
    DCHECK(get_resource() == nullptr);
    mutex_lock l(mu_);
    factory_ = factory;
  }

  void Compute(OpKernelContext* context) override {
    if (!IsCancellable()) {
      ResourceOpKernel<ReaderInterface>::Compute(context);
    } else {
      // Install cancellation
      CancellationManager* cm = context->cancellation_manager();
      CancellationToken token = cm->get_cancellation_token();
      bool already_cancelled =
          !cm->RegisterCallback(token, [this]() { this->Cancel(); });

      if (!already_cancelled) {
        ResourceOpKernel<ReaderInterface>::Compute(context);
      } else {
        context->SetStatus(errors::Cancelled("read operation was cancelled"));
      }
    }
  }

 private:
  virtual bool IsCancellable() const { return false; }
  virtual void Cancel() {}

  absl::Status CreateResource(ReaderInterface** reader)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *reader = factory_();
    if (*reader == nullptr) {
      return errors::ResourceExhausted("Failed to allocate reader");
    }
    std::function<ReaderInterface*()> temp = nullptr;
    factory_.swap(temp);
    return absl::OkStatus();
  }

  std::function<ReaderInterface*()> factory_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_READER_OP_KERNEL_H_
