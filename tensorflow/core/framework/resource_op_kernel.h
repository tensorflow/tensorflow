/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RESOURCE_OP_KERNEL_H_
#define TENSORFLOW_CORE_FRAMEWORK_RESOURCE_OP_KERNEL_H_

#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// ResourceOpKernel<T> is a virtual base class for resource op implementing
// interface type T. The inherited op looks up the resource name (determined by
// ContainerInfo), and creates a new resource if necessary.
//
// Requirements:
//  - Op must be marked as stateful.
//  - Op must have `container` and `shared_name` attributes. Empty `container`
//  means using the default container. Empty `shared_name` means private
//  resource.
//  - Subclass must override CreateResource().
//  - Subclass is encouraged to override VerifyResource().
template <typename T>
class ResourceOpKernel : public OpKernel {
 public:
  explicit ResourceOpKernel(OpKernelConstruction* context) : OpKernel(context) {
    has_resource_type_ = (context->output_type(0) == DT_RESOURCE);
    if (!has_resource_type_) {
      // The resource variant of the op may be placed on non-CPU devices, but
      // this allocation is always on the host. Fortunately we don't need it in
      // the resource case.
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DT_STRING, TensorShape({2}), &tensor_));
    }
  }

  // The resource is deleted from the resource manager only when it is private
  // to kernel. Ideally the resource should be deleted when it is no longer held
  // by anyone, but it would break backward compatibility.
  ~ResourceOpKernel() override {
    if (resource_ != nullptr) {
      resource_->Unref();
      if (cinfo_.resource_is_private_to_kernel()) {
        if (!cinfo_.resource_manager()
                 ->template Delete<T>(cinfo_.container(), cinfo_.name())
                 .ok()) {
          // Do nothing; the resource can have been deleted by session resets.
        }
      }
    }
  }

  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (resource_ == nullptr) {
      ResourceMgr* mgr = context->resource_manager();
      OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

      T* resource;
      OP_REQUIRES_OK(context,
                     mgr->LookupOrCreate<T>(
                         cinfo_.container(), cinfo_.name(), &resource,
                         [this](T** ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                           Status s = CreateResource(ret);
                           if (!s.ok() && *ret != nullptr) {
                             CHECK((*ret)->Unref());
                           }
                           return s;
                         }));

      Status s = VerifyResource(resource);
      if (TF_PREDICT_FALSE(!s.ok())) {
        resource->Unref();
        context->SetStatus(s);
        return;
      }

      if (!has_resource_type_) {
        auto h = tensor_.template flat<tstring>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
      resource_ = resource;
    }
    if (has_resource_type_) {
      OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                  context, 0, cinfo_.container(), cinfo_.name(),
                                  TypeIndex::Make<T>()));
    } else {
      context->set_output_ref(0, &mu_, &tensor_);
    }
  }

 protected:
  // Variables accessible from subclasses.
  mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
  T* resource_ TF_GUARDED_BY(mu_) = nullptr;

 private:
  // Must return a T descendant allocated with new that ResourceOpKernel will
  // take ownership of.
  virtual Status CreateResource(T** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  virtual Status VerifyResource(T* resource) { return Status::OK(); }

  Tensor tensor_ TF_GUARDED_BY(mu_);

  // Is the output of the operator of type DT_RESOURCE?
  bool has_resource_type_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_RESOURCE_OP_KERNEL_H_
