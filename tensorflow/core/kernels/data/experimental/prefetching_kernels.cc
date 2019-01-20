/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <deque>

#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace data {
namespace {

class IteratorGetDeviceOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
    // NOTE(mrry): We do not currently Validate that the handle
    // corresponds to a real IteratorResource, because that symbol is
    // not exposed from the framework library.
    Tensor* device_name_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &device_name_t));
    // NOTE(mrry): Since the operation's input is a resource, we must be
    // colocated with it, and so we can simply return the current device's
    // name without looking at the input.
    device_name_t->scalar<string>()() = ctx->device()->name();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("ExperimentalIteratorGetDevice").Device(DEVICE_CPU),
    IteratorGetDeviceOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
