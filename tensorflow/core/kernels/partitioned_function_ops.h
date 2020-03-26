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

#ifndef TENSORFLOW_CORE_KERNELS_PARTITIONED_FUNCTION_OPS_H_
#define TENSORFLOW_CORE_KERNELS_PARTITIONED_FUNCTION_OPS_H_

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class NameAttrList;
class ConfigProto;

// A `PartitionedCallOp` asynchronously executes a function, potentially across
// multiple devices but within a single process. The kernel places and
// partitions a given function's underlying graph, and executes each of the
// partitioned subgraphs as a function.
//
// TODO(akshayka): Support distributed execution.
class PartitionedCallOp : public AsyncOpKernel {
 public:
  explicit PartitionedCallOp(OpKernelConstruction* ctx);

  ~PartitionedCallOp() override;

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  Status FillOutputDevices(const FunctionLibraryRuntime& lib,
                           const Device& cpu_device, AttrSlice attrs,
                           FunctionLibraryRuntime::InstantiateOptions* opts);

  Status Instantiate(FunctionLibraryRuntime* lib, OpKernelContext* ctx,
                     std::vector<Tensor>* inputs,
                     FunctionLibraryRuntime::Handle* handle);

  void RunFunction(FunctionLibraryRuntime::Handle handle,
                   const std::vector<Tensor>& inputs,
                   FunctionLibraryRuntime* lib, OpKernelContext* ctx,
                   DoneCallback done);

  // Using unique pointers to avoid including proto headers in kernel headers
  std::unique_ptr<NameAttrList> func_;
  std::unique_ptr<ConfigProto> config_proto_;
  string executor_type_;
  mutex mu_;
  // Cache the handle per FLR because this kernel may be instantiated for
  // a stateful op, different invocations of it may use different FLRs.
  // Different device placements of PartitionedCallOp also use
  // different FLRs.
  gtl::FlatMap<FunctionLibraryRuntime*, FunctionLibraryRuntime::Handle> handles_
      TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_PARTITIONED_FUNCTION_OPS_H_
