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

#ifndef TENSORFLOW_CORE_KERNELS_FUNCTION_OPS_H_
#define TENSORFLOW_CORE_KERNELS_FUNCTION_OPS_H_

#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

static const char* const kArgOp = FunctionLibraryDefinition::kArgOp;
static const char* const kDeviceArgOp = FunctionLibraryDefinition::kDeviceArgOp;
static const char* const kRetOp = FunctionLibraryDefinition::kRetOp;
static const char* const kDeviceRetOp = FunctionLibraryDefinition::kDeviceRetOp;

class ArgOp : public OpKernel {
 public:
  explicit ArgOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

  bool IsExpensive() override { return false; }

 private:
  int index_;
  DataType dtype_;

  ArgOp(const ArgOp&) = delete;
  void operator=(const ArgOp&) = delete;
};

class RetvalOp : public OpKernel {
 public:
  explicit RetvalOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

  bool IsExpensive() override { return false; }

 private:
  int index_;
  DataType dtype_;

  RetvalOp(const RetvalOp&) = delete;
  void operator=(const RetvalOp&) = delete;
};

class RemoteCallOp : public AsyncOpKernel {
 public:
  explicit RemoteCallOp(OpKernelConstruction* ctx);

  ~RemoteCallOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

  string TraceString(const OpKernelContext& ctx, bool verbose) const override;

 private:
  NameAttrList func_;
  DataTypeVector input_dtypes_;
  DataTypeVector output_dtypes_;
  // Note that in the future if all RemoteCall ops have full type
  // information, the kernel will not need access to the "Tout" Attr and
  // return_type_ will replace output_dtypes_.
  FullTypeDef return_type_;

  mutex mu_;
  typedef std::pair<string, FunctionLibraryRuntime*> FunctionTarget;
  std::map<FunctionTarget, FunctionLibraryRuntime::Handle> handle_cache_
      TF_GUARDED_BY(mu_);

  RemoteCallOp(const RemoteCallOp&) = delete;
  void operator=(const RemoteCallOp&) = delete;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_FUNCTION_OPS_H_
