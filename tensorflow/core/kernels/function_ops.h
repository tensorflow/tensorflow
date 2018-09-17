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

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

static const char* const kArgOp = FunctionLibraryDefinition::kArgOp;
static const char* const kRetOp = FunctionLibraryDefinition::kRetOp;

class ArgOp : public OpKernel {
 public:
  explicit ArgOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

  bool IsExpensive() override { return false; }

 private:
  int index_;
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(ArgOp);
};

class RetvalOp : public OpKernel {
 public:
  explicit RetvalOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

  bool IsExpensive() override { return false; }

 private:
  int index_;
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(RetvalOp);
};

class RemoteCallOp : public AsyncOpKernel {
 public:
  explicit RemoteCallOp(OpKernelConstruction* ctx);

  ~RemoteCallOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  NameAttrList func_;
  DataTypeVector input_dtypes_;
  DataTypeVector output_dtypes_;

  mutex mu_;
  typedef std::pair<string, FunctionLibraryRuntime*> FunctionTarget;
  std::map<FunctionTarget, FunctionLibraryRuntime::Handle> handle_cache_
      GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RemoteCallOp);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_FUNCTION_OPS_H_
