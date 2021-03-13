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

#ifndef TENSORFLOW_CORE_KERNELS_SENDRECV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_SENDRECV_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class SendOp : public OpKernel {
 public:
  explicit SendOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

  string TraceString(const OpKernelContext& ctx, bool verbose) const override;

 private:
  string key_prefix_;
  Rendezvous::ParsedKey parsed_key_;
  bool hostmem_sendrecv_;

  TF_DISALLOW_COPY_AND_ASSIGN(SendOp);
};

class RecvOp : public AsyncOpKernel {
 public:
  explicit RecvOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

  string TraceString(const OpKernelContext& ctx, bool verbose) const override;

 private:
  string key_prefix_;
  Rendezvous::ParsedKey parsed_key_;
  bool hostmem_sendrecv_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvOp);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SENDRECV_OPS_H_
