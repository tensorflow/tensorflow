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

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace {

class CheckNumericsOp : public XlaOpKernel {
 public:
  explicit CheckNumericsOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // TODO(b/32223192): add a real implementation of CheckNumerics
    {
      static mutex mu(tensorflow::LINKER_INITIALIZED);
      static int log_counter = 0;
      mutex_lock l(mu);
      if (log_counter < 20) {
        ++log_counter;
        LOG(WARNING) << "Ignoring CheckNumerics operator " << name();
      }
    }
    ctx->SetOutput(0, ctx->Input(0));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CheckNumericsOp);
};

REGISTER_XLA_OP(Name("CheckNumerics"), CheckNumericsOp);

}  // anonymous namespace
}  // namespace tensorflow
