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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

REGISTER_RESOURCE_HANDLE_KERNEL(Var);

class CreateVariableOp : public OpKernel {
 public:
  CreateVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* c) override {
    Var* var = new Var(dtype_);
    var->Ref();
    core::ScopedUnref ur(var);
    OP_REQUIRES_OK(c, CreateResource<Var>(c, HandleFromInput(c, 0), var));
    // TODO(apassos): this currently does not initialize the tensor, so it's
    // pointless, other than checking construction in tests. Fix this.
  }

 private:
  DataType dtype_;
};
REGISTER_KERNEL_BUILDER(Name("CreateVariableOp").Device(DEVICE_CPU),
                        CreateVariableOp);

}  // namespace tensorflow
