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

#include "unicode/errorcode.h"  // from @icu
#include "unicode/uscript.h"  // from @icu
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class UnicodeScriptOp : public OpKernel {
 public:
  explicit UnicodeScriptOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<int32>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    icu::ErrorCode status;
    for (int i = 0; i < input_flat.size(); i++) {
      UScriptCode script_code = uscript_getScript(input_flat(i), status);
      if (status.isSuccess()) {
        output_flat(i) = script_code;
      } else {
        output_flat(i) = -1;
        status.reset();
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("UnicodeScript").Device(DEVICE_CPU),
                        UnicodeScriptOp);

}  // namespace tensorflow
