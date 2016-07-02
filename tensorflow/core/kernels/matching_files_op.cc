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

// See docs in ../ops/io_ops.cc.

#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/match.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class MatchingFilesOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* context) override {
    const Tensor* pattern;
    OP_REQUIRES_OK(context, context->input("pattern", &pattern));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(pattern->shape()),
                errors::InvalidArgument(
                    "Input pattern tensor must be scalar, but had shape: ",
                    pattern->shape().DebugString()));
    std::vector<string> fnames;
    OP_REQUIRES_OK(context,
                   io::GetMatchingFiles(context->env(),
                                        pattern->scalar<string>()(), &fnames));
    const int num_out = fnames.size();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "filenames", TensorShape({num_out}), &output));
    auto output_vec = output->vec<string>();
    for (int i = 0; i < num_out; ++i) {
      output_vec(i) = fnames[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MatchingFiles").Device(DEVICE_CPU),
                        MatchingFilesOp);

}  // namespace tensorflow
