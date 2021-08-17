/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/get_options_op.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace data {

void GetOptionsOp::Compute(OpKernelContext* ctx) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  if (ctx->status().ok()) {
    Tensor* string_handle_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &string_handle_t));
    string_handle_t->scalar<tstring>()() = input->options().SerializeAsString();
  }
}

string GetOptionsOp::TraceString(const OpKernelContext& ctx,
                                 bool verbose) const {
  return profiler::TraceMeOp(name_view(), type_string_view());
}

namespace {
REGISTER_KERNEL_BUILDER(Name("GetOptions").Device(DEVICE_CPU).Priority(2),
                        GetOptionsOp);
REGISTER_KERNEL_BUILDER(Name("GetOptions")
                            .Device(DEVICE_GPU)
                            .HostMemory("input_dataset")
                            .HostMemory("serialized_options")
                            .Priority(1),
                        GetOptionsOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
