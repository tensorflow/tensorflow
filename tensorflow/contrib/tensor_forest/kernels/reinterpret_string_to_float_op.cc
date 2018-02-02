// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
// Converts strings of arbitrary length to float values by
// hashing and cramming bits.
#include <functional>

#include "tensorflow/contrib/tensor_forest/kernels/tree_utils.h"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using tensorforest::CheckTensorBounds;

float Convert(const string& in) {
  const std::size_t intval = std::hash<string>()(in);
  return static_cast<float>(intval);
}

void Evaluate(const Tensor& input_data, Tensor output_data, int32 start,
              int32 end) {
  auto out_data = output_data.unaligned_flat<float>();
  const auto in_data = input_data.unaligned_flat<string>();

  for (int32 i = start; i < end; ++i) {
    out_data(i) = Convert(in_data(i));
  }
}

class ReinterpretStringToFloat : public OpKernel {
 public:
  explicit ReinterpretStringToFloat(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);

    // Check tensor bounds.
    if (!CheckTensorBounds(context, input_data)) return;

    Tensor* output_data = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, input_data.shape(), &output_data));

    // Evaluate input data in parallel.
    const int32 num_data = static_cast<int32>(input_data.NumElements());
    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;
    if (num_threads <= 1) {
      Evaluate(input_data, *output_data, 0, num_data);
    } else {
      auto work = [&input_data, output_data, num_data](int64 start, int64 end) {
        CHECK(start <= end);
        CHECK(end <= num_data);
        Evaluate(input_data, *output_data, static_cast<int32>(start),
                 static_cast<int32>(end));
      };
      Shard(num_threads, worker_threads->workers, num_data, 100, work);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ReinterpretStringToFloat").Device(DEVICE_CPU),
                        ReinterpretStringToFloat);

}  // namespace tensorflow
