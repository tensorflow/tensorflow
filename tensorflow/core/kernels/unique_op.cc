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

#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
class UniqueOp : public OpKernel {
 public:
  explicit UniqueOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input.shape()),
                errors::InvalidArgument("unique expects a 1D vector."));
    // TODO(dga):  Make unique polymorphic for returning int32 and int64
    // vectors to support large tensors.
    OP_REQUIRES(context,
                input.NumElements() <= std::numeric_limits<int32>::max(),
                errors::InvalidArgument(
                    "unique does not support input tensors larger than ",
                    std::numeric_limits<int32>::max(), " elements"));
    auto Tin = input.vec<T>();
    const int64 N = static_cast<int64>(Tin.size());

    Tensor* idx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, input.shape(), &idx));
    auto idx_vec = idx->template vec<int32>();

    std::unordered_map<T, int32> uniq;
    uniq.reserve(2 * N);
    for (int64 i = 0, j = 0; i < N; ++i) {
      auto it = uniq.insert(std::make_pair(Tin(i), j));
      idx_vec(i) = it.first->second;
      if (it.second) {
        ++j;
      }
    }
    int64 uniq_size = static_cast<int64>(uniq.size());
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({uniq_size}), &output));
    auto output_vec = output->template vec<T>();

    for (auto it : uniq) {
      output_vec(it.second) = it.first;
    }

    if (num_outputs() > 2) {
      OP_REQUIRES_OK(context, context->allocate_output(
                                  2, TensorShape({uniq_size}), &output));
      auto count_output_vec = output->template vec<int32>();
      count_output_vec.setZero();
      for (int64 i = 0; i < N; ++i) {
        count_output_vec(idx_vec(i))++;
      }
    }
  }
};

#define REGISTER_UNIQUE(type)                                                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("Unique").Device(DEVICE_CPU).TypeConstraint<type>("T"),           \
      UniqueOp<type>);                                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("UniqueWithCounts").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      UniqueOp<type>)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_UNIQUE);
REGISTER_UNIQUE(string)
#undef REGISTER_UNIQUE
}  // namespace tensorflow
