/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/isotonic_regression_op.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace {

using ::int64_t;
using tensorflow::int32;
using tensorflow::isotonic_regression::L2PavaSegment;
using tensorflow::isotonic_regression::Segment;
using tensorflow::isotonic_regression::SolvePava1D;

// The # of ops estimated for the isotonic regression solver is the size of the
// array multiplied by this constant. This is used by the thread pool executor
// when deciding how many threads to use.
constexpr int kCostMultiplier = 100;

// Solve a batch of problems using the pool-adjacent violators algorithm.
// The problems are solved in parallel using tensorflow's thread pool.
template <typename Tin, typename Tout, typename FloatTensor, typename IntTensor>
void solve_pava_batch(const Tin* input_data, FloatTensor* solution,
                      IntTensor* segments,
                      tensorflow::OpKernelContext* context) {
  const int batch_size = solution->dimensions()[0];
  const int problem_size = solution->dimensions()[1];

  auto thread_pool =
      context->device()->tensorflow_cpu_worker_threads()->workers;

  thread_pool->ParallelFor(
      batch_size, kCostMultiplier * problem_size,
      [input_data, solution, segments, problem_size](int64_t row_start,
                                                     int64_t row_limit) {
        std::vector<L2PavaSegment<Tout>> pools;
        pools.reserve(problem_size);
        for (int row_index = static_cast<int>(row_start);
             row_index < static_cast<int>(row_limit); ++row_index) {
          const Tin* row_in = input_data + row_index * problem_size;
          Tout* row_sol = &(*solution)(row_index, 0);
          int32* row_seg = &(*segments)(row_index, 0);
          SolvePava1D<Tin, Tout>(row_in, row_sol, row_seg, problem_size,
                                 &pools);
        }
      });
}

}  // namespace

template <typename Tin, typename Tout>
class IsotonicRegressionOp : public tensorflow::OpKernel {
 public:
  explicit IsotonicRegressionOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Grab the input tensor.
    const tensorflow::Tensor& input_tensor = context->input(0);
    const auto input = input_tensor.flat_inner_dims<Tin, 2>();
    int int_max = std::numeric_limits<int32_t>::max();
    OP_REQUIRES(context,
                tensorflow::FastBoundsCheck(input.dimensions()[0], int_max) &&
                    tensorflow::FastBoundsCheck(input.dimensions()[1], int_max),
                absl::InvalidArgumentError("Tensor too large"));

    // Create the output tensor holding the minimizers.
    const auto shape = input_tensor.shape();
    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, shape, &output_tensor));
    auto output = output_tensor->flat_inner_dims<Tout, 2>();

    // Create the output tensor holding the segment memberships.
    tensorflow::Tensor* segments_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, shape, &segments_tensor));
    auto segments = segments_tensor->flat_inner_dims<int>();

    solve_pava_batch<Tin, Tout>(input.data(), &output, &segments, context);
  }
};

#define REGISTER_CPU_KERNEL(Tin, Tout)                               \
  REGISTER_KERNEL_BUILDER(Name("IsotonicRegression")                 \
                              .Device(tensorflow::DEVICE_CPU)        \
                              .TypeConstraint<Tin>("T")              \
                              .TypeConstraint<Tout>("output_dtype"), \
                          IsotonicRegressionOp<Tin, Tout>);

// Float types have the same input and output.
#define REGISTER_CPU_SAME_KERNEL(T) REGISTER_CPU_KERNEL(T, T)
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SAME_KERNEL);

// 8 and 16 bit integers get converted to 32 bit floats.
#define REGISTER_CPU_KERNEL_FLOAT(Tin) REGISTER_CPU_KERNEL(Tin, float)
TF_CALL_int16(REGISTER_CPU_KERNEL_FLOAT);
TF_CALL_int8(REGISTER_CPU_KERNEL_FLOAT);

// 32 and 64 bit integers get converted to 64 bit floats.
#define REGISTER_CPU_KERNEL_DOUBLE(Tin) REGISTER_CPU_KERNEL(Tin, double)
TF_CALL_int64(REGISTER_CPU_KERNEL_DOUBLE);
TF_CALL_int32(REGISTER_CPU_KERNEL_DOUBLE);

}  // namespace tensorflow
