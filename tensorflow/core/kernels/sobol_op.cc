/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Based on "Notes on generating Sobol sequences. August 2008" by Joe and Kuo.
// [1] https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "third_party/eigen3/Eigen/Core"
#include "sobol_data.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/platform_strings.h"

namespace tensorflow {

// Embed the platform strings in this binary.
TF_PLATFORM_STRINGS()

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {

// Each thread will calculate at least kMinBlockSize points in the sequence.
constexpr int kMinBlockSize = 512;

// Returns number of digits in binary representation of n.
// Example: n=13. Binary representation is 1101. NumBinaryDigits(13) -> 4.
int NumBinaryDigits(int n) {
  return static_cast<int>(std::log2(n) + 1);
}

// Returns position of rightmost zero digit in binary representation of n.
// Example: n=13. Binary representation is 1101. RightmostZeroBit(13) -> 1.
int RightmostZeroBit(int n) {
  int k = 0;
  while (n & 1) {
    n >>= 1;
    ++k;
  }
  return k;
}

// Returns an integer representation of point `i` in the Sobol sequence of
// dimension `dim` using the given direction numbers.
Eigen::VectorXi GetFirstPoint(int i, int dim,
                              const Eigen::MatrixXi& direction_numbers) {
  // Index variables used in this function, consistent with notation in [1].
  // i - point in the Sobol sequence
  // j - dimension
  // k - binary digit
  Eigen::VectorXi integer_sequence = Eigen::VectorXi::Zero(dim);
  // go/wiki/Sobol_sequence#A_fast_algorithm_for_the_construction_of_Sobol_sequences
  int gray_code = i ^ (i >> 1);
  int num_digits = NumBinaryDigits(i);
  for (int j = 0; j < dim; ++j) {
    for (int k = 0; k < num_digits; ++k) {
      if ((gray_code >> k) & 1) integer_sequence(j) ^= direction_numbers(j, k);
    }
  }
  return integer_sequence;
}

// Calculates `num_results` Sobol points of dimension `dim` starting at the
// point `start_point + skip` and writes them into `output` starting at point
// `start_point`.
template <typename T>
void CalculateSobolSample(int32_t dim, int32_t num_results, int32_t skip,
                          int32_t start_point,
                          typename TTypes<T>::Flat output) {
  // Index variables used in this function, consistent with notation in [1].
  // i - point in the Sobol sequence
  // j - dimension
  // k - binary digit
  const int num_digits =
      NumBinaryDigits(skip + start_point + num_results + 1);
  Eigen::MatrixXi direction_numbers(dim, num_digits);

  // Shift things so we can use integers everywhere. Before we write to output,
  // divide by constant to convert back to floats.
  const T normalizing_constant = 1./(1 << num_digits);
  for (int j = 0; j < dim; ++j) {
    for (int k = 0; k < num_digits; ++k) {
      direction_numbers(j, k) = sobol_data::kDirectionNumbers[j][k]
                                << (num_digits - k - 1);
    }
  }

  // If needed, skip ahead to the appropriate point in the sequence. Otherwise
  // we start with the first column of direction numbers.
  Eigen::VectorXi integer_sequence =
      (skip + start_point > 0)
          ? GetFirstPoint(skip + start_point + 1, dim, direction_numbers)
          : direction_numbers.col(0);

  for (int j = 0; j < dim; ++j) {
    output(start_point * dim + j) = integer_sequence(j) * normalizing_constant;
  }
  // go/wiki/Sobol_sequence#A_fast_algorithm_for_the_construction_of_Sobol_sequences
  for (int i = start_point + 1; i < num_results + start_point; ++i) {
    // The Gray code for the current point differs from the preceding one by
    // just a single bit -- the rightmost bit.
    int k = RightmostZeroBit(i + skip);
    // Update the current point from the preceding one with a single XOR
    // operation per dimension.
    for (int j = 0; j < dim; ++j) {
      integer_sequence(j) ^= direction_numbers(j, k);
      output(i * dim + j) = integer_sequence(j) * normalizing_constant;
    }
  }
}

}  // namespace

template <typename Device, typename T>
class SobolSampleOp : public OpKernel {
 public:
  explicit SobolSampleOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    int32_t dim = context->input(0).scalar<int32_t>()();
    int32_t num_results = context->input(1).scalar<int32_t>()();
    int32_t skip = context->input(2).scalar<int32_t>()();

    OP_REQUIRES(context, dim >= 1,
                errors::InvalidArgument("dim must be at least one"));
    OP_REQUIRES(context, dim <= sobol_data::kMaxSobolDim,
                errors::InvalidArgument("dim must be at most ",
                                        sobol_data::kMaxSobolDim));
    OP_REQUIRES(context, num_results >= 1,
                errors::InvalidArgument("num_results must be at least one"));
    OP_REQUIRES(context, skip >= 0,
                errors::InvalidArgument("skip must be non-negative"));
    OP_REQUIRES(context,
                num_results < std::numeric_limits<int32_t>::max() - skip,
                errors::InvalidArgument("num_results+skip must be less than ",
                                        std::numeric_limits<int32_t>::max()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({num_results, dim}), &output));
    auto output_flat = output->flat<T>();
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    int num_threads = worker_threads.num_threads;
    int block_size = std::max(
        kMinBlockSize, static_cast<int>(std::ceil(
                           static_cast<float>(num_results) / num_threads)));
    worker_threads.workers->TransformRangeConcurrently(
        block_size, num_results /* total */,
        [&dim, &skip, &output_flat](const int start, const int end) {
          CalculateSobolSample<T>(dim, end - start /* num_results */, skip,
                                  start, output_flat);
        });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SobolSample").Device(DEVICE_CPU).TypeConstraint<double>("dtype"),
    SobolSampleOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(
    Name("SobolSample").Device(DEVICE_CPU).TypeConstraint<float>("dtype"),
    SobolSampleOp<CPUDevice, float>);

}  // namespace tensorflow
