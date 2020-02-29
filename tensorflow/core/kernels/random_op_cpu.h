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

#ifndef TENSORFLOW_CORE_KERNELS_RANDOM_OP_CPU_H_
#define TENSORFLOW_CORE_KERNELS_RANDOM_OP_CPU_H_

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/work_sharder.h"

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace functor {
using random::PhiloxRandom;
using random::SingleSampleAdapter;

// The default implementation of the functor, which should never be invoked
// But we still need to provide implementation for now for the linker to work,
// since we do not support all the distributions yet.
template <typename Device, class Distribution>
struct FillPhiloxRandom {
  typedef typename Distribution::ResultElementType T;
  void operator()(OpKernelContext* ctx, const Device&, random::PhiloxRandom gen,
                  T* data, int64 size, Distribution dist) {
    OP_REQUIRES(
        ctx, false,
        errors::Internal(
            "Default `FillPhiloxRandom` implementation should not be executed. "
            "The cause of this error is probably that `FillPhiloxRandom` does "
            "not support this device or random distribution yet."));
  }
};

// A class to fill a specified range of random groups
template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomTask;

// Specialization for distribution that takes a fixed number of samples for
// each output.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  static void Run(random::PhiloxRandom gen, T* data, int64 size,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    gen.Skip(start_group);
    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    for (int64 index = start_group; index < limit_group_full; ++index) {
      auto samples = dist(&gen);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    if (limit_group_full < limit_group) {
      int64 remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist(&gen);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
  }
};

// Specialization for distribution that takes a variable number of samples for
// each output. This will be slower due to the generality.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  static const int64 kReservedSamplesPerOutput = 256;

  static void Run(random::PhiloxRandom base_gen, T* data, int64 size,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    static const int kGeneratorSkipPerOutputGroup =
        kGroupSize * kReservedSamplesPerOutput /
        PhiloxRandom::kResultElementCount;

    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    int64 group_index;
    for (group_index = start_group; group_index < limit_group_full;
         ++group_index) {
      // Reset the generator to the beginning of the output group region
      // This is necessary if we want the results to be independent of order
      // of work
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      auto samples = dist(&single_samples);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    if (limit_group_full < limit_group) {
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      int64 remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist(&single_samples);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
  }
};

// Partial specialization for CPU to fill the entire region with randoms
// It splits the work into several tasks and run them in parallel
template <class Distribution>
void FillPhiloxRandom<CPUDevice, Distribution>::operator()(
    OpKernelContext* context, const CPUDevice&, random::PhiloxRandom gen,
    typename Distribution::ResultElementType* data, int64 size,
    Distribution dist) {
  const int kGroupSize = Distribution::kResultElementCount;

  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

  int64 total_group_count = (size + kGroupSize - 1) / kGroupSize;

  const int kGroupCost =
      random::PhiloxRandom::kResultElementCount *
      (random::PhiloxRandom::kElementCost + Distribution::kElementCost);
  Shard(worker_threads.num_threads, worker_threads.workers, total_group_count,
        kGroupCost,
        [&gen, data, size, dist](int64 start_group, int64 limit_group) {
          FillPhiloxRandomTask<
              Distribution,
              Distribution::kVariableSamplesPerOutput>::Run(gen, data, size,
                                                            start_group,
                                                            limit_group, dist);
        });
}

}  // namespace functor

#ifdef TENSORFLOW_USE_SYCL

namespace functor {

template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomKernel;

template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  using write_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::write,
                                        sycl::access::target::global_buffer>;

  FillPhiloxRandomKernel(write_accessor& data, random::PhiloxRandom& gen,
                         Distribution& dist)
      : data_(data), gen_(gen), dist_(dist) {}

  void operator()(sycl::nd_item<1> item) {
    const size_t kGroupSize = Distribution::kResultElementCount;

    const size_t item_id = item.get_global(0);
    const size_t total_item_count = item.get_global_range();
    size_t offset = item_id * kGroupSize;
    gen_.Skip(item_id);

    const size_t size = data_.get_size() / sizeof(T);
    T* data = ConvertToActualTypeSycl(T, data_);

    while (offset + kGroupSize <= size) {
      const typename Distribution::ResultType samples = dist_(&gen_);
      for (size_t i = 0; i < kGroupSize; ++i) {
        data[offset + i] = samples[i];
      }

      offset += (total_item_count - 1) * kGroupSize;
      gen_.Skip(total_item_count - 1);
    }

    const typename Distribution::ResultType samples = dist_(&gen_);
    for (size_t i = 0; i < kGroupSize; ++i) {
      if (offset >= size) {
        return;
      }
      data[offset] = samples[i];
      ++offset;
    }
  }

 private:
  write_accessor data_;
  random::PhiloxRandom gen_;
  Distribution dist_;
};

template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  using write_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::write,
                                        sycl::access::target::global_buffer>;

  FillPhiloxRandomKernel(write_accessor& data, random::PhiloxRandom& gen,
                         Distribution& dist)
      : data_(data), gen_(gen), dist_(dist) {}

  void operator()(sycl::nd_item<1> item) {
    using random::PhiloxRandom;
    using random::SingleSampleAdapter;

    const size_t kReservedSamplesPerOutput = 256;
    const size_t kGroupSize = Distribution::kResultElementCount;
    const size_t kGeneratorSkipPerOutputGroup =
        kGroupSize * kReservedSamplesPerOutput /
        PhiloxRandom::kResultElementCount;

    const size_t item_id = item.get_global(0);
    const size_t total_item_count = item.get_global_range();
    size_t group_index = item_id;
    size_t offset = group_index * kGroupSize;

    T* data = ConvertToActualTypeSycl(T, data_);
    const size_t size = data_.get_size() / sizeof(T);

    while (offset < size) {
      // Since each output takes a variable number of samples, we need to
      // realign the generator to the beginning for the current output group
      PhiloxRandom gen = gen_;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      const typename Distribution::ResultType samples = dist_(&single_samples);

      for (size_t i = 0; i < kGroupSize; ++i) {
        if (offset >= size) {
          return;
        }
        data[offset] = samples[i];
        ++offset;
      }

      offset += (total_item_count - 1) * kGroupSize;
      group_index += total_item_count;
    }
  }

 private:
  write_accessor data_;
  random::PhiloxRandom gen_;
  Distribution dist_;
};

template <typename T>
class FillRandomKernel;
// Partial specialization for SYCL to fill the entire region with randoms
// It splits the work into several tasks and run them in parallel
template <class Distribution>
void FillPhiloxRandom<SYCLDevice, Distribution>::operator()(
    OpKernelContext* context, const SYCLDevice& device,
    random::PhiloxRandom gen, typename Distribution::ResultElementType* data,
    int64 size, Distribution dist) {
  const size_t group_size = device.maxSyclThreadsPerBlock();
  const size_t group_count = (size + group_size - 1) / group_size;

  auto buffer = device.get_sycl_buffer(data);

  device.sycl_queue().submit([&](sycl::handler& cgh) {
    auto access = buffer.template get_access<sycl::access::mode::write>(cgh);

    FillPhiloxRandomKernel<Distribution,
                           Distribution::kVariableSamplesPerOutput>
        task(access, gen, dist);
    cgh.parallel_for<class FillRandomKernel<Distribution>>(
        sycl::nd_range<1>(sycl::range<1>(group_count * group_size),
                          sycl::range<1>(group_size)),
        task);
  });
}

}  // namespace functor

#endif  // TENSORFLOW_USE_SYCL

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RANDOM_OP_CPU_H_
