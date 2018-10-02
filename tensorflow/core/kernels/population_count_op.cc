/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc

#define EIGEN_USE_THREADS

#include <bitset>

#include "tensorflow/core/kernels/population_count_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class PopulationCountOp : public OpKernel {
 public:
  explicit PopulationCountOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& input_t = c->input(0);
    Tensor* output_t;
    OP_REQUIRES_OK(c, c->allocate_output(0, input_t.shape(), &output_t));

    auto input = input_t.flat<T>();
    auto output = output_t->flat<uint8>();

    functor::PopulationCount<Device, T> popcnt;
    popcnt(c, input, output);
  }
};

#define REGISTER_POPULATION_COUNT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("PopulationCount").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      PopulationCountOp<CPUDevice, type>);

TF_CALL_uint8(REGISTER_POPULATION_COUNT);
TF_CALL_int8(REGISTER_POPULATION_COUNT);
TF_CALL_uint16(REGISTER_POPULATION_COUNT);
TF_CALL_int16(REGISTER_POPULATION_COUNT);
TF_CALL_int32(REGISTER_POPULATION_COUNT);
TF_CALL_int64(REGISTER_POPULATION_COUNT);

#undef REGISTER_POPULATION_COUNT

namespace functor {

namespace {

template <typename T>
inline uint8 PopCnt(const T v);

#define POPCNT(T, N)                  \
  template <>                         \
  uint8 PopCnt<T>(const T v) {        \
    return std::bitset<N>(v).count(); \
  }

POPCNT(int8, 8);
POPCNT(uint8, 8);
POPCNT(int16, 16);
POPCNT(uint16, 16);
POPCNT(int32, 32);
POPCNT(int64, 64);

#undef POPCNT

}  // namespace

template <typename T>
struct PopulationCount<CPUDevice, T> {
  void operator()(OpKernelContext* c, typename TTypes<T>::ConstFlat input,
                  TTypes<uint8>::Flat output) {
    const T* input_ptr = input.data();
    uint8* output_ptr = output.data();
    auto shard = [input_ptr, output_ptr](int64 start, int64 limit) {
      for (int64 i = start; i < limit; ++i) {
        output_ptr[i] = PopCnt<T>(input_ptr[i]);
      }
    };
    int64 total_shards = input.size();
    // Approximating cost of popcnt: convert T to int64
    // (std::bitset constructor) and convert int64 to uint8
    // (bitset.count() -> output).  The .count() itself is relatively cheap.
    const double total_cost = (Eigen::TensorOpCost::CastCost<T, uint8>() +
                               Eigen::TensorOpCost::CastCost<int64, uint8>());
    const int64 shard_cost = (total_cost >= static_cast<double>(kint64max))
                                 ? kint64max
                                 : static_cast<int64>(total_cost);

    auto worker_threads = *(c->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, total_shards,
          shard_cost, shard);
  }
};

}  // namespace functor

#if GOOGLE_CUDA

#define REGISTER_POPULATION_COUNT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("PopulationCount").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PopulationCountOp<GPUDevice, type>)

TF_CALL_uint8(REGISTER_POPULATION_COUNT);
TF_CALL_int8(REGISTER_POPULATION_COUNT);
TF_CALL_uint16(REGISTER_POPULATION_COUNT);
TF_CALL_int16(REGISTER_POPULATION_COUNT);
TF_CALL_int32(REGISTER_POPULATION_COUNT);
TF_CALL_int64(REGISTER_POPULATION_COUNT);

#undef REGISTER_POPULATION_COUNT

namespace functor {

#define DECLARE_GPU_SPEC(T)                                    \
  template <>                                                  \
  void PopulationCount<GPUDevice, T>::operator()(              \
      OpKernelContext* c, typename TTypes<T>::ConstFlat input, \
      TTypes<uint8>::Flat output);                             \
  extern template struct PopulationCount<GPUDevice, T>

TF_CALL_uint8(DECLARE_GPU_SPEC);
TF_CALL_int8(DECLARE_GPU_SPEC);
TF_CALL_uint16(DECLARE_GPU_SPEC);
TF_CALL_int16(DECLARE_GPU_SPEC);
TF_CALL_int32(DECLARE_GPU_SPEC);
TF_CALL_int64(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // namespace functor

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
