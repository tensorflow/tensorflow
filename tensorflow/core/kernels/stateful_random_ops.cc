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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>
#include <memory>

#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/kernels/resource_variable_ops.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

using random::PhiloxRandom;

namespace {

// 'Variable' doesn't support uint32 or uint64 yet (due to reasons explained
// in b/111604096 and cl/171681867), so I use signed int here. I choose int64
// instead of int32 because `VarHandleOp` doesn't support int32 on GPU.
using StateElementType = int64;
static constexpr DataType STATE_ELEMENT_DTYPE = DT_INT64;

using Algorithm = StateElementType;
static constexpr Algorithm RNG_ALG_PHILOX = 1;

using SkippableRNG = absl::variant<PhiloxRandom>;

// This function is for hiding the implementation detail about the
// `absl::variant` index of each algorithm.
Algorithm GetAlgorithm(SkippableRNG const& rng) {
  auto idx = rng.index();
  if (idx == 0) {
    return RNG_ALG_PHILOX;
  }
  // unreachable
  return RNG_ALG_PHILOX;
}

// Fills a buffer with random numbers sampled from a given distribution.
template <class Device, class Distribution>
Status FillRandom(OpKernelContext* ctx, const Device& device,
                  SkippableRNG const& gen, int64 size, Distribution dist,
                  typename Distribution::ResultElementType* data) {
  auto algorithm = GetAlgorithm(gen);
  if (algorithm == RNG_ALG_PHILOX) {
    auto philox = absl::get<PhiloxRandom>(gen);
    functor::FillPhiloxRandom<Device, Distribution>()(ctx, device, philox, data,
                                                      size, dist);
    return Status::OK();
  } else {
    // return errors::InvalidArgument("Unsupported algorithm id: ", algorithm);
    return Status::OK();
  }
}

// The following two functions use the contract "lower 32 bits for the first
// uint32, higher 32 bits for the second". Note that this is endian-neutral,
// unlike a direct memory copy `memcpy(output, &input, 8)`.
void Int64ToUint32s(int64 input, uint32* output1, uint32* output2) {
  auto u64 = static_cast<uint64>(input);
  *output1 = static_cast<uint32>(u64);
  *output2 = static_cast<uint32>(u64 >> 32);
}

int64 Uint32sToInt64(uint32 input1, uint32 input2) {
  auto u64_1 = static_cast<uint64>(input1);
  auto u64_2 = static_cast<uint64>(input2);
  return static_cast<int64>(u64_1 | (u64_2 << 32));
}

void GetPhiloxStateFromTensor(Tensor const& tensor,
                              PhiloxRandom::ResultType* counter,
                              PhiloxRandom::Key* key) {
  auto tensor_flat = tensor.flat<StateElementType>();
  auto tensor_ptr = tensor_flat.data();
  // tensor_ptr's index is added by 1 to skip the algorithm tag.
  Int64ToUint32s(tensor_ptr[1], &(*counter)[0], &(*counter)[1]);
  Int64ToUint32s(tensor_ptr[2], &(*counter)[2], &(*counter)[3]);
  Int64ToUint32s(tensor_ptr[3], &(*key)[0], &(*key)[1]);
}

void WritePhiloxStateToTensor(PhiloxRandom::ResultType const& counter,
                              PhiloxRandom::Key const& key, Tensor* tensor) {
  auto tensor_flat = tensor->flat<StateElementType>();
  auto tensor_ptr = tensor_flat.data();
  // tensor_ptr's index is added by 1 to skip the algorithm tag.
  tensor_ptr[1] = Uint32sToInt64(counter[0], counter[1]);
  tensor_ptr[2] = Uint32sToInt64(counter[2], counter[3]);
  tensor_ptr[3] = Uint32sToInt64(key[0], key[1]);
}

// A helper function that does the actual work for
// 'MakeRNGCopyAndUpdateVariable'.
template <typename Device>
Status GetRNGCopyAndUpdateTensor(Tensor* tensor, int64 delta,
                                 SkippableRNG* rng_copy);

template <>
Status GetRNGCopyAndUpdateTensor<CPUDevice>(Tensor* tensor, int64 delta,
                                            SkippableRNG* rng_copy) {
  // The dtype of `tensor` should be `StateElementType` and the first element
  // is the algorithm.
  if (tensor->dims() != 1) {
    return errors::InvalidArgument(
        "RNG state must have one and only one dimension, not ", tensor->dims());
  }
  auto tensor_flat = tensor->flat<StateElementType>();
  if (tensor_flat.size() < 1) {
    return errors::InvalidArgument("Size of tensor must be at least 1");
  }
  auto algorithm = tensor_flat.data()[0];
  if (algorithm == RNG_ALG_PHILOX) {
    // Delegates to PhiloxRandom to do the actual increasing.
    static_assert(std::is_same<StateElementType, int64>::value,
                  "StateElementType must be int64");
    static_assert(std::is_same<PhiloxRandom::ResultElementType, uint32>::value,
                  "PhiloxRandom::ResultElementType must be uint32");
    auto counter_size = PhiloxRandom::ResultType::kElementCount;
    auto key_size = PhiloxRandom::Key::kElementCount;
    auto min_tensor_size = 1 + (counter_size + key_size) / 2;
    if (tensor_flat.size() < min_tensor_size) {
      return errors::InvalidArgument(
          "For Philox algorithm, the size of state"
          " must be at least ",
          min_tensor_size, "; got ", tensor_flat.size());
    }
    PhiloxRandom::ResultType counter;
    PhiloxRandom::Key key;
    GetPhiloxStateFromTensor(*tensor, &counter, &key);
    PhiloxRandom philox(counter, key);
    auto old_philox = philox;
    philox.Skip(delta);  // do the actual increasing
    WritePhiloxStateToTensor(philox.counter(), philox.key(), tensor);
    *rng_copy = SkippableRNG(old_philox);
    return Status::OK();
  } else {
    // return errors::InvalidArgument("Unsupported algorithm id: ", algorithm);
    *rng_copy = SkippableRNG(PhiloxRandom());
    return Status::OK();
  }
}

// Gets a copy of the RNG and updates the variable. The copy can be used to
// generate upto 'samples' random numbers, and the variable is updated as if
// 'samples' random numbers have been generated (e.g. if the variable is a
// counnter, the counter is increased by 'samples').
template <class Device>
Status MakeRNGCopyAndUpdateVariable(OpKernelContext* ctx, int input_idx,
                                    int64 samples, SkippableRNG* rng_copy) {
  Var* var = nullptr;
  TF_RETURN_IF_ERROR(
      LookupResource(ctx, HandleFromInput(ctx, input_idx), &var));
  core::ScopedUnref s(var);
  mutex_lock ml(*var->mu());
  Tensor* var_tensor = var->tensor();
  if (var_tensor->dtype() != STATE_ELEMENT_DTYPE) {
    return errors::InvalidArgument("dtype of RNG state variable must be ",
                                   DataTypeString(STATE_ELEMENT_DTYPE),
                                   ", not ",
                                   DataTypeString(var_tensor->dtype()));
  }
  TF_RETURN_IF_ERROR(PrepareToUpdateVariable<Device, StateElementType>(
      ctx, var_tensor, var->copy_on_read_mode.load()));
  TF_RETURN_IF_ERROR(
      GetRNGCopyAndUpdateTensor<Device>(var_tensor, samples, rng_copy));
  return Status::OK();
}

template <typename Device, class Distribution>
class StatefulRandomOp : public OpKernel {
 public:
  using T = typename Distribution::ResultElementType;
  explicit StatefulRandomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Assumes that input(0) is an existing resource.
  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(1);
    Tensor* output;
    TensorShape shape;
    OP_REQUIRES_OK(ctx, MakeShape(shape_t, &shape));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) return;

    auto output_flat = output->flat<T>();
    SkippableRNG rng;
    // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
    // it just here.
    OP_REQUIRES_OK(ctx, MakeRNGCopyAndUpdateVariable<Device>(
                            ctx, 0, output_flat.size() * 256, &rng));
    // Fill in the random numbers
    OP_REQUIRES_OK(ctx, FillRandom(ctx, ctx->eigen_device<Device>(), rng,
                                   output_flat.size(), Distribution(),
                                   output_flat.data()));
  }
};

}  // namespace

// So far the 'Distribution' type parameter is only used when the algorithm is
// philox, so 'NormalDistribution<PhiloxRandom, ...>' is fine for now.
#define REGISTER(DEVICE, TYPE)            \
  REGISTER_KERNEL_BUILDER(                \
      Name("StatefulStandardNormal")      \
          .Device(DEVICE_##DEVICE)        \
          .HostMemory("resource")         \
          .HostMemory("shape")            \
          .TypeConstraint<TYPE>("dtype"), \
      StatefulRandomOp<DEVICE##Device,    \
                       random::NormalDistribution<PhiloxRandom, TYPE> >);

#define REGISTER_CPU(TYPE) REGISTER(CPU, TYPE)

TF_CALL_half(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU
#undef REGISTER

// TODO(wangpeng): Add RNG ops for other distributions.
// TODO(wangpeng): Add support for GPU and XLA.

}  // end namespace tensorflow
