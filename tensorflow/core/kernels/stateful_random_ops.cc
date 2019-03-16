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

#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

template <typename Distribution>
struct UpdateVariableAndFill_Philox<CPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const CPUDevice& device,
                  int64 output_size, int64 alg_tag_skip,
                  ScopedUnlockUnrefVar* state_var_guard, Tensor* state_tensor,
                  typename Distribution::ResultElementType* output_data) {
    auto state_tensor_flat = state_tensor->flat<StateElementType>();
    auto state_data = state_tensor_flat.data();
    // Delegates to PhiloxRandom to do the actual increasing.
    auto philox = GetPhiloxRandomFromMem(state_data + alg_tag_skip);
    UpdateMemWithPhiloxRandom(philox, output_size, state_data + alg_tag_skip);
    // No longer needs the lock.
    state_var_guard->Release();
    functor::FillPhiloxRandom<CPUDevice, Distribution>()(
        ctx, device, philox, output_data, output_size, Distribution());
  }
};

template <typename Device, typename Distribution>
Status UpdateVariableAndFill(
    OpKernelContext* ctx, int state_input_idx, bool read_alg_from_state,
    Algorithm alg, int64 output_size,
    typename Distribution::ResultElementType* output_data) {
  Var* var = nullptr;
  TF_RETURN_IF_ERROR(
      LookupResource(ctx, HandleFromInput(ctx, state_input_idx), &var));
  // Use `ScopedUnlockUnrefVar` here instead of `mutex_lock` and `ScopedUnref`
  // because the former supports early releasing which is needed by
  // `UpdateVariableAndFill_Philox<CPU>` to avoid holding the lock while
  // filling.
  ScopedUnlockUnrefVar state_var_guard(var);
  Tensor* var_tensor = var->tensor();
  if (var_tensor->dtype() != STATE_ELEMENT_DTYPE) {
    return errors::InvalidArgument("dtype of RNG state variable must be ",
                                   DataTypeString(STATE_ELEMENT_DTYPE),
                                   ", not ",
                                   DataTypeString(var_tensor->dtype()));
  }
  if (var_tensor->dims() != 1) {
    return errors::InvalidArgument(
        "RNG state must have one and only one dimension, not ",
        var_tensor->dims());
  }
  auto var_tensor_flat = var_tensor->flat<StateElementType>();
  int64 alg_tag_skip = 0;
  if (read_alg_from_state) {
    alg_tag_skip = 1;
    if (var_tensor_flat.size() < 1) {
      return errors::InvalidArgument("Size of tensor must be at least 1");
    }
    alg = var_tensor_flat(0);
  }
  if (alg == RNG_ALG_PHILOX) {
    static_assert(std::is_same<StateElementType, int64>::value,
                  "StateElementType must be int64");
    static_assert(std::is_same<PhiloxRandom::ResultElementType, uint32>::value,
                  "PhiloxRandom::ResultElementType must be uint32");
    if (var_tensor_flat.size() < alg_tag_skip + PHILOX_MIN_STATE_SIZE) {
      return errors::InvalidArgument(
          "For the Philox algorithm, the size of state"
          " must be at least ",
          alg_tag_skip + PHILOX_MIN_STATE_SIZE, "; got ",
          var_tensor_flat.size());
    }
    TF_RETURN_IF_ERROR(PrepareToUpdateVariable<Device, StateElementType>(
        ctx, var_tensor, var->copy_on_read_mode.load()));
    UpdateVariableAndFill_Philox<Device, Distribution>()(
        ctx, ctx->eigen_device<Device>(), output_size, alg_tag_skip,
        &state_var_guard, var_tensor, output_data);
    return Status::OK();
  } else {
    return errors::InvalidArgument("Unsupported algorithm id: ", alg);
  }
}

// Preconditon: input(0) is an existing resource.
template <typename Device, class Distribution>
void ComputeImpl(OpKernelContext* ctx, int state_input_idx, int shape_input_idx,
                 bool read_alg_from_state, Algorithm alg) {
  using T = typename Distribution::ResultElementType;
  const Tensor& shape_t = ctx->input(shape_input_idx);
  TensorShape shape;
  OP_REQUIRES_OK(ctx, ctx->op_kernel().MakeShape(shape_t, &shape));
  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
  auto output_flat = output->flat<T>();
  OP_REQUIRES_OK(ctx, UpdateVariableAndFill<Device, Distribution>(
                          ctx, state_input_idx, read_alg_from_state, alg,
                          output_flat.size(), output_flat.data()));
}

template <typename Device, class Distribution>
class StatefulRandomOp : public OpKernel {
 public:
  explicit StatefulRandomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    ComputeImpl<Device, Distribution>(ctx, 0, 1, true, 0);
  }
};

template <typename Device, class Distribution>
class StatefulRandomOpV2 : public OpKernel {
 public:
  explicit StatefulRandomOpV2(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& alg_tensor = ctx->input(1);
    OP_REQUIRES(ctx, alg_tensor.dims() == 0,
                errors::InvalidArgument("algorithm must be of shape [], not ",
                                        alg_tensor.shape().DebugString()));
    OP_REQUIRES(
        ctx, alg_tensor.dtype() == ALGORITHM_DTYPE,
        errors::InvalidArgument("algorithm's dtype must be ",
                                DataTypeString(ALGORITHM_DTYPE), ", not ",
                                DataTypeString(alg_tensor.dtype())));
    auto alg = alg_tensor.flat<Algorithm>()(0);
    ComputeImpl<Device, Distribution>(ctx, 0, 2, false, alg);
  }
};

template <typename T>
class NonDeterministicIntsOp : public OpKernel {
 public:
  explicit NonDeterministicIntsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->op_kernel().MakeShape(shape_t, &shape));
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) return;

    switch (dtype_) {
      case DT_INT32:
      case DT_UINT32:
      case DT_INT64:
      case DT_UINT64: {
        auto output_flat = output->flat<T>();
        auto data = output_flat.data();
        for (int64 i = 0; i < output_flat.size(); ++i) {
          data[i] = static_cast<T>(random::New64());
        }
        break;
      }
      default:
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument("Unsupported dtype: ",
                                            DataTypeString(dtype_)));
    }
  }

 private:
  DataType dtype_;
};

// So far the 'Distribution' type parameter is only used when the algorithm is
// philox, so 'NormalDistribution<PhiloxRandom, ...>' is fine for now.
#define REGISTER(DEVICE, TYPE)            \
  REGISTER_KERNEL_BUILDER(                \
      Name("StatefulStandardNormalV2")    \
          .Device(DEVICE_##DEVICE)        \
          .HostMemory("resource")         \
          .HostMemory("algorithm")        \
          .HostMemory("shape")            \
          .TypeConstraint<TYPE>("dtype"), \
      StatefulRandomOpV2<DEVICE##Device,  \
                         random::NormalDistribution<PhiloxRandom, TYPE> >);

// CPU also has the old 'StatefulStandardNormal' op for backward compatibility.
#define REGISTER_CPU(TYPE)                \
  REGISTER(CPU, TYPE)                     \
  REGISTER_KERNEL_BUILDER(                \
      Name("StatefulStandardNormal")      \
          .Device(DEVICE_CPU)             \
          .HostMemory("resource")         \
          .HostMemory("shape")            \
          .TypeConstraint<TYPE>("dtype"), \
      StatefulRandomOp<CPUDevice,         \
                       random::NormalDistribution<PhiloxRandom, TYPE> >);

#define REGISTER_GPU(TYPE) REGISTER(GPU, TYPE)

TF_CALL_half(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#if GOOGLE_CUDA

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);

#endif  // GOOGLE_CUDA

#undef REGISTER_GPU
#undef REGISTER_CPU
#undef REGISTER

#define REGISTER_NonDeterministicInts(TYPE)                   \
  REGISTER_KERNEL_BUILDER(Name("NonDeterministicInts")        \
                              .Device(DEVICE_CPU)             \
                              .HostMemory("shape")            \
                              .TypeConstraint<TYPE>("dtype"), \
                          NonDeterministicIntsOp<TYPE>);

TF_CALL_int32(REGISTER_NonDeterministicInts);
TF_CALL_uint32(REGISTER_NonDeterministicInts);
TF_CALL_int64(REGISTER_NonDeterministicInts);
TF_CALL_uint64(REGISTER_NonDeterministicInts);

#undef REGISTER_NonDeterministicInts

// TODO(wangpeng): Add RNG ops for other distributions.

}  // end namespace tensorflow
