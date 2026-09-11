/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"

#import <Metal/Metal.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_shader_library.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// The parameterised random ops: a truncated normal with per-batch bounds,
// categorical sampling, and the gamma distribution, each in both a seeded and
// a stateless form.
//
// Like the plain generators, these are shaders over a counter-based Philox
// stream rather than MPSGraph random ops, for the same reason: this backend
// caches graphs by shape, and a seed baked into a cached graph would make
// every call after the first return the same numbers.
//
// The stateless forms take their seed as a tensor and use it directly, so the
// same seed gives the same output, which is the whole point of them. The
// seeded forms bump a per-kernel counter on each call.

struct DistOp {
  uint32_t seed_lo = 0;
  uint32_t seed_hi = 0;
  bool stateless = false;
  TF_DataType output_dtype = TF_INT64;
  std::atomic<uint32_t> counter{0};
};

void* DistOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new DistOp();

  int64_t seed = 0;
  int64_t seed2 = 0;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "seed", &seed, status);
  const bool has_seed_attr = TF_GetCode(status) == TF_OK;
  TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrInt64(ctx, "seed2", &seed2, status);
  TF_SetStatus(status, TF_OK, "");

  // No seed attribute at all means a stateless op, whose seed arrives as an
  // input instead.
  op->stateless = !has_seed_attr;
  if (seed == 0 && seed2 == 0) {
    // Zero means "pick something". A fixed fallback would make every unseeded
    // op in a process draw identically, so the kernel's address stands in for
    // the missing entropy.
    const uintptr_t address = reinterpret_cast<uintptr_t>(op);
    seed = static_cast<int64_t>(address);
    seed2 = static_cast<int64_t>(address >> 32);
  }
  op->seed_lo = static_cast<uint32_t>(seed);
  op->seed_hi = static_cast<uint32_t>(seed2);

  TF_DataType out_dtype = TF_INT64;
  TF_OpKernelConstruction_GetAttrType(ctx, "output_dtype", &out_dtype, status);
  if (TF_GetCode(status) == TF_OK) op->output_dtype = out_dtype;
  TF_SetStatus(status, TF_OK, "");

  TF_DeleteStatus(status);
  return op;
}

void DistOp_Delete(void* kernel) { delete static_cast<DistOp*>(kernel); }

// Reads a shape or size argument that lives in host memory.
bool ReadIndexVector(TF_Tensor* t, std::vector<int64_t>* out,
                     TF_Status* status) {
  const int64_t count = TF_TensorElementCount(t);
  const TF_DataType dtype = TF_TensorType(t);
  const void* data = TF_TensorData(t);
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a host-memory argument has no data.");
    return false;
  }
  out->clear();
  for (int64_t i = 0; i < count; ++i) {
    if (dtype == TF_INT32) {
      out->push_back(static_cast<const int32_t*>(data)[i]);
    } else if (dtype == TF_INT64) {
      out->push_back(static_cast<const int64_t*>(data)[i]);
    } else if (dtype == TF_UINT32) {
      out->push_back(static_cast<const uint32_t*>(data)[i]);
    } else if (dtype == TF_UINT64) {
      // The V3 stateless generators take their key and counter as uint64,
      // which this refused, so every StatelessRandom*V3 op failed on a type
      // its own op def requires.
      out->push_back(
          static_cast<int64_t>(static_cast<const uint64_t*>(data)[i]));
    } else {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: expected an integer argument.");
      return false;
    }
  }
  return true;
}

// The seed the shader should use, and the counter that goes with it.
//
// A stateless op must be a pure function of its seed input, so its counter is
// fixed; a seeded op advances a counter so that repeated calls differ.
struct SeedAndCounter {
  uint32_t lo = 0;
  uint32_t hi = 0;
  uint32_t counter = 0;
};

bool ResolveSeed(DistOp* op, TF_OpKernelContext* ctx, int seed_index,
                 SeedAndCounter* out, TF_Status* status) {
  if (!op->stateless || seed_index < 0) {
    out->lo = op->seed_lo;
    out->hi = op->seed_hi;
    out->counter = op->counter.fetch_add(1) + 1;
    return true;
  }
  ScopedTensor seed;
  TF_GetInput(ctx, seed_index, seed.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  std::vector<int64_t> values;
  if (!ReadIndexVector(seed.get(), &values, status)) return false;
  if (values.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a stateless seed must have at least one entry.");
    return false;
  }
  if (values.size() == 1) {
    // The V3 generators take a single 64-bit key where the V2 ones take a
    // pair of 32-bit seeds. Splitting the key in half is the same quantity of
    // seed material arriving in one input instead of two; requiring two
    // entries rejected every V3 op outright.
    const uint64_t key = static_cast<uint64_t>(values[0]);
    out->lo = static_cast<uint32_t>(key & 0xFFFFFFFFu);
    out->hi = static_cast<uint32_t>(key >> 32);
  } else {
    out->lo = static_cast<uint32_t>(values[0]);
    out->hi = static_cast<uint32_t>(values[1]);
  }
  out->counter = 0;
  return true;
}

// Runs one of the shaders with a parameter block and a list of buffers.
bool Dispatch(TF_OpKernelContext* ctx, SP_Stream stream, const char* function,
              const std::vector<BufferSlice>& buffers, const void* params,
              size_t params_size, uint32_t count, TF_Status* status) {
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), function, status);
  if (pipeline == nil) return false;
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a random op.");
    return false;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  NSUInteger index = 0;
  for (const BufferSlice& slice : buffers) {
    [encoder setBuffer:slice.buffer offset:slice.offset atIndex:index];
    ++index;
  }
  [encoder setBytes:params length:params_size atIndex:index];
  Dispatch1D(encoder, pipeline, count);
  [encoder endEncoding];
  command_buffer.Commit();
  return true;
}

/*** PARAMETERISED TRUNCATED NORMAL ***/

// `seed_index` is the stateless seed input, and `first_param` the index of
// the means, which the other three range vectors follow.
void ParamTruncated_ComputeImpl(DistOp* op, TF_OpKernelContext* ctx,
                                int seed_index, int first_param,
                                TF_Status* status) {
  ScopedTensor shape_tensor;
  TF_GetInput(ctx, 0, shape_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  std::vector<int64_t> shape;
  if (!ReadIndexVector(shape_tensor.get(), &shape, status)) return;
  if (shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ParameterizedTruncatedNormal needs a rank of at "
                 "least one.");
    return;
  }

  ScopedTensor ranges[4];
  for (int i = 0; i < 4; ++i) {
    TF_GetInput(ctx, first_param + i, ranges[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  int64_t count = 1;
  for (int64_t d : shape) count *= d;
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, shape.data(),
                                 static_cast<int>(shape.size()),
                                 static_cast<size_t>(count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  const int64_t batch = shape[0];
  const int64_t num_params = NumElements(ranges[0].get());
  if (num_params != 1 && num_params != batch) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the truncated normal parameters must be scalars or "
                 "one per batch entry.");
    return;
  }

  SeedAndCounter seed;
  if (!ResolveSeed(op, ctx, seed_index, &seed, status)) return;
  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<BufferSlice> buffers(5);
  if (!SliceForTensor(output.get(), &buffers[0], status)) return;
  for (int i = 0; i < 4; ++i) {
    if (!SliceForTensor(ranges[i].get(), &buffers[i + 1], status)) return;
  }

  ParamTruncatedParams params;
  params.count = static_cast<uint32_t>(count);
  params.seed_lo = seed.lo;
  params.seed_hi = seed.hi;
  params.counter = seed.counter;
  params.samples_per_batch =
      static_cast<uint32_t>(batch > 0 ? count / batch : count);
  params.num_params = static_cast<uint32_t>(num_params);
  params.padding0 = 0;
  params.padding1 = 0;
  Dispatch(ctx, stream, "tf_parameterized_truncated_normal_float", buffers,
           &params, sizeof(params), params.count, status);
}

/*** MULTINOMIAL ***/

void Multinomial_ComputeImpl(DistOp* op, TF_OpKernelContext* ctx,
                             int seed_index, TF_Status* status) {
  ScopedTensor logits, num_samples;
  TF_GetInput(ctx, 0, logits.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, num_samples.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> logits_shape = ShapeOf(logits.get());
  if (logits_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Multinomial expects logits of shape [batch, "
                 "classes].");
    return;
  }
  std::vector<int64_t> samples;
  if (!ReadIndexVector(num_samples.get(), &samples, status)) return;
  if (samples.empty() || samples[0] < 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: num_samples must be a non-negative scalar in host "
                 "memory.");
    return;
  }

  const std::vector<int64_t> out_shape = {logits_shape[0], samples[0]};
  const int64_t count = out_shape[0] * out_shape[1];
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->output_dtype, out_shape.data(), 2,
      static_cast<size_t>(count) * TF_DataTypeSize(op->output_dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SeedAndCounter seed;
  if (!ResolveSeed(op, ctx, seed_index, &seed, status)) return;
  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<BufferSlice> buffers(2);
  if (!SliceForTensor(logits.get(), &buffers[0], status)) return;
  if (!SliceForTensor(output.get(), &buffers[1], status)) return;

  MultinomialParams params;
  params.count = static_cast<uint32_t>(count);
  params.seed_lo = seed.lo;
  params.seed_hi = seed.hi;
  params.counter = seed.counter;
  params.batch = static_cast<uint32_t>(out_shape[0]);
  params.classes = static_cast<uint32_t>(logits_shape[1]);
  params.samples = static_cast<uint32_t>(out_shape[1]);
  params.padding0 = 0;
  Dispatch(ctx, stream,
           op->output_dtype == TF_INT32 ? "tf_multinomial_float"
                                        : "tf_multinomial_float_i64",
           buffers, &params, sizeof(params), params.count, status);
}

/*** GAMMA ***/

// `alpha_index` is where the shape parameter sits, which differs between the
// three forms.
void Gamma_ComputeImpl(DistOp* op, TF_OpKernelContext* ctx, int seed_index,
                       int alpha_index, TF_Status* status) {
  ScopedTensor shape_tensor, alpha;
  TF_GetInput(ctx, 0, shape_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, alpha_index, alpha.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> shape;
  if (!ReadIndexVector(shape_tensor.get(), &shape, status)) return;
  // The output is the requested shape followed by alpha's own shape, with
  // alpha varying fastest, which is what makes the index arithmetic in the
  // shader a single remainder.
  const std::vector<int64_t> alpha_shape = ShapeOf(alpha.get());
  std::vector<int64_t> out_shape = shape;
  for (int64_t d : alpha_shape) out_shape.push_back(d);

  int64_t count = 1;
  for (int64_t d : out_shape) count *= d;
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, out_shape.data(),
                                 static_cast<int>(out_shape.size()),
                                 static_cast<size_t>(count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SeedAndCounter seed;
  if (!ResolveSeed(op, ctx, seed_index, &seed, status)) return;
  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<BufferSlice> buffers(2);
  if (!SliceForTensor(output.get(), &buffers[0], status)) return;
  if (!SliceForTensor(alpha.get(), &buffers[1], status)) return;

  GammaParams params;
  params.count = static_cast<uint32_t>(count);
  params.seed_lo = seed.lo;
  params.seed_hi = seed.hi;
  params.counter = seed.counter;
  params.num_alphas = static_cast<uint32_t>(NumElements(alpha.get()));
  params.padding0 = 0;
  params.padding1 = 0;
  params.padding2 = 0;
  Dispatch(ctx, stream, "tf_random_gamma_float", buffers, &params,
           sizeof(params), params.count, status);
}

#define METAL_DIST_COMPUTE(NAME, BODY)                                      \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<DistOp*>(kernel);                                \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a random kernel has no state.");                 \
    } else {                                                                \
      BODY;                                                                 \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_DIST_COMPUTE(ParamTruncated_Compute,
                   ParamTruncated_ComputeImpl(op, ctx, -1, 1, status))
METAL_DIST_COMPUTE(StatelessParamTruncated_Compute,
                   ParamTruncated_ComputeImpl(op, ctx, 1, 2, status))
METAL_DIST_COMPUTE(Multinomial_Compute,
                   Multinomial_ComputeImpl(op, ctx, -1, status))
METAL_DIST_COMPUTE(StatelessMultinomial_Compute,
                   Multinomial_ComputeImpl(op, ctx, 2, status))
METAL_DIST_COMPUTE(Gamma_Compute, Gamma_ComputeImpl(op, ctx, -1, 1, status))
METAL_DIST_COMPUTE(StatelessGammaV2_Compute,
                   Gamma_ComputeImpl(op, ctx, 1, 2, status))
// V3 replaces the seed pair with a key, a counter and an algorithm; the key
// is the seed and the counter input is folded in through it.
METAL_DIST_COMPUTE(StatelessGammaV3_Compute,
                   Gamma_ComputeImpl(op, ctx, 1, 4, status))

#undef METAL_DIST_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name,
              const std::vector<const char*>& host_inputs,
              const char* type_attr, TF_DataType type) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &DistOp_Create, compute, &DistOp_Delete);
  if (type_attr != nullptr) {
    TF_KernelBuilder_TypeConstraint(builder, type_attr, type, status);
  }
  for (const char* input : host_inputs) {
    TF_KernelBuilder_HostMemory(builder, input);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(name.c_str(), builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalRandomDistKernels() {
  // Shapes, sizes and stateless seeds are all read on the host: the first two
  // size the output and the third is two scalars the shader takes by value.
  Register("ParameterizedTruncatedNormal", &ParamTruncated_Compute,
           "MetalParameterizedTruncatedNormal", {"shape"}, "dtype", TF_FLOAT);
  Register("StatelessParameterizedTruncatedNormal",
           &StatelessParamTruncated_Compute,
           "MetalStatelessParameterizedTruncatedNormal", {"shape", "seed"},
           "dtype", TF_FLOAT);

  static constexpr TF_DataType kOutputs[] = {TF_INT32, TF_INT64};
  static constexpr const char* kSuffixes[] = {"Int32", "Int64"};
  for (int i = 0; i < 2; ++i) {
    Register("Multinomial", &Multinomial_Compute,
             std::string("MetalMultinomial") + kSuffixes[i], {"num_samples"},
             "output_dtype", kOutputs[i]);
    Register("StatelessMultinomial", &StatelessMultinomial_Compute,
             std::string("MetalStatelessMultinomial") + kSuffixes[i],
             {"num_samples", "seed"}, "output_dtype", kOutputs[i]);
  }

  Register("RandomGamma", &Gamma_Compute, "MetalRandomGamma", {"shape"}, "T",
           TF_FLOAT);
  Register("StatelessRandomGammaV2", &StatelessGammaV2_Compute,
           "MetalStatelessRandomGammaV2", {"shape", "seed"}, "dtype",
           TF_FLOAT);
  Register("StatelessRandomGammaV3", &StatelessGammaV3_Compute,
           "MetalStatelessRandomGammaV3", {"shape", "key", "counter", "alg"},
           "dtype", TF_FLOAT);
}

}  // namespace metal
}  // namespace tensorflow
