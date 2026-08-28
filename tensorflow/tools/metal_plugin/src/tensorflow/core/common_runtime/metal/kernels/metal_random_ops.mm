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
#include "absl/synchronization/mutex.h"
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

// RandomUniform, RandomStandardNormal and TruncatedNormal, which is what
// initialises a model's weights before the first step.
//
// These are compute shaders rather than MPSGraph. MPSGraph's random ops take
// the seed either as a value baked into the graph or as a state tensor threaded
// between calls, and this backend caches graphs by shape: a baked seed would
// make every call after the first return the identical tensor, and every
// variable in a model would be initialised to the same numbers. A
// counter-based generator sidesteps the problem entirely, because the counter
// is a per-call input rather than part of the graph.

enum class Distribution { kUniform, kNormal, kTruncatedNormal };

struct RandomOp {
  TF_DataType dtype = TF_FLOAT;
  uint32_t seed_lo = 0;
  uint32_t seed_hi = 0;
  // Bumped on every Compute so repeated calls draw different numbers. Atomic
  // because one kernel object serves every thread running that node.
  std::atomic<uint32_t> counter{0};
};

void* RandomOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new RandomOp();
  // Most of the generators name their output type "dtype"; RandomUniformInt
  // names it "Tout". Reading only "dtype" made that op fail on every call
  // with "No attr named 'dtype' in NodeDef".
  TF_OpKernelConstruction_GetAttrType(ctx, "dtype", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    TF_OpKernelConstruction_GetAttrType(ctx, "Tout", &op->dtype, status);
  }
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }

  // seed and seed2 are both optional and both default to 0, which TensorFlow
  // treats as "pick something". A fixed fallback would make every unseeded
  // model in a process start from identical weights, so the address of this
  // kernel object stands in for the missing entropy.
  int64_t seed = 0;
  int64_t seed2 = 0;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "seed", &seed, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrInt64(ctx, "seed2", &seed2, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");

  if (seed == 0 && seed2 == 0) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(op);
    seed = static_cast<int64_t>(address);
    seed2 = static_cast<int64_t>(address >> 32);
  }
  op->seed_lo = static_cast<uint32_t>(seed);
  op->seed_hi = static_cast<uint32_t>(seed2);

  TF_DeleteStatus(status);
  return op;
}

void RandomOp_Delete(void* kernel) { delete static_cast<RandomOp*>(kernel); }

const char* ShaderFor(Distribution distribution) {
  switch (distribution) {
    case Distribution::kUniform:
      return "tf_random_uniform_float";
    case Distribution::kNormal:
      return "tf_random_normal_float";
    case Distribution::kTruncatedNormal:
      return "tf_truncated_normal_float";
  }
  return nullptr;
}

template <Distribution kDistribution>
void Random_ComputeImpl(RandomOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  ScopedTensor shape_tensor;
  TF_GetInput(ctx, 0, shape_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  // The shape arrives in host memory, so it can be read without draining.
  const int64_t rank = TF_TensorElementCount(shape_tensor.get());
  const TF_DataType index_dtype = TF_TensorType(shape_tensor.get());
  const void* shape_data = TF_TensorData(shape_tensor.get());
  std::vector<int64_t> shape;
  shape.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    if (index_dtype == TF_INT32) {
      shape.push_back(static_cast<const int32_t*>(shape_data)[i]);
    } else {
      shape.push_back(static_cast<const int64_t*>(shape_data)[i]);
    }
  }

  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  if (op->dtype != TF_FLOAT) {
    // The generators are written for float32 only. Half-precision weights are
    // normally initialised in float32 and cast, so this has not been needed.
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: random ops are implemented for float32 only.");
    return;
  }

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), ShaderFor(kDistribution), status);
  if (pipeline == nil) return;

  BufferSlice out_slice;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a random op.");
    return;
  }

  RandomParams params;
  params.count = static_cast<uint32_t>(count);
  params.seed_lo = op->seed_lo;
  params.seed_hi = op->seed_hi;
  params.counter = op->counter.fetch_add(1, std::memory_order_relaxed);

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:0];
  [encoder setBytes:&params length:sizeof(params) atIndex:1];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

// RandomUniformInt draws in [minval, maxval), both device scalars. Unlike the
// float generators the bounds have to be known on the host to build the
// modulus, so this one does drain the stream. Weight initialisation does not
// use it; it appears in data pipelines and shuffles, where a drain per call is
// acceptable and a wrong distribution would not be.
void RandomUniformInt_ComputeImpl(RandomOp* op, TF_OpKernelContext* ctx,
                                  TF_Status* status) {
  ScopedTensor shape_tensor, lo_t, hi_t;
  TF_GetInput(ctx, 0, shape_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, lo_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, hi_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const int64_t rank = TF_TensorElementCount(shape_tensor.get());
  const TF_DataType index_dtype = TF_TensorType(shape_tensor.get());
  const void* shape_data = TF_TensorData(shape_tensor.get());
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < rank; ++i) {
    shape.push_back(index_dtype == TF_INT32
                        ? static_cast<const int32_t*>(shape_data)[i]
                        : static_cast<const int64_t*>(shape_data)[i]);
  }
  int64_t count = 1;
  for (int64_t d : shape) count *= d;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_INT32, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_INT32), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  // The bounds live on the device; the modulus needs them on the host.
  uint64_t target = 0;
  {
    absl::MutexLock lock(&stream->mu);
    target = stream->last_enqueued;
  }
  if (target > 0) {
    [stream->order_event waitUntilSignaledValue:target timeoutMS:UINT64_MAX];
  }
  const int32_t* lo_p = static_cast<const int32_t*>(TF_TensorData(lo_t.get()));
  const int32_t* hi_p = static_cast<const int32_t*>(TF_TensorData(hi_t.get()));
  if (lo_p == nullptr || hi_p == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: RandomUniformInt bounds have no data.");
    return;
  }
  if (*hi_p <= *lo_p) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: RandomUniformInt needs maxval greater than minval.");
    return;
  }

  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), "tf_random_uniform_int", status);
  if (pipeline == nil) return;

  BufferSlice out_slice;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a random op.");
    return;
  }

  RandomIntParams params;
  params.count = static_cast<uint32_t>(count);
  params.seed_lo = op->seed_lo;
  params.seed_hi = op->seed_hi;
  params.counter = op->counter.fetch_add(1, std::memory_order_relaxed);
  params.lo = *lo_p;
  params.span = static_cast<uint32_t>(*hi_p - *lo_p);
  params.padding0 = 0;
  params.padding1 = 0;

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:0];
  [encoder setBytes:&params length:sizeof(params) atIndex:1];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

void RandomUniformInt_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<RandomOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: RandomUniformInt kernel has no state.");
  } else {
    RandomUniformInt_ComputeImpl(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

#define METAL_DEFINE_RANDOM_COMPUTE(NAME, DIST)                               \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<RandomOp*>(kernel);                                \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL,                                       \
                   "Metal: random kernel has no state.");                     \
    } else {                                                                  \
      Random_ComputeImpl<DIST>(op, ctx, status);                              \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_DEFINE_RANDOM_COMPUTE(RandomUniform_Compute, Distribution::kUniform)
METAL_DEFINE_RANDOM_COMPUTE(RandomNormal_Compute, Distribution::kNormal)
METAL_DEFINE_RANDOM_COMPUTE(TruncatedNormal_Compute,
                            Distribution::kTruncatedNormal)

#undef METAL_DEFINE_RANDOM_COMPUTE

void RegisterRandom(const char* op_name,
                    void (*compute)(void*, TF_OpKernelContext*),
                    TF_DataType index_dtype, const std::string& kernel_name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &RandomOp_Create, compute, &RandomOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "dtype", TF_FLOAT, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_KernelBuilder_TypeConstraint(builder, "T", index_dtype, status);
  }
  // The shape is read on the host to size the allocation.
  TF_KernelBuilder_HostMemory(builder, "shape");
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(kernel_name.c_str(), builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << kernel_name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalRandomKernels() {
  static constexpr TF_DataType kIndexTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kIndexSuffixes[] = {"Int32", "Int64"};
  for (int i = 0; i < 2; ++i) {
    RegisterRandom("RandomUniform", &RandomUniform_Compute, kIndexTypes[i],
                   std::string("MetalRandomUniform") + kIndexSuffixes[i]);
    RegisterRandom("RandomStandardNormal", &RandomNormal_Compute,
                   kIndexTypes[i],
                   std::string("MetalRandomStandardNormal") +
                       kIndexSuffixes[i]);
    RegisterRandom("TruncatedNormal", &TruncatedNormal_Compute, kIndexTypes[i],
                   std::string("MetalTruncatedNormal") + kIndexSuffixes[i]);
    // RandomUniformInt's dtype constraint names the output, which is int32,
    // so it cannot go through RegisterRandom's float32 constraint.
    {
      TF_Status* st = TF_NewStatus();
      TF_KernelBuilder* b = TF_NewKernelBuilder(
          "RandomUniformInt", kMetalDeviceType, &RandomOp_Create,
          &RandomUniformInt_Compute, &RandomOp_Delete);
      TF_KernelBuilder_TypeConstraint(b, "Tout", TF_INT32, st);
      if (TF_GetCode(st) == TF_OK) {
        TF_KernelBuilder_TypeConstraint(b, "T", kIndexTypes[i], st);
      }
      TF_KernelBuilder_HostMemory(b, "shape");
      const std::string name =
          std::string("MetalRandomUniformInt") + kIndexSuffixes[i];
      if (TF_GetCode(st) == TF_OK) {
        TF_RegisterKernelBuilder(name.c_str(), b, st);
      } else {
        TF_DeleteKernelBuilder(b);
      }
      if (TF_GetCode(st) != TF_OK) {
        LOG(ERROR) << "Metal: could not register kernel " << name << ": "
                   << TF_Message(st);
      }
      TF_DeleteStatus(st);
    }
  }
}

}  // namespace metal
}  // namespace tensorflow
