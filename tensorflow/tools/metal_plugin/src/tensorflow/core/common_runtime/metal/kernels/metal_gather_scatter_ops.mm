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
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_mps_graph.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_shader_library.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// ResourceGather, ResourceGatherNd and ResourceScatterUpdate.
//
// These read or write through a resource variable rather than a plain tensor,
// so the variable machinery from the experimental kernel C API does the
// locking and the copy-on-read, and the gather or scatter itself is the same
// MPSGraph operation the non-resource forms use.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// Copies one device tensor onto another, for the variable machinery.
void CopyTensorOnDevice(TF_OpKernelContext* ctx, TF_Tensor* source,
                        TF_Tensor* dest) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  SP_Stream stream = StreamForContext(ctx, status);
  BufferSlice src, dst;
  if (TF_GetCode(status) == TF_OK && SliceForTensor(source, &src, status) &&
      SliceForTensor(dest, &dst, status)) {
    const size_t bytes = TF_TensorByteSize(source);
    OrderedCommandBuffer command_buffer(stream);
    if (command_buffer.ok() && bytes > 0) {
      id<MTLBlitCommandEncoder> encoder =
          [command_buffer.get() blitCommandEncoder];
      [encoder copyFromBuffer:src.buffer
                 sourceOffset:src.offset
                     toBuffer:dst.buffer
            destinationOffset:dst.offset
                         size:bytes];
      [encoder endEncoding];
      command_buffer.Commit();
    }
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: variable copy failed: " << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

struct ResourceOp {
  TF_DataType dtype = TF_FLOAT;
  int64_t batch_dims = 0;
};

void* ResourceOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ResourceOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "dtype", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_SetStatus(status, TF_OK, "");
      op->dtype = TF_FLOAT;
    }
  }
  int32_t bd = 0;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "batch_dims", &bd, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->batch_dims = bd;
  TF_DeleteStatus(status);
  return op;
}

void ResourceOp_Delete(void* kernel) { delete static_cast<ResourceOp*>(kernel); }

/*** RESOURCE GATHER ***/

void ResourceGather_ComputeImpl(ResourceOp* op, TF_OpKernelContext* ctx,
                                TF_Status* status) {
  ScopedTensor params;
  // Reading through the variable takes the same lock an assignment would, so
  // a concurrent write cannot tear the read.
  TF_GetInputTensorFromVariable(ctx, 0, /*lock_held=*/false,
                                /*isVariantType=*/false, /*sparse=*/false,
                                &CopyTensorOnDevice, params.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  ScopedTensor indices;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> p_shape = ShapeOf(params.get());
  const std::vector<int64_t> i_shape = ShapeOf(indices.get());
  if (p_shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ResourceGather needs a rank of at least 1.");
    return;
  }
  const int bd = static_cast<int>(op->batch_dims);
  std::vector<int64_t> out_shape;
  for (size_t i = bd; i < i_shape.size(); ++i) out_shape.push_back(i_shape[i]);
  for (size_t i = 1; i < p_shape.size(); ++i) out_shape.push_back(p_shape[i]);
  for (int i = 0; i < bd; ++i) out_shape.insert(out_shape.begin(), i_shape[i]);

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype, idx_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;
  if (!MPSTypeFor(TF_TensorType(indices.get()), &idx_dtype, status)) return;

  std::string key = "ResourceGather";
  AppendShapeToKey(p_shape, &key);
  AppendShapeToKey(i_shape, &key);
  key.append("/b").append(std::to_string(bd));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSUInteger mps_bd = static_cast<NSUInteger>(bd);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* p = [out->graph placeholderWithShape:MPSShape(p_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* i = [out->graph placeholderWithShape:MPSShape(i_shape)
                                                    dataType:idx_dtype
                                                        name:nil];
        [out->inputs addObject:p];
        [out->inputs addObject:i];
        [out->outputs addObject:[out->graph gatherWithUpdatesTensor:p
                                                      indicesTensor:i
                                                               axis:0
                                                    batchDimensions:mps_bd
                                                               name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* p_data =
      TensorDataForTensor(params.get(), op->dtype, device, status);
  if (p_data == nil) return;
  MPSGraphTensorData* i_data = TensorDataForTensor(
      indices.get(), TF_TensorType(indices.get()), device, status);
  if (i_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ p_data, i_data ], @[ o_data ], status);
}

/*** GATHER ND ***/

// Indexes with a whole coordinate per entry rather than one along an axis:
// the last dimension of `indices` names a position in the leading dimensions
// of `params`, and everything past those dimensions comes along as a slice.
//
// `params` arrives either as a plain tensor or through a variable; the graph
// is the same either way, which is why both ops share this.
void GatherNd_Run(ResourceOp* op, TF_OpKernelContext* ctx, TF_Tensor* params,
                  TF_Tensor* indices, TF_Status* status) {
  const std::vector<int64_t> p_shape = ShapeOf(params);
  const std::vector<int64_t> i_shape = ShapeOf(indices);
  if (p_shape.empty() || i_shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: GatherNd needs a rank of at least 1 on both inputs.");
    return;
  }
  const int64_t index_depth = i_shape.back();
  if (index_depth < 1 || index_depth > static_cast<int64_t>(p_shape.size())) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the last dimension of GatherNd indices must not "
                 "exceed the rank of params.");
    return;
  }

  std::vector<int64_t> out_shape;
  for (size_t i = 0; i + 1 < i_shape.size(); ++i) {
    out_shape.push_back(i_shape[i]);
  }
  for (size_t i = index_depth; i < p_shape.size(); ++i) {
    out_shape.push_back(p_shape[i]);
  }
  // A full-depth index over a rank-1 params leaves nothing behind, and an
  // empty shape is a scalar, which the allocator accepts as rank zero.
  const int64_t count = ElementCount(out_shape);

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype, idx_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;
  const TF_DataType index_type = TF_TensorType(indices);
  if (!MPSTypeFor(index_type, &idx_dtype, status)) return;

  std::string key = "GatherNd";
  AppendShapeToKey(p_shape, &key);
  AppendShapeToKey(i_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  key.append("/i").append(std::to_string(static_cast<int>(index_type)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* p = [g placeholderWithShape:MPSShape(p_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* i = [g placeholderWithShape:MPSShape(i_shape)
                                           dataType:idx_dtype
                                               name:nil];
        [out->inputs addObject:p];
        [out->inputs addObject:i];
        // No batch dimensions: TensorFlow's GatherNd indexes from the front of
        // params, and its batched behaviour is expressed by the caller instead.
        [out->outputs addObject:[g gatherNDWithUpdatesTensor:p
                                              indicesTensor:i
                                            batchDimensions:0
                                                       name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* p_data =
      TensorDataForTensor(params, op->dtype, device, status);
  if (p_data == nil) return;
  MPSGraphTensorData* i_data =
      TensorDataForTensor(indices, index_type, device, status);
  if (i_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ p_data, i_data ], @[ o_data ], status);
}

void GatherNd_ComputeImpl(ResourceOp* op, TF_OpKernelContext* ctx,
                          TF_Status* status) {
  ScopedTensor params, indices;
  TF_GetInput(ctx, 0, params.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  // The plain form's type attribute is Tparams, not dtype, so it is taken
  // from the tensor rather than from construction.
  op->dtype = TF_TensorType(params.get());
  GatherNd_Run(op, ctx, params.get(), indices.get(), status);
}

void ResourceGatherNd_ComputeImpl(ResourceOp* op, TF_OpKernelContext* ctx,
                                  TF_Status* status) {
  ScopedTensor params;
  // Reading through the variable takes the same lock an assignment would, so
  // a concurrent write cannot tear the read.
  TF_GetInputTensorFromVariable(ctx, 0, /*lock_held=*/false,
                                /*isVariantType=*/false, /*sparse=*/false,
                                &CopyTensorOnDevice, params.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  ScopedTensor indices;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  GatherNd_Run(op, ctx, params.get(), indices.get(), status);
}

/*** RESOURCE SCATTER UPDATE ***/

// Writes updates into the variable at the given indices along axis 0. The
// variable is locked for the whole read-modify-write, which is what makes two
// concurrent scatters to the same variable safe.
void ResourceScatterUpdate_ComputeImpl(ResourceOp* op, TF_OpKernelContext* ctx,
                                       TF_Status* status) {
  const int locked[] = {0};
  TF_VariableInputLockHolder* holder = nullptr;
  TF_MaybeLockVariableInputMutexesInOrder(ctx, /*do_lock=*/true,
                                          /*sparse=*/false, locked, 1,
                                          &CopyTensorOnDevice, &holder, status);
  if (TF_GetCode(status) != TF_OK) return;
  struct Unlock {
    TF_VariableInputLockHolder* h;
    ~Unlock() {
      if (h != nullptr) TF_ReleaseVariableInputLockHolder(h);
    }
  } unlock{holder};

  ScopedTensor var;
  TF_GetInputTensorFromVariable(ctx, 0, /*lock_held=*/true, false, false,
                                &CopyTensorOnDevice, var.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  ScopedTensor indices, updates;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, updates.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> v_shape = ShapeOf(var.get());
  const std::vector<int64_t> i_shape = ShapeOf(indices.get());
  const std::vector<int64_t> u_shape = ShapeOf(updates.get());
  if (v_shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ResourceScatterUpdate needs a rank of at least 1.");
    return;
  }
  if (ElementCount(v_shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype, idx_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;
  if (!MPSTypeFor(TF_TensorType(indices.get()), &idx_dtype, status)) return;

  std::string key = "ResourceScatterUpdate";
  AppendShapeToKey(v_shape, &key);
  AppendShapeToKey(i_shape, &key);
  AppendShapeToKey(u_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* v = [g placeholderWithShape:MPSShape(v_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* i = [g placeholderWithShape:MPSShape(i_shape)
                                           dataType:idx_dtype
                                               name:nil];
        MPSGraphTensor* u = [g placeholderWithShape:MPSShape(u_shape)
                                           dataType:mps_dtype
                                               name:nil];
        // Set, not add: an update overwrites whatever was at the index, and
        // repeated indices resolve to one of the updates rather than a sum.
        [out->inputs addObject:v];
        [out->inputs addObject:i];
        [out->inputs addObject:u];
        [out->outputs addObject:[g scatterWithDataTensor:v
                                           updatesTensor:u
                                           indicesTensor:i
                                                    axis:0
                                                    mode:MPSGraphScatterModeSet
                                                    name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* v_data =
      TensorDataForTensor(var.get(), op->dtype, device, status);
  if (v_data == nil) return;
  MPSGraphTensorData* i_data = TensorDataForTensor(
      indices.get(), TF_TensorType(indices.get()), device, status);
  if (i_data == nil) return;
  MPSGraphTensorData* u_data =
      TensorDataForTensor(updates.get(), op->dtype, device, status);
  if (u_data == nil) return;
  // The result is written back over the variable's own storage.
  MPSGraphTensorData* out_data =
      TensorDataForTensor(var.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ v_data, i_data, u_data ], @[ out_data ],
           status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<ResourceOp*>(kernel);                              \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(ResourceGather_Compute, ResourceGather_ComputeImpl)
METAL_COMPUTE(GatherNd_Compute, GatherNd_ComputeImpl)
METAL_COMPUTE(ResourceGatherNd_Compute, ResourceGatherNd_ComputeImpl)
METAL_COMPUTE(ResourceScatterUpdate_Compute, ResourceScatterUpdate_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              TF_DataType index_dtype, const std::string& name,
              const char* type_attr = "dtype", bool resource_input = true) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, &ResourceOp_Create,
                          compute, &ResourceOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, type_attr, dtype, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_KernelBuilder_TypeConstraint(builder, "Tindices", index_dtype, status);
  }
  // The variable handle is a resource and lives on the host.
  if (resource_input) TF_KernelBuilder_HostMemory(builder, "resource");
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

/*** WRITING TO A VARIABLE ***/

// The arithmetic behind AssignAdd and AssignSub, in place on the variable's
// own storage. `op` is 1 for an addition and anything else for a subtraction;
// the value is this file's own, since the C API passes it through untouched.
void UpdateVariableOnDevice(TF_OpKernelContext* ctx, TF_Tensor* tensor,
                            TF_Tensor* value, int op) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  const int64_t count = TF_TensorElementCount(tensor);
  const TF_DataType dtype = TF_TensorType(tensor);
  SP_Stream stream = StreamForContext(ctx, status);
  BufferSlice target, delta;
  if (TF_GetCode(status) == TF_OK && count > 0 &&
      count == TF_TensorElementCount(value) &&
      SliceForTensor(tensor, &target, status) &&
      SliceForTensor(value, &delta, status)) {
    const char* fn = dtype == TF_HALF ? "tf_assign_update_half"
                                      : "tf_assign_update_float";
    id<MTLComputePipelineState> pipeline =
        PipelineFor(DeviceForStream(stream), fn, status);
    OrderedCommandBuffer command_buffer(stream);
    if (pipeline != nil && command_buffer.ok()) {
      FillParams params = {};
      params.count = static_cast<uint32_t>(count);
      params.value = op == 1 ? 1.0f : -1.0f;
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer.get() computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:target.buffer offset:target.offset atIndex:0];
      [encoder setBuffer:delta.buffer offset:delta.offset atIndex:1];
      [encoder setBytes:&params length:sizeof(params) atIndex:2];
      Dispatch1D(encoder, pipeline, params.count);
      [encoder endEncoding];
      command_buffer.Commit();
    }
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: a variable update failed: " << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

struct VariableOp {
  bool validate_shape = false;
};

void* VariableOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new VariableOp();
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "validate_shape", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->validate_shape = flag != 0;
  TF_DeleteStatus(status);
  return op;
}

void VariableOp_Delete(void* kernel) {
  delete static_cast<VariableOp*>(kernel);
}

void AssignVariable_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<VariableOp*>(kernel);
  TF_AssignVariable(ctx, 0, 1, op != nullptr && op->validate_shape,
                    &CopyTensorOnDevice, status);
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

template <int kOp>
void AssignUpdateVariable_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  TF_AssignUpdateVariable(ctx, 0, 1, kOp, /*isVariantType=*/0,
                          &CopyTensorOnDevice, &UpdateVariableOnDevice,
                          status);
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void RegisterVariableOp(const char* op_name,
                        void (*compute)(void*, TF_OpKernelContext*),
                        TF_DataType dtype, const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &VariableOp_Create, compute,
      &VariableOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "dtype", dtype, status);
  // The handle is a resource and lives on the host.
  TF_KernelBuilder_HostMemory(builder, "resource");
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

void RegisterMetalResourceKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kIndexTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kIndexSuffixes[] = {"Int32", "Int64"};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      const std::string suffix =
          std::string(kSuffixes[i]) + kIndexSuffixes[j];
      Register("ResourceGather", &ResourceGather_Compute, kDTypes[i],
               kIndexTypes[j], "MetalResourceGather" + suffix);
      Register("ResourceScatterUpdate", &ResourceScatterUpdate_Compute,
               kDTypes[i], kIndexTypes[j],
               "MetalResourceScatterUpdate" + suffix);
      Register("ResourceGatherNd", &ResourceGatherNd_Compute, kDTypes[i],
               kIndexTypes[j], "MetalResourceGatherNd" + suffix);
    }
  }

  // Writing to a variable, on the device.
  //
  // Without these TensorFlow falls back to its own DEVICE_DEFAULT kernels,
  // which reach the variable through its data pointer. On a unified memory
  // device that pointer is host-addressable, so those kernels read and write
  // device memory from the host with no idea that GPU work is in flight
  // against it, and an optimizer that reads a slot and writes it back races
  // with the arithmetic that produced the value. It is not a hypothetical: on
  // a TensorFlow that does export this C API, six steps of an Adam-shaped
  // update disagreed with the CPU and then went non-finite.
  for (int i = 0; i < 2; ++i) {
    const std::string s = kSuffixes[i];
    RegisterVariableOp("AssignVariableOp", &AssignVariable_Compute, kDTypes[i],
                       "MetalAssignVariableOp" + s);
    RegisterVariableOp("AssignAddVariableOp",
                       &AssignUpdateVariable_Compute<1>, kDTypes[i],
                       "MetalAssignAddVariableOp" + s);
    RegisterVariableOp("AssignSubVariableOp",
                       &AssignUpdateVariable_Compute<2>, kDTypes[i],
                       "MetalAssignSubVariableOp" + s);
  }
}

// Registered on its own, because it is not a resource op.
//
// GatherNd takes a plain tensor and needs none of the variable entry points,
// but it used to be registered inside RegisterMetalResourceKernels, which an
// out-of-tree build skips when those entry points are missing. It went
// missing with them, and an op with no kernel is indistinguishable from an
// op the backend never claimed.
void RegisterMetalGatherNdKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kIndexTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kIndexSuffixes[] = {"Int32", "Int64"};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      Register("GatherNd", &GatherNd_Compute, kDTypes[i], kIndexTypes[j],
               std::string("MetalGatherNd") + kSuffixes[i] + kIndexSuffixes[j],
               "Tparams", /*resource_input=*/false);
    }
  }
}

}  // namespace metal
}  // namespace tensorflow
