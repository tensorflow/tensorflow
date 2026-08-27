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
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_mps_graph.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// Sum and Mean, the two reductions a training step needs: Sum for gradient
// accumulation over broadcast axes, Mean for reducing a per-example loss to a
// scalar.
//
// TensorFlow's DEVICE_DEFAULT registrations for these cover only int32 in host
// memory, so a float reduction genuinely has to be provided here.

enum class ReductionKind { kSum, kMean, kMax, kMin, kProd, kAny, kAll,
                           kEuclideanNorm };

const char* NameOf(ReductionKind k) {
  switch (k) {
    case ReductionKind::kSum: return "Sum";
    case ReductionKind::kMean: return "Mean";
    case ReductionKind::kMax: return "Max";
    case ReductionKind::kMin: return "Min";
    case ReductionKind::kProd: return "Prod";
    case ReductionKind::kAny: return "Any";
    case ReductionKind::kAll: return "All";
    case ReductionKind::kEuclideanNorm: return "EuclideanNorm";
  }
  return "?";
}

MPSGraphTensor* ApplyReduction(MPSGraph* g, ReductionKind k, MPSGraphTensor* x,
                               NSArray<NSNumber*>* axes) {
  switch (k) {
    case ReductionKind::kSum:
      return [g reductionSumWithTensor:x axes:axes name:nil];
    case ReductionKind::kMean:
      return [g meanOfTensor:x axes:axes name:nil];
    case ReductionKind::kMax:
      return [g reductionMaximumWithTensor:x axes:axes name:nil];
    case ReductionKind::kMin:
      return [g reductionMinimumWithTensor:x axes:axes name:nil];
    case ReductionKind::kProd:
      return [g reductionProductWithTensor:x axes:axes name:nil];
    case ReductionKind::kAny:
      return [g reductionOrWithTensor:x axes:axes name:nil];
    case ReductionKind::kAll:
      return [g reductionAndWithTensor:x axes:axes name:nil];
    case ReductionKind::kEuclideanNorm: {
      // sqrt(sum(x^2)), which is what TensorFlow's EuclideanNorm computes.
      MPSGraphTensor* sq = [g squareWithTensor:x name:nil];
      return [g squareRootWithTensor:[g reductionSumWithTensor:sq
                                                          axes:axes
                                                          name:nil]
                                name:nil];
    }
  }
  return nil;
}

struct ReductionOp {
  TF_DataType dtype = TF_FLOAT;
  bool keep_dims = false;
};

void* ReductionOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ReductionOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_Bool keep_dims = 0;
    TF_OpKernelConstruction_GetAttrBool(ctx, "keep_dims", &keep_dims, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_SetStatus(status, TF_OK, "");
      keep_dims = 0;
    }
    op->keep_dims = keep_dims != 0;
  }
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void ReductionOp_Delete(void* kernel) {
  delete static_cast<ReductionOp*>(kernel);
}

// Reads the reduction axes, which arrive as a host-memory int32 or int64
// tensor, and normalises negative entries against the input rank.
bool ReadAxes(TF_Tensor* indices, int rank, std::vector<int>* axes,
              TF_Status* status) {
  const int64_t count = TF_TensorElementCount(indices);
  const TF_DataType dtype = TF_TensorType(indices);
  const void* data = TF_TensorData(indices);
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: reduction_indices has no data.");
    return false;
  }

  // An empty index list means "reduce nothing", which TensorFlow treats as the
  // identity rather than as a full reduction.
  for (int64_t i = 0; i < count; ++i) {
    int64_t axis;
    if (dtype == TF_INT32) {
      axis = static_cast<const int32_t*>(data)[i];
    } else if (dtype == TF_INT64) {
      axis = static_cast<const int64_t*>(data)[i];
    } else {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: reduction_indices must be int32 or int64.");
      return false;
    }
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   ("Metal: reduction axis " + std::to_string(axis) +
                    " is out of range for a rank-" + std::to_string(rank) +
                    " tensor.")
                       .c_str());
      return false;
    }
    axes->push_back(static_cast<int>(axis));
  }
  std::sort(axes->begin(), axes->end());
  axes->erase(std::unique(axes->begin(), axes->end()), axes->end());
  return true;
}

template <ReductionKind kKind>
void Reduction_ComputeImpl(ReductionOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor input;
  ScopedTensor indices;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int rank = static_cast<int>(in_shape.size());
  std::vector<int> axes;
  if (!ReadAxes(indices.get(), rank, &axes, status)) return;

  // Shape MPSGraph produces: reduced axes become 1. TensorFlow drops them
  // unless keep_dims, so the two shapes are tracked separately and a reshape
  // bridges them.
  std::vector<int64_t> reduced_shape = in_shape;
  for (int axis : axes) reduced_shape[axis] = 1;

  std::vector<int64_t> out_shape;
  if (op->keep_dims) {
    out_shape = reduced_shape;
  } else {
    for (int i = 0; i < rank; ++i) {
      if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
        out_shape.push_back(in_shape[i]);
      }
    }
  }

  int64_t out_count = 1;
  for (int64_t dim : out_shape) out_count *= dim;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(out_count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (out_count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = NameOf(kKind);
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(out_shape, &key);
  key.push_back('/');
  for (int axis : axes) key.append(std::to_string(axis)).push_back(',');
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  NSMutableArray<NSNumber*>* mps_axes = [NSMutableArray array];
  for (int axis : axes) [mps_axes addObject:@(static_cast<NSInteger>(axis))];
  const std::vector<int64_t> final_shape = out_shape;
  const bool empty_axes = axes.empty();

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* source =
            [out->graph placeholderWithShape:MPSShape(in_shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* reduced = source;
        if (!empty_axes) {
          reduced = ApplyReduction(out->graph, kKind, source, mps_axes);
        }
        MPSGraphTensor* shaped =
            [out->graph reshapeTensor:reduced
                            withShape:MPSShape(final_shape)
                                 name:nil];
        [out->inputs addObject:source];
        [out->outputs addObject:shaped];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ out_data ], status);
}

template <ReductionKind kKind>
void Reduction_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<ReductionOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: reduction kernel has no state.");
  } else {
    Reduction_ComputeImpl<kKind>(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void RegisterReduction(const char* op_name,
                       void (*compute)(void*, TF_OpKernelContext*),
                       TF_DataType dtype, TF_DataType index_dtype,
                       const std::string& kernel_name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &ReductionOp_Create, compute,
      &ReductionOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_KernelBuilder_TypeConstraint(builder, "Tidx", index_dtype, status);
  }
  // The axes are read on the host to work out the output shape, so they must
  // not be placed on the device.
  TF_KernelBuilder_HostMemory(builder, "reduction_indices");
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

void RegisterMetalReductionKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kIndexTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kIndexSuffixes[] = {"Int32", "Int64"};

  struct Entry {
    const char* op;
    void (*compute)(void*, TF_OpKernelContext*);
  };
  static const Entry kNumeric[] = {
      {"Sum", &Reduction_Compute<ReductionKind::kSum>},
      {"Mean", &Reduction_Compute<ReductionKind::kMean>},
      {"Max", &Reduction_Compute<ReductionKind::kMax>},
      {"Min", &Reduction_Compute<ReductionKind::kMin>},
      {"Prod", &Reduction_Compute<ReductionKind::kProd>},
      {"EuclideanNorm",
       &Reduction_Compute<ReductionKind::kEuclideanNorm>},
  };

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (const Entry& e : kNumeric) {
        RegisterReduction(e.op, e.compute, kDTypes[i], kIndexTypes[j],
                          std::string("Metal") + e.op + kSuffixes[i] +
                              kIndexSuffixes[j]);
      }
    }
  }

  // Any and All reduce bools, so they take no float instantiation.
  for (int j = 0; j < 2; ++j) {
    RegisterReduction("Any", &Reduction_Compute<ReductionKind::kAny>, TF_BOOL,
                      kIndexTypes[j],
                      std::string("MetalAny") + kIndexSuffixes[j]);
    RegisterReduction("All", &Reduction_Compute<ReductionKind::kAll>, TF_BOOL,
                      kIndexTypes[j],
                      std::string("MetalAll") + kIndexSuffixes[j]);
  }
}

}  // namespace metal
}  // namespace tensorflow
