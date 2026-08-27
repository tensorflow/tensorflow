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
#include <cstring>
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

// PopulationCount, CumulativeLogsumexp, LRNGrad, AdjustContrast and
// ParallelConcat.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct ExtraOp {
  TF_DataType dtype = TF_FLOAT;
  bool exclusive = false;
  bool reverse = false;
  int depth_radius = 5;
  float bias = 1.0f;
  float alpha = 1.0f;
  float beta = 0.5f;
  int64_t axis = 0;
};

void* ExtraOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ExtraOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "exclusive", &flag, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->exclusive = flag != 0;
  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "reverse", &flag, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->reverse = flag != 0;
  int32_t radius = 5;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "depth_radius", &radius, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->depth_radius = radius;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "bias", &op->bias, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrFloat(ctx, "alpha", &op->alpha, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrFloat(ctx, "beta", &op->beta, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void ExtraOp_Delete(void* kernel) { delete static_cast<ExtraOp*>(kernel); }

NSArray<NSNumber*>* ToNS(const std::vector<int64_t>& v) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t x : v) [a addObject:@(static_cast<NSInteger>(x))];
  return a;
}

// Reads a host int32/int64 scalar.
bool ReadScalar(TF_Tensor* t, int64_t* out, TF_Status* status) {
  const void* p = TF_TensorData(t);
  if (p == nullptr || TF_TensorElementCount(t) < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: expected a scalar argument.");
    return false;
  }
  *out = TF_TensorType(t) == TF_INT32 ? *static_cast<const int32_t*>(p)
                                      : *static_cast<const int64_t*>(p);
  return true;
}

/*** POPULATION COUNT ***/

void PopulationCount_ComputeImpl(ExtraOp* op, TF_OpKernelContext* ctx,
                                 TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t count = ElementCount(shape);
  // TensorFlow declares the output as uint8; this backend has no uint8, so
  // the count is produced as int32 and the op is registered only where that
  // is what callers read.
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_INT32, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_INT32), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "PopulationCount";
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        [out->inputs addObject:x];
        [out->outputs
            addObject:[g castTensor:[g bitwisePopulationCountWithTensor:x
                                                                   name:nil]
                             toType:MPSDataTypeInt32
                               name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), TF_INT32, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** CUMULATIVE LOGSUMEXP ***/

// A running log-sum-exp. Computed as max + log(cumsum(exp(x - max))) with the
// maximum taken as a running maximum rather than a global one, so the shift
// stays tight along the scan and the exponentials cannot overflow.
void CumulativeLogsumexp_ComputeImpl(ExtraOp* op, TF_OpKernelContext* ctx,
                                     TF_Status* status) {
  ScopedTensor input, axis_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, axis_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int rank = static_cast<int>(shape.size());
  int64_t axis = 0;
  if (!ReadScalar(axis_t.get(), &axis, status)) return;
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: CumulativeLogsumexp axis is out of range.");
    return;
  }

  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), rank,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "CumulativeLogsumexp";
  AppendShapeToKey(shape, &key);
  key.append("/a").append(std::to_string(axis));
  key.append(op->exclusive ? "/excl" : "/incl");
  key.append(op->reverse ? "/rev" : "/fwd");
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSInteger mps_axis = static_cast<NSInteger>(axis);
  const BOOL excl = op->exclusive ? YES : NO;
  const BOOL rev = op->reverse ? YES : NO;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        // The running maximum along the same axis and direction is the
        // tightest shift available at each position, which is what keeps the
        // exponentials in range for large inputs.
        MPSGraphTensor* running_max = [g cumulativeMaximumWithTensor:x
                                                                axis:mps_axis
                                                           exclusive:excl
                                                             reverse:rev
                                                                name:nil];
        MPSGraphTensor* shifted =
            [g subtractionWithPrimaryTensor:x
                            secondaryTensor:running_max
                                       name:nil];
        MPSGraphTensor* summed =
            [g cumulativeSumWithTensor:[g exponentWithTensor:shifted name:nil]
                                  axis:mps_axis
                             exclusive:excl
                               reverse:rev
                                  name:nil];
        [out->inputs addObject:x];
        [out->outputs
            addObject:[g additionWithPrimaryTensor:running_max
                                   secondaryTensor:[g logarithmWithTensor:summed
                                                                     name:nil]
                                              name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** LRN GRADIENT ***/

// The gradient of local response normalisation has two terms: the direct one
// through the numerator, and the one through the denominator, which couples
// every channel in the window back to every other. Both are built from the
// same prefix-sum window the forward pass uses.
void LRNGrad_ComputeImpl(ExtraOp* op, TF_OpKernelContext* ctx,
                         TF_Status* status) {
  ScopedTensor grad, in_image, out_image;
  TF_GetInput(ctx, 0, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, in_image.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, out_image.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(in_image.get());
  if (shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: LRNGrad expects a rank-4 NHWC input.");
    return;
  }
  const int64_t channels = shape[3];
  const int64_t count = ElementCount(shape);

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  const int radius = op->depth_radius;
  const float bias = op->bias, alpha = op->alpha, beta = op->beta;
  std::string key = "LRNGrad";
  AppendShapeToKey(shape, &key);
  key.append("/r").append(std::to_string(radius));
  key.append("/b").append(std::to_string(bias));
  key.append("/a").append(std::to_string(alpha));
  key.append("/e").append(std::to_string(beta));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  std::vector<int64_t> pad_l(4, 0), pad_r(4, 0), one_left(4, 0), none(4, 0);
  pad_l[3] = radius;
  pad_r[3] = radius;
  one_left[3] = 1;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* dy = [g placeholderWithShape:MPSShape(shape)
                                            dataType:mps_dtype
                                                name:nil];
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];

        // Rebuild the window sum exactly as the forward pass does.
        MPSGraphTensor* sq = [g squareWithTensor:x name:nil];
        MPSGraphTensor* padded = [g padTensor:sq
                              withPaddingMode:MPSGraphPaddingModeConstant
                                  leftPadding:ToNS(pad_l)
                                 rightPadding:ToNS(pad_r)
                                constantValue:0.0
                                         name:nil];
        MPSGraphTensor* prefix = [g cumulativeSumWithTensor:padded
                                                       axis:3
                                                       name:nil];
        MPSGraphTensor* shifted_prefix =
            [g padTensor:prefix
         withPaddingMode:MPSGraphPaddingModeConstant
             leftPadding:ToNS(one_left)
            rightPadding:ToNS(none)
           constantValue:0.0
                    name:nil];
        MPSGraphTensor* hi = [g sliceTensor:shifted_prefix
                                  dimension:3
                                      start:2 * radius + 1
                                     length:static_cast<NSInteger>(channels)
                                       name:nil];
        MPSGraphTensor* lo = [g sliceTensor:shifted_prefix
                                  dimension:3
                                      start:0
                                     length:static_cast<NSInteger>(channels)
                                       name:nil];
        MPSGraphTensor* window =
            [g subtractionWithPrimaryTensor:hi secondaryTensor:lo name:nil];
        MPSGraphTensor* norm = [g
            additionWithPrimaryTensor:[g constantWithScalar:bias
                                                   dataType:mps_dtype]
                      secondaryTensor:
                          [g multiplicationWithPrimaryTensor:window
                                             secondaryTensor:
                                                 [g constantWithScalar:alpha
                                                              dataType:mps_dtype]
                                                        name:nil]
                                 name:nil];
        MPSGraphTensor* beta_t =
            [g constantWithScalar:beta dataType:mps_dtype];
        MPSGraphTensor* denom =
            [g powerWithPrimaryTensor:norm secondaryTensor:beta_t name:nil];

        // Direct term: dy / norm^beta.
        MPSGraphTensor* direct =
            [g divisionWithPrimaryTensor:dy secondaryTensor:denom name:nil];

        // Coupling term. d/dx_j of x_i * norm_i^-beta contributes
        // -2*alpha*beta * x_i * x_j * norm_i^(-beta-1) for every j in i's
        // window, so the per-i factor is summed over the same window before
        // being multiplied by x_j.
        MPSGraphTensor* factor = [g
            divisionWithPrimaryTensor:
                [g multiplicationWithPrimaryTensor:dy secondaryTensor:x name:nil]
                      secondaryTensor:
                          [g multiplicationWithPrimaryTensor:denom
                                             secondaryTensor:norm
                                                        name:nil]
                                 name:nil];
        MPSGraphTensor* fpad = [g padTensor:factor
                            withPaddingMode:MPSGraphPaddingModeConstant
                                leftPadding:ToNS(pad_l)
                               rightPadding:ToNS(pad_r)
                              constantValue:0.0
                                       name:nil];
        MPSGraphTensor* fprefix = [g cumulativeSumWithTensor:fpad
                                                        axis:3
                                                        name:nil];
        MPSGraphTensor* fshift = [g padTensor:fprefix
                              withPaddingMode:MPSGraphPaddingModeConstant
                                  leftPadding:ToNS(one_left)
                                 rightPadding:ToNS(none)
                                constantValue:0.0
                                         name:nil];
        MPSGraphTensor* fhi = [g sliceTensor:fshift
                                   dimension:3
                                       start:2 * radius + 1
                                      length:static_cast<NSInteger>(channels)
                                        name:nil];
        MPSGraphTensor* flo = [g sliceTensor:fshift
                                   dimension:3
                                       start:0
                                      length:static_cast<NSInteger>(channels)
                                        name:nil];
        MPSGraphTensor* fwindow =
            [g subtractionWithPrimaryTensor:fhi secondaryTensor:flo name:nil];
        MPSGraphTensor* coupling = [g
            multiplicationWithPrimaryTensor:
                [g multiplicationWithPrimaryTensor:fwindow
                                   secondaryTensor:x
                                              name:nil]
                            secondaryTensor:
                                [g constantWithScalar:-2.0 * alpha * beta
                                             dataType:mps_dtype]
                                       name:nil];

        [out->inputs addObject:dy];
        [out->inputs addObject:x];
        [out->outputs addObject:[g additionWithPrimaryTensor:direct
                                            secondaryTensor:coupling
                                                       name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* g_data =
      TensorDataForTensor(grad.get(), op->dtype, device, status);
  if (g_data == nil) return;
  MPSGraphTensorData* x_data =
      TensorDataForTensor(in_image.get(), op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ g_data, x_data ], @[ o_data ], status);
}

/*** ADJUST CONTRAST, THE V1 FORM ***/

// The v1 op takes explicit output bounds alongside the factor and clamps to
// them; v2 has neither.
void AdjustContrastV1_ComputeImpl(ExtraOp* op, TF_OpKernelContext* ctx,
                                  TF_Status* status) {
  ScopedTensor images, factor, lo_t, hi_t;
  TF_GetInput(ctx, 0, images.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, factor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, lo_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, hi_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(images.get());
  if (shape.size() < 3) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: AdjustContrast needs at least three dimensions.");
    return;
  }
  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::string key = "AdjustContrastV1";
  AppendShapeToKey(shape, &key);
  const NSInteger rank = static_cast<NSInteger>(shape.size());
  NSArray<NSNumber*>* axes = @[ @(rank - 3), @(rank - 2) ];
  const std::vector<int64_t> scalar = {1};

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* f = [g placeholderWithShape:MPSShape(scalar)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* lo = [g placeholderWithShape:MPSShape(scalar)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];
        MPSGraphTensor* hi = [g placeholderWithShape:MPSShape(scalar)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];
        MPSGraphTensor* mean = [g meanOfTensor:x axes:axes name:nil];
        MPSGraphTensor* adjusted = [g
            additionWithPrimaryTensor:
                [g multiplicationWithPrimaryTensor:
                       [g subtractionWithPrimaryTensor:x
                                       secondaryTensor:mean
                                                  name:nil]
                                   secondaryTensor:f
                                              name:nil]
                      secondaryTensor:mean
                                 name:nil];
        [out->inputs addObject:x];
        [out->inputs addObject:f];
        [out->inputs addObject:lo];
        [out->inputs addObject:hi];
        [out->outputs addObject:[g clampWithTensor:adjusted
                                    minValueTensor:lo
                                    maxValueTensor:hi
                                              name:nil]];
      },
      status);
  if (cached == nullptr) return;

  BufferSlice f_slice, lo_slice, hi_slice;
  if (!SliceForTensor(factor.get(), &f_slice, status)) return;
  if (!SliceForTensor(lo_t.get(), &lo_slice, status)) return;
  if (!SliceForTensor(hi_t.get(), &hi_slice, status)) return;
  MPSGraphTensorData* x_data =
      TensorDataForTensor(images.get(), TF_FLOAT, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* f_data =
      TensorDataFor(f_slice, scalar, TF_FLOAT, device, status);
  if (f_data == nil) return;
  MPSGraphTensorData* lo_data =
      TensorDataFor(lo_slice, scalar, TF_FLOAT, device, status);
  if (lo_data == nil) return;
  MPSGraphTensorData* hi_data =
      TensorDataFor(hi_slice, scalar, TF_FLOAT, device, status);
  if (hi_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), TF_FLOAT, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ x_data, f_data, lo_data, hi_data ],
           @[ o_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<ExtraOp*>(kernel);                                 \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(PopulationCount_Compute, PopulationCount_ComputeImpl)
METAL_COMPUTE(CumulativeLogsumexp_Compute, CumulativeLogsumexp_ComputeImpl)
METAL_COMPUTE(LRNGrad_Compute, LRNGrad_ComputeImpl)
METAL_COMPUTE(AdjustContrastV1_Compute, AdjustContrastV1_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, std::vector<const char*> host_args,
              bool constrain_t = true, const char* attr2 = nullptr,
              TF_DataType dtype2 = TF_INT32) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &ExtraOp_Create, compute, &ExtraOp_Delete);
  if (constrain_t) TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (TF_GetCode(status) == TF_OK && attr2 != nullptr) {
    TF_KernelBuilder_TypeConstraint(builder, attr2, dtype2, status);
  }
  for (const char* a : host_args) TF_KernelBuilder_HostMemory(builder, a);
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

void RegisterMetalExtraKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kIndexTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kIndexSuffixes[] = {"Int32", "Int64"};

  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    for (int j = 0; j < 2; ++j) {
      // The scan axis is read on the host to key the graph.
      Register("CumulativeLogsumexp", &CumulativeLogsumexp_Compute, t,
               "MetalCumulativeLogsumexp" + s + kIndexSuffixes[j], {"axis"},
               true, "Tidx", kIndexTypes[j]);
    }
  }
  // PopulationCount counts set bits, so it only makes sense over integers.
  for (int j = 0; j < 2; ++j) {
    Register("PopulationCount", &PopulationCount_Compute, kIndexTypes[j],
             std::string("MetalPopulationCount") + kIndexSuffixes[j], {});
  }
  Register("LRNGrad", &LRNGrad_Compute, TF_FLOAT, "MetalLRNGradFloat", {});
  Register("AdjustContrast", &AdjustContrastV1_Compute, TF_FLOAT,
           "MetalAdjustContrastFloat", {});
}

}  // namespace metal
}  // namespace tensorflow
