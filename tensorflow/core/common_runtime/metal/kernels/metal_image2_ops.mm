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

// Colour space conversion and contrast adjustment.
//
// RGBToHSV and HSVToRGB are elementwise over the channel axis, so they are
// built from selects rather than branches: every pixel evaluates all three
// hue sectors and picks one, which is what a data-parallel form of the CPU
// kernel's if-chain looks like.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct ImageOp {
  TF_DataType dtype = TF_FLOAT;
};

void* ImageOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ImageOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }
  TF_DeleteStatus(status);
  return op;
}

void ImageOp_Delete(void* kernel) { delete static_cast<ImageOp*>(kernel); }

// Splits the trailing channel axis into three tensors of one channel each.
void SplitChannels(MPSGraph* g, MPSGraphTensor* x, NSUInteger axis,
                   MPSGraphTensor** a, MPSGraphTensor** b,
                   MPSGraphTensor** c) {
  *a = [g sliceTensor:x dimension:axis start:0 length:1 name:nil];
  *b = [g sliceTensor:x dimension:axis start:1 length:1 name:nil];
  *c = [g sliceTensor:x dimension:axis start:2 length:1 name:nil];
}

/*** RGB TO HSV ***/

void RGBToHSV_ComputeImpl(ImageOp* op, TF_OpKernelContext* ctx,
                          TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  if (shape.empty() || shape.back() != 3) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: RGBToHSV needs a trailing channel axis of size 3.");
    return;
  }
  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "RGBToHSV";
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSUInteger axis = static_cast<NSUInteger>(shape.size()) - 1;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor *r, *gr, *b;
        SplitChannels(g, x, axis, &r, &gr, &b);

        MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:mps_dtype];
        MPSGraphTensor* six = [g constantWithScalar:6.0 dataType:mps_dtype];
        MPSGraphTensor* two = [g constantWithScalar:2.0 dataType:mps_dtype];
        MPSGraphTensor* four = [g constantWithScalar:4.0 dataType:mps_dtype];

        MPSGraphTensor* vmax =
            [g maximumWithPrimaryTensor:r
                        secondaryTensor:[g maximumWithPrimaryTensor:gr
                                                    secondaryTensor:b
                                                               name:nil]
                                   name:nil];
        MPSGraphTensor* vmin =
            [g minimumWithPrimaryTensor:r
                        secondaryTensor:[g minimumWithPrimaryTensor:gr
                                                    secondaryTensor:b
                                                               name:nil]
                                   name:nil];
        MPSGraphTensor* range =
            [g subtractionWithPrimaryTensor:vmax secondaryTensor:vmin name:nil];

        // Saturation and hue are zero where the range is, and dividing by it
        // would give NaN, so the denominator is nudged and the result selected.
        MPSGraphTensor* flat =
            [g equalWithPrimaryTensor:range secondaryTensor:zero name:nil];
        MPSGraphTensor* safe_range =
            [g selectWithPredicateTensor:flat
                     truePredicateTensor:[g constantWithScalar:1.0
                                                      dataType:mps_dtype]
                    falsePredicateTensor:range
                                    name:nil];
        MPSGraphTensor* safe_max =
            [g selectWithPredicateTensor:
                   [g equalWithPrimaryTensor:vmax secondaryTensor:zero name:nil]
                     truePredicateTensor:[g constantWithScalar:1.0
                                                      dataType:mps_dtype]
                    falsePredicateTensor:vmax
                                    name:nil];
        MPSGraphTensor* s = [g divisionWithPrimaryTensor:range
                                         secondaryTensor:safe_max
                                                    name:nil];

        // Hue sector: which channel holds the maximum decides the offset.
        MPSGraphTensor* norm_rg = [g divisionWithPrimaryTensor:
                                         [g subtractionWithPrimaryTensor:r
                                                         secondaryTensor:gr
                                                                    name:nil]
                                              secondaryTensor:safe_range
                                                         name:nil];
        MPSGraphTensor* norm_gb = [g divisionWithPrimaryTensor:
                                         [g subtractionWithPrimaryTensor:gr
                                                         secondaryTensor:b
                                                                    name:nil]
                                              secondaryTensor:safe_range
                                                         name:nil];
        MPSGraphTensor* norm_br = [g divisionWithPrimaryTensor:
                                         [g subtractionWithPrimaryTensor:b
                                                         secondaryTensor:r
                                                                    name:nil]
                                              secondaryTensor:safe_range
                                                         name:nil];
        MPSGraphTensor* h_r = norm_gb;
        MPSGraphTensor* h_g =
            [g additionWithPrimaryTensor:norm_br secondaryTensor:two name:nil];
        MPSGraphTensor* h_b =
            [g additionWithPrimaryTensor:norm_rg secondaryTensor:four name:nil];

        MPSGraphTensor* max_is_r =
            [g equalWithPrimaryTensor:vmax secondaryTensor:r name:nil];
        MPSGraphTensor* max_is_g =
            [g equalWithPrimaryTensor:vmax secondaryTensor:gr name:nil];
        MPSGraphTensor* h_sector =
            [g selectWithPredicateTensor:max_is_r
                     truePredicateTensor:h_r
                    falsePredicateTensor:
                        [g selectWithPredicateTensor:max_is_g
                                 truePredicateTensor:h_g
                                falsePredicateTensor:h_b
                                                name:nil]
                                    name:nil];
        // Scale to [0,1) and wrap the negative sixth.
        MPSGraphTensor* h_scaled =
            [g divisionWithPrimaryTensor:h_sector secondaryTensor:six name:nil];
        MPSGraphTensor* h_wrapped =
            [g selectWithPredicateTensor:[g lessThanWithPrimaryTensor:h_scaled
                                                      secondaryTensor:zero
                                                                 name:nil]
                     truePredicateTensor:[g additionWithPrimaryTensor:h_scaled
                                                      secondaryTensor:
                                                          [g constantWithScalar:1.0
                                                                       dataType:mps_dtype]
                                                                 name:nil]
                    falsePredicateTensor:h_scaled
                                    name:nil];
        MPSGraphTensor* h = [g selectWithPredicateTensor:flat
                                     truePredicateTensor:zero
                                    falsePredicateTensor:h_wrapped
                                                    name:nil];

        [out->inputs addObject:x];
        [out->outputs addObject:[g concatTensors:@[ h, s, vmax ]
                                       dimension:static_cast<NSInteger>(axis)
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

/*** ADJUST CONTRAST ***/

// AdjustContrastv2 moves each channel toward that channel's mean over the
// image: out = (in - mean) * factor + mean.
void AdjustContrast_ComputeImpl(ImageOp* op, TF_OpKernelContext* ctx,
                                TF_Status* status) {
  ScopedTensor images, factor;
  TF_GetInput(ctx, 0, images.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, factor.address(), status);
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
      ctx, 0, op->dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "AdjustContrast";
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  // The mean is taken over the two spatial axes, leaving batch and channel.
  const NSInteger rank = static_cast<NSInteger>(shape.size());
  NSArray<NSNumber*>* axes = @[ @(rank - 3), @(rank - 2) ];
  const std::vector<int64_t> scalar = {1};

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* f = [g placeholderWithShape:MPSShape(scalar)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* mean = [g meanOfTensor:x axes:axes name:nil];
        MPSGraphTensor* centred =
            [g subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
        [out->inputs addObject:x];
        [out->inputs addObject:f];
        [out->outputs
            addObject:[g additionWithPrimaryTensor:
                             [g multiplicationWithPrimaryTensor:centred
                                                secondaryTensor:f
                                                           name:nil]
                                   secondaryTensor:mean
                                              name:nil]];
      },
      status);
  if (cached == nullptr) return;

  BufferSlice f_slice;
  if (!SliceForTensor(factor.get(), &f_slice, status)) return;
  MPSGraphTensorData* x_data =
      TensorDataForTensor(images.get(), op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* f_data =
      TensorDataFor(f_slice, scalar, op->dtype, device, status);
  if (f_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ x_data, f_data ], @[ o_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<ImageOp*>(kernel);                                 \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(RGBToHSV_Compute, RGBToHSV_ComputeImpl)
METAL_COMPUTE(AdjustContrast_Compute, AdjustContrast_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &ImageOp_Create, compute, &ImageOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
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

void RegisterMetalImage2Kernels() {
  Register("RGBToHSV", &RGBToHSV_Compute, TF_FLOAT, "MetalRGBToHSVFloat");
  Register("AdjustContrastv2", &AdjustContrast_Compute, TF_FLOAT,
           "MetalAdjustContrastv2Float");
}

}  // namespace metal
}  // namespace tensorflow
