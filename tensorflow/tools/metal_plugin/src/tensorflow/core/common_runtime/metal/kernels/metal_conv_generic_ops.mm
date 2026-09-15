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

// Conv, the rank-agnostic convolution.
//
// It is the same operation Conv2D and Conv3D perform; what differs is that the
// number of spatial dimensions comes from the input's rank rather than from
// the op's name. One, two and three spatial dimensions are handled, the first
// by giving the input a height of one and running it as a two-dimensional
// convolution, which is exactly what a one-dimensional convolution is.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct ConvOp {
  TF_DataType dtype = TF_FLOAT;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  bool same_padding = false;
  bool channels_last = true;
};

void* ConvOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ConvOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }

  int32_t groups = 1;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "groups", &groups, status);
  TF_SetStatus(status, TF_OK, "");
  int32_t batch_dims = 1;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "batch_dims", &batch_dims, status);
  TF_SetStatus(status, TF_OK, "");
  if (groups != 1 || batch_dims != 1) {
    // The CPU kernel refuses grouped convolutions too, and more than one batch
    // dimension would have to be flattened before MPS could see it.
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: Conv supports one batch dimension and one group.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }

  char format[24] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", format,
                                        sizeof(format) - 1, status);
  if (TF_GetCode(status) == TF_OK && format[0] != '\0') {
    op->channels_last = std::strcmp(format, "CHANNELS_FIRST") != 0;
  }
  TF_SetStatus(status, TF_OK, "");

  char padding[16] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", padding,
                                        sizeof(padding) - 1, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  if (std::strcmp(padding, "SAME") == 0) {
    op->same_padding = true;
  } else if (std::strcmp(padding, "VALID") != 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: Conv supports SAME and VALID padding only.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }

  // strides and dilations may be given per spatial dimension or per tensor
  // dimension; both are read here and reconciled once the rank is known.
  struct { const char* name; std::vector<int64_t>* out; } lists[] = {
      {"strides", &op->strides},
      {"dilations", &op->dilations},
  };
  for (auto& list : lists) {
    int32_t list_size = 0;
    int32_t total = 0;
    TF_OpKernelConstruction_GetAttrSize(ctx, list.name, &list_size, &total,
                                        status);
    if (TF_GetCode(status) == TF_OK && list_size > 0) {
      std::vector<int64_t> values(static_cast<size_t>(list_size), 1);
      TF_OpKernelConstruction_GetAttrInt64List(ctx, list.name, values.data(),
                                               list_size, status);
      if (TF_GetCode(status) == TF_OK) *list.out = values;
    }
    TF_SetStatus(status, TF_OK, "");
  }

  TF_DeleteStatus(status);
  return op;
}

void ConvOp_Delete(void* kernel) { delete static_cast<ConvOp*>(kernel); }

// Picks the per-spatial-dimension entries out of an attribute that may have
// been given either way round.
bool SpatialEntries(const std::vector<int64_t>& values, int spatial,
                    bool channels_last, std::vector<int64_t>* out,
                    TF_Status* status) {
  out->assign(static_cast<size_t>(spatial), 1);
  if (values.empty()) return true;
  if (static_cast<int>(values.size()) == spatial) {
    *out = values;
    return true;
  }
  if (static_cast<int>(values.size()) == spatial + 2) {
    const int first = channels_last ? 1 : 2;
    for (int i = 0; i < spatial; ++i) (*out)[i] = values[first + i];
    return true;
  }
  TF_SetStatus(status, TF_INVALID_ARGUMENT,
               "Metal: Conv strides and dilations must have one entry per "
               "spatial dimension, or one per tensor dimension.");
  return false;
}

void Conv_ComputeImpl(ConvOp* op, TF_OpKernelContext* ctx,
                      TF_Status* status) {
  ScopedTensor input, filter;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, filter.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const std::vector<int64_t> filter_shape = ShapeOf(filter.get());
  const int spatial = static_cast<int>(in_shape.size()) - 2;
  if (spatial < 1 || spatial > 3) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: Conv handles one, two or three spatial dimensions.");
    return;
  }
  if (static_cast<int>(filter_shape.size()) != spatial + 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the filter's rank does not match the input's.");
    return;
  }

  std::vector<int64_t> strides, dilations;
  if (!SpatialEntries(op->strides, spatial, op->channels_last, &strides,
                      status)) {
    return;
  }
  if (!SpatialEntries(op->dilations, spatial, op->channels_last, &dilations,
                      status)) {
    return;
  }

  // A one-dimensional convolution is a two-dimensional one over a single row,
  // so the tensors gain a unit height and the whole thing goes through the
  // same path.
  const bool as_2d = spatial <= 2;
  std::vector<int64_t> graph_in = in_shape;
  std::vector<int64_t> graph_filter = filter_shape;
  std::vector<int64_t> use_strides = strides;
  std::vector<int64_t> use_dilations = dilations;
  if (spatial == 1) {
    const int axis = op->channels_last ? 1 : 2;
    graph_in.insert(graph_in.begin() + axis, 1);
    graph_filter.insert(graph_filter.begin(), 1);
    use_strides.insert(use_strides.begin(), 1);
    use_dilations.insert(use_dilations.begin(), 1);
  }

  // The output's spatial extent follows TensorFlow's padding rules, and the
  // channel count is the filter's last dimension.
  std::vector<int64_t> out_shape = in_shape;
  const int first_spatial = op->channels_last ? 1 : 2;
  for (int i = 0; i < spatial; ++i) {
    const int64_t in = in_shape[first_spatial + i];
    const int64_t window = (filter_shape[i] - 1) * dilations[i] + 1;
    out_shape[first_spatial + i] =
        op->same_padding ? (in + strides[i] - 1) / strides[i]
                         : (in < window ? 0 : (in - window) / strides[i] + 1);
  }
  const int channel_axis =
      op->channels_last ? static_cast<int>(out_shape.size()) - 1 : 1;
  out_shape[channel_axis] = filter_shape[filter_shape.size() - 1];

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  std::vector<int64_t> graph_out = out_shape;
  if (spatial == 1) {
    graph_out.insert(graph_out.begin() + (op->channels_last ? 1 : 2), 1);
  }

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Conv";
  AppendShapeToKey(graph_in, &key);
  AppendShapeToKey(graph_filter, &key);
  for (int64_t v : use_strides) key.append("/s").append(std::to_string(v));
  for (int64_t v : use_dilations) key.append("/d").append(std::to_string(v));
  key.append(op->same_padding ? "/SAME" : "/VALID");
  key.append(op->channels_last ? "/last" : "/first");
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const bool same_padding = op->same_padding;
  const bool channels_last = op->channels_last;
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(graph_in)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* w = [g placeholderWithShape:MPSShape(graph_filter)
                                           dataType:mps_dtype
                                               name:nil];
        const MPSGraphPaddingStyle padding =
            same_padding ? MPSGraphPaddingStyleTF_SAME
                         : MPSGraphPaddingStyleTF_VALID;
        MPSGraphTensor* result = nil;
        if (as_2d) {
          MPSGraphConvolution2DOpDescriptor* descriptor =
              [MPSGraphConvolution2DOpDescriptor
                  descriptorWithStrideInX:static_cast<NSUInteger>(
                                              use_strides[1])
                                strideInY:static_cast<NSUInteger>(
                                              use_strides[0])
                          dilationRateInX:static_cast<NSUInteger>(
                                              use_dilations[1])
                          dilationRateInY:static_cast<NSUInteger>(
                                              use_dilations[0])
                                   groups:1
                             paddingStyle:padding
                               dataLayout:channels_last
                                   ? MPSGraphTensorNamedDataLayoutNHWC
                                   : MPSGraphTensorNamedDataLayoutNCHW
                            weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
          result = [g convolution2DWithSourceTensor:x
                                      weightsTensor:w
                                         descriptor:descriptor
                                               name:nil];
        } else {
          MPSGraphConvolution3DOpDescriptor* descriptor =
              [MPSGraphConvolution3DOpDescriptor
                  descriptorWithStrideInX:static_cast<NSUInteger>(
                                              use_strides[2])
                                strideInY:static_cast<NSUInteger>(
                                              use_strides[1])
                                strideInZ:static_cast<NSUInteger>(
                                              use_strides[0])
                          dilationRateInX:static_cast<NSUInteger>(
                                              use_dilations[2])
                          dilationRateInY:static_cast<NSUInteger>(
                                              use_dilations[1])
                          dilationRateInZ:static_cast<NSUInteger>(
                                              use_dilations[0])
                                   groups:1
                             paddingStyle:padding
                               dataLayout:channels_last
                                   ? MPSGraphTensorNamedDataLayoutNDHWC
                                   : MPSGraphTensorNamedDataLayoutNCDHW
                            weightsLayout:MPSGraphTensorNamedDataLayoutDHWIO];
          result = [g convolution3DWithSourceTensor:x
                                      weightsTensor:w
                                         descriptor:descriptor
                                               name:nil];
        }
        [out->inputs addObject:x];
        [out->inputs addObject:w];
        [out->outputs addObject:result];
      },
      status);
  if (cached == nullptr) return;

  BufferSlice in_slice, filter_slice, out_slice;
  if (!SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(filter.get(), &filter_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  // The unit height exists only in the graph's view of the tensors; the
  // storage is untouched, so the reshape costs nothing.
  MPSGraphTensorData* x_data =
      TensorDataFor(in_slice, graph_in, op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* w_data =
      TensorDataFor(filter_slice, graph_filter, op->dtype, device, status);
  if (w_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataFor(out_slice, graph_out, op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ x_data, w_data ], @[ o_data ], status);
}

void Conv_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<ConvOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: Conv has no state.");
  } else {
    Conv_ComputeImpl(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &ConvOp_Create, compute, &ConvOp_Delete);
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

void RegisterMetalGenericConvKernels() {
  Register("Conv", &Conv_Compute, TF_FLOAT, "MetalConvFloat");
  Register("Conv", &Conv_Compute, TF_HALF, "MetalConvHalf");
}

}  // namespace metal
}  // namespace tensorflow
