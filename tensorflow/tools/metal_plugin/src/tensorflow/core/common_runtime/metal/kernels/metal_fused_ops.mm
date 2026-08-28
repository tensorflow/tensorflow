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

// _FusedConv2D and _FusedMatMul.
//
// These are what the graph optimiser leaves behind when it folds a bias and an
// activation into the operation before them. Nothing about the arithmetic is
// new: the fusion exists so the intermediate result never reaches memory, and
// MPSGraph gets the same benefit from being handed the whole expression at
// once, since it plans the graph before running it.
//
// The fusions handled are a bias followed by an optional activation, which is
// what the optimiser produces for these two ops. A combination outside that
// set is refused rather than silently computed as something else.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

enum class Activation {
  kNone,
  kRelu,
  kRelu6,
  kElu,
  kLeakyRelu,
  kSigmoid,
  kTanh,
  kGeluExact,
  kGeluApproximate,
};

struct FusedOp {
  TF_DataType dtype = TF_FLOAT;
  bool has_bias = false;
  Activation activation = Activation::kNone;
  float leaky_alpha = 0.2f;
  // Convolution only.
  SpatialParams spatial;
  // Matrix multiply only.
  bool transpose_a = false;
  bool transpose_b = false;
  bool valid = false;
};

bool ParseFusion(TF_OpKernelConstruction* ctx, FusedOp* op,
                 TF_Status* status) {
  int32_t list_size = 0;
  int32_t total = 0;
  TF_OpKernelConstruction_GetAttrSize(ctx, "fused_ops", &list_size, &total,
                                      status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->valid = true;
    return true;
  }
  if (list_size == 0) {
    op->valid = true;
    return true;
  }
  // The strings arrive packed: one buffer of bytes plus a vector of pointers
  // into it.
  std::vector<char> storage(static_cast<size_t>(std::max(total, 1)) + 1, 0);
  std::vector<char*> pointers(static_cast<size_t>(list_size), nullptr);
  std::vector<size_t> sizes(static_cast<size_t>(list_size), 0);
  TF_OpKernelConstruction_GetAttrStringList(
      ctx, "fused_ops", pointers.data(), sizes.data(), list_size,
      storage.data(), storage.size(), status);
  if (TF_GetCode(status) != TF_OK) return false;

  for (int i = 0; i < list_size; ++i) {
    const std::string name(pointers[i], sizes[i]);
    if (name == "BiasAdd") {
      op->has_bias = true;
    } else if (name == "Relu") {
      op->activation = Activation::kRelu;
    } else if (name == "Relu6") {
      op->activation = Activation::kRelu6;
    } else if (name == "Elu") {
      op->activation = Activation::kElu;
    } else if (name == "LeakyRelu") {
      op->activation = Activation::kLeakyRelu;
    } else if (name == "Sigmoid") {
      op->activation = Activation::kSigmoid;
    } else if (name == "Tanh") {
      op->activation = Activation::kTanh;
    } else if (name == "GeluExact") {
      op->activation = Activation::kGeluExact;
    } else if (name == "GeluApproximate") {
      op->activation = Activation::kGeluApproximate;
    } else {
      // Anything else, a folded batch normalisation among them, would need
      // more inputs than this kernel reads. Saying so is better than
      // computing a different expression under the same name.
      TF_SetStatus(
          status, TF_UNIMPLEMENTED,
          ("Metal: this backend fuses a bias and an activation only; " + name +
           " is not handled.")
              .c_str());
      return false;
    }
  }
  op->valid = true;
  return true;
}

void* FusedConvOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new FusedOp();
  if (!ReadSpatialParams(ctx, /*want_dilations=*/true, &op->spatial, status)) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  op->dtype = op->spatial.dtype;
  float alpha = 0.2f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "leakyrelu_alpha", &alpha, status);
  if (TF_GetCode(status) == TF_OK) op->leaky_alpha = alpha;
  TF_SetStatus(status, TF_OK, "");
  if (!ParseFusion(ctx, op, status)) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void* FusedMatMulOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new FusedOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_a", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->transpose_a = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_b", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->transpose_b = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  float alpha = 0.2f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "leakyrelu_alpha", &alpha, status);
  if (TF_GetCode(status) == TF_OK) op->leaky_alpha = alpha;
  TF_SetStatus(status, TF_OK, "");
  if (!ParseFusion(ctx, op, status)) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void FusedOp_Delete(void* kernel) { delete static_cast<FusedOp*>(kernel); }

MPSGraphTensor* ApplyActivation(MPSGraph* g, MPSGraphTensor* x,
                                Activation activation, float alpha) {
  MPSGraphTensor* zero =
      [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
  switch (activation) {
    case Activation::kNone:
      return x;
    case Activation::kRelu:
      return [g maximumWithPrimaryTensor:x secondaryTensor:zero name:nil];
    case Activation::kRelu6:
      return [g clampWithTensor:x
                 minValueTensor:zero
                 maxValueTensor:[g constantWithScalar:6.0
                                             dataType:MPSDataTypeFloat32]
                           name:nil];
    case Activation::kElu: {
      MPSGraphTensor* one =
          [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
      MPSGraphTensor* expm1 =
          [g subtractionWithPrimaryTensor:[g exponentWithTensor:x name:nil]
                          secondaryTensor:one
                                     name:nil];
      return [g selectWithPredicateTensor:[g greaterThanWithPrimaryTensor:x
                                                          secondaryTensor:zero
                                                                     name:nil]
                      truePredicateTensor:x
                     falsePredicateTensor:expm1
                                     name:nil];
    }
    case Activation::kLeakyRelu: {
      MPSGraphTensor* scaled = [g
          multiplicationWithPrimaryTensor:x
                          secondaryTensor:[g constantWithScalar:alpha
                                                      dataType:
                                                          MPSDataTypeFloat32]
                                     name:nil];
      return [g selectWithPredicateTensor:[g greaterThanWithPrimaryTensor:x
                                                          secondaryTensor:zero
                                                                     name:nil]
                      truePredicateTensor:x
                     falsePredicateTensor:scaled
                                     name:nil];
    }
    case Activation::kSigmoid:
      return [g sigmoidWithTensor:x name:nil];
    case Activation::kTanh:
      return [g tanhWithTensor:x name:nil];
    case Activation::kGeluExact: {
      // x * 0.5 * (1 + erf(x / sqrt(2))).
      MPSGraphTensor* scaled =
          [g divisionWithPrimaryTensor:x
                       secondaryTensor:[g constantWithScalar:1.4142135623730951
                                                    dataType:
                                                        MPSDataTypeFloat32]
                                  name:nil];
      MPSGraphTensor* cdf = [g
          multiplicationWithPrimaryTensor:
              [g additionWithPrimaryTensor:[g erfWithTensor:scaled name:nil]
                           secondaryTensor:[g constantWithScalar:1.0
                                                       dataType:
                                                           MPSDataTypeFloat32]
                                      name:nil]
                          secondaryTensor:[g constantWithScalar:0.5
                                                       dataType:
                                                           MPSDataTypeFloat32]
                                     name:nil];
      return [g multiplicationWithPrimaryTensor:x
                                secondaryTensor:cdf
                                           name:nil];
    }
    case Activation::kGeluApproximate: {
      // The tanh form, which is what the optimiser means by approximate.
      MPSGraphTensor* cube =
          [g multiplicationWithPrimaryTensor:[g multiplicationWithPrimaryTensor:x
                                                                secondaryTensor:x
                                                                           name:nil]
                             secondaryTensor:x
                                        name:nil];
      MPSGraphTensor* inner = [g
          multiplicationWithPrimaryTensor:
              [g additionWithPrimaryTensor:x
                           secondaryTensor:
                               [g multiplicationWithPrimaryTensor:cube
                                                  secondaryTensor:
                                                      [g constantWithScalar:
                                                             0.044715
                                                                   dataType:
                                                                       MPSDataTypeFloat32]
                                                             name:nil]
                                      name:nil]
                          secondaryTensor:[g constantWithScalar:0.7978845608028654
                                                       dataType:
                                                           MPSDataTypeFloat32]
                                     name:nil];
      MPSGraphTensor* cdf = [g
          multiplicationWithPrimaryTensor:
              [g additionWithPrimaryTensor:[g tanhWithTensor:inner name:nil]
                           secondaryTensor:[g constantWithScalar:1.0
                                                       dataType:
                                                           MPSDataTypeFloat32]
                                      name:nil]
                          secondaryTensor:[g constantWithScalar:0.5
                                                       dataType:
                                                           MPSDataTypeFloat32]
                                     name:nil];
      return [g multiplicationWithPrimaryTensor:x
                                secondaryTensor:cdf
                                           name:nil];
    }
  }
  return x;
}

void AppendFusionToKey(const FusedOp& op, std::string* key) {
  key->append(op.has_bias ? "/bias" : "/plain");
  key->append("/a").append(
      std::to_string(static_cast<int>(op.activation)));
  key->append("/l").append(std::to_string(op.leaky_alpha));
  key->append("/t").append(std::to_string(static_cast<int>(op.dtype)));
}

void FusedConv_ComputeImpl(FusedOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor input, filter, bias;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, filter.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (op->has_bias) {
    TF_GetInput(ctx, 2, bias.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const std::vector<int64_t> filter_shape = ShapeOf(filter.get());
  if (in_shape.size() != 4 || filter_shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a fused convolution expects rank-4 inputs.");
    return;
  }
  const bool nhwc = op->spatial.nhwc;
  const int h_axis = nhwc ? 1 : 2;
  const int w_axis = nhwc ? 2 : 3;
  const int c_axis = nhwc ? 3 : 1;
  const int64_t out_channels = filter_shape[3];

  std::vector<int64_t> out_shape = in_shape;
  for (int i = 0; i < 2; ++i) {
    const int axis = i == 0 ? h_axis : w_axis;
    const int stride = i == 0 ? op->spatial.stride_h : op->spatial.stride_w;
    const int dilation =
        i == 0 ? op->spatial.dilation_h : op->spatial.dilation_w;
    const int64_t window = (filter_shape[i] - 1) * dilation + 1;
    out_shape[axis] =
        op->spatial.same_padding
            ? (in_shape[axis] + stride - 1) / stride
            : (in_shape[axis] < window ? 0
                                       : (in_shape[axis] - window) / stride + 1);
  }
  out_shape[c_axis] = out_channels;

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  const std::vector<int64_t> bias_shape = {out_channels};
  std::string key = "FusedConv2D";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(filter_shape, &key);
  key.append("/s").append(std::to_string(op->spatial.stride_h)).push_back('x');
  key.append(std::to_string(op->spatial.stride_w));
  key.append("/d").append(std::to_string(op->spatial.dilation_h)).push_back('x');
  key.append(std::to_string(op->spatial.dilation_w));
  key.append(op->spatial.same_padding ? "/SAME" : "/VALID");
  key.append(nhwc ? "/nhwc" : "/nchw");
  AppendFusionToKey(*op, &key);

  const FusedOp captured = *op;
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* w = [g placeholderWithShape:MPSShape(filter_shape)
                                           dataType:mps_dtype
                                               name:nil];
        [out->inputs addObject:x];
        [out->inputs addObject:w];
        MPSGraphConvolution2DOpDescriptor* descriptor =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:static_cast<NSUInteger>(
                                            captured.spatial.stride_w)
                              strideInY:static_cast<NSUInteger>(
                                            captured.spatial.stride_h)
                        dilationRateInX:static_cast<NSUInteger>(
                                            captured.spatial.dilation_w)
                        dilationRateInY:static_cast<NSUInteger>(
                                            captured.spatial.dilation_h)
                                 groups:1
                           paddingStyle:captured.spatial.same_padding
                                            ? MPSGraphPaddingStyleTF_SAME
                                            : MPSGraphPaddingStyleTF_VALID
                             dataLayout:nhwc
                                 ? MPSGraphTensorNamedDataLayoutNHWC
                                 : MPSGraphTensorNamedDataLayoutNCHW
                          weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
        MPSGraphTensor* result = [g convolution2DWithSourceTensor:x
                                                    weightsTensor:w
                                                       descriptor:descriptor
                                                             name:nil];
        if (captured.has_bias) {
          MPSGraphTensor* b = [g placeholderWithShape:MPSShape(bias_shape)
                                             dataType:mps_dtype
                                                 name:nil];
          [out->inputs addObject:b];
          // The bias is one value per output channel, broadcast along the
          // channel axis wherever that axis happens to be.
          NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
          for (int i = 0; i < 4; ++i) {
            [shape addObject:@(i == c_axis
                                   ? static_cast<NSInteger>(out_channels)
                                   : 1)];
          }
          result = [g additionWithPrimaryTensor:result
                                secondaryTensor:[g reshapeTensor:b
                                                       withShape:shape
                                                            name:nil]
                                           name:nil];
        }
        [out->outputs addObject:ApplyActivation(g, result,
                                                captured.activation,
                                                captured.leaky_alpha)];
      },
      status);
  if (cached == nullptr) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* w_data =
      TensorDataForTensor(filter.get(), op->dtype, device, status);
  if (w_data == nil) return;
  [feeds addObject:x_data];
  [feeds addObject:w_data];
  if (op->has_bias) {
    MPSGraphTensorData* b_data =
        TensorDataForTensor(bias.get(), op->dtype, device, status);
    if (b_data == nil) return;
    [feeds addObject:b_data];
  }
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, feeds, @[ o_data ], status);
}

void FusedMatMul_ComputeImpl(FusedOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor lhs, rhs, bias;
  TF_GetInput(ctx, 0, lhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, rhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (op->has_bias) {
    TF_GetInput(ctx, 2, bias.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  const std::vector<int64_t> a_shape = ShapeOf(lhs.get());
  const std::vector<int64_t> b_shape = ShapeOf(rhs.get());
  if (a_shape.size() != 2 || b_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a fused matrix multiply expects rank-2 inputs.");
    return;
  }
  const int64_t rows = op->transpose_a ? a_shape[1] : a_shape[0];
  const int64_t cols = op->transpose_b ? b_shape[0] : b_shape[1];
  const std::vector<int64_t> out_shape = {rows, cols};
  const std::vector<int64_t> bias_shape = {cols};

  const int64_t count = rows * cols;
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), 2,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "FusedMatMul";
  AppendShapeToKey(a_shape, &key);
  AppendShapeToKey(b_shape, &key);
  key.append(op->transpose_a ? "/ta" : "/na");
  key.append(op->transpose_b ? "/tb" : "/nb");
  AppendFusionToKey(*op, &key);

  const FusedOp captured = *op;
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* a = [g placeholderWithShape:MPSShape(a_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* b = [g placeholderWithShape:MPSShape(b_shape)
                                           dataType:mps_dtype
                                               name:nil];
        [out->inputs addObject:a];
        [out->inputs addObject:b];
        MPSGraphTensor* left =
            captured.transpose_a
                ? [g transposeTensor:a dimension:0 withDimension:1 name:nil]
                : a;
        MPSGraphTensor* right =
            captured.transpose_b
                ? [g transposeTensor:b dimension:0 withDimension:1 name:nil]
                : b;
        MPSGraphTensor* result =
            [g matrixMultiplicationWithPrimaryTensor:left
                                     secondaryTensor:right
                                                name:nil];
        if (captured.has_bias) {
          MPSGraphTensor* bias_tensor =
              [g placeholderWithShape:MPSShape(bias_shape)
                             dataType:mps_dtype
                                 name:nil];
          [out->inputs addObject:bias_tensor];
          result = [g
              additionWithPrimaryTensor:result
                        secondaryTensor:
                            [g reshapeTensor:bias_tensor
                                   withShape:@[
                                     @1, @(static_cast<NSInteger>(cols))
                                   ]
                                        name:nil]
                                   name:nil];
        }
        [out->outputs addObject:ApplyActivation(g, result,
                                                captured.activation,
                                                captured.leaky_alpha)];
      },
      status);
  if (cached == nullptr) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  MPSGraphTensorData* a_data =
      TensorDataForTensor(lhs.get(), op->dtype, device, status);
  if (a_data == nil) return;
  MPSGraphTensorData* b_data =
      TensorDataForTensor(rhs.get(), op->dtype, device, status);
  if (b_data == nil) return;
  [feeds addObject:a_data];
  [feeds addObject:b_data];
  if (op->has_bias) {
    MPSGraphTensorData* bias_data =
        TensorDataForTensor(bias.get(), op->dtype, device, status);
    if (bias_data == nil) return;
    [feeds addObject:bias_data];
  }
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, feeds, @[ o_data ], status);
}

#define METAL_FUSED_COMPUTE(NAME, IMPL)                                     \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<FusedOp*>(kernel);                               \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a fused kernel has no state.");                  \
    } else {                                                                \
      IMPL(op, ctx, status);                                                \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_FUSED_COMPUTE(FusedConv_Compute, FusedConv_ComputeImpl)
METAL_FUSED_COMPUTE(FusedMatMul_Compute, FusedMatMul_ComputeImpl)

#undef METAL_FUSED_COMPUTE

void Register(const char* op_name, void* (*create)(TF_OpKernelConstruction*),
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, create, compute, &FusedOp_Delete);
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

void RegisterMetalFusedKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    Register("_FusedConv2D", &FusedConvOp_Create, &FusedConv_Compute,
             kDTypes[i], std::string("Metal_FusedConv2D") + kSuffixes[i]);
    Register("_FusedMatMul", &FusedMatMulOp_Create, &FusedMatMul_Compute,
             kDTypes[i], std::string("Metal_FusedMatMul") + kSuffixes[i]);
  }
}

}  // namespace metal
}  // namespace tensorflow
