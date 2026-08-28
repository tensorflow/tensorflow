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

// LowerBound, UpperBound, HistogramFixedWidth and TopK.
//
// The two bound searches are binary searches on the CPU, but a binary search
// is not what a GPU wants. Counting is: the lower bound of a value in a sorted
// row is exactly the number of entries strictly below it, and the upper bound
// the number at or below it. That turns a search into one broadcast comparison
// and a reduction, and it is where the strict/non-strict distinction between
// the two ops lives.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct SearchOp {
  TF_DataType dtype = TF_FLOAT;
  TF_DataType out_dtype = TF_INT32;
  int64_t k = 1;
  bool sorted = true;
  // ApproxTopK can be asked for the smallest rather than the largest.
  bool is_max_k = true;
};

void* SearchOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new SearchOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }
  TF_OpKernelConstruction_GetAttrType(ctx, "out_type", &op->out_dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->out_dtype = TF_INT32;
  }
  int64_t k = 1;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "k", &k, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->k = k;
  TF_Bool sorted = 1;
  TF_OpKernelConstruction_GetAttrBool(ctx, "sorted", &sorted, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->sorted = sorted != 0;
  TF_Bool is_max = 1;
  TF_OpKernelConstruction_GetAttrBool(ctx, "is_max_k", &is_max, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->is_max_k = is_max != 0;
  int32_t reduction_dimension = -1;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "reduction_dimension",
                                       &reduction_dimension, status);
  if (TF_GetCode(status) == TF_OK && reduction_dimension >= 0) {
    // The op allows any axis; only the last one is handled here, which is the
    // axis every caller of ApproxTopK actually uses. Rejecting the rest is
    // better than transposing behind the caller's back and calling it
    // approximate.
    op->k = -1;
  }
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void SearchOp_Delete(void* kernel) { delete static_cast<SearchOp*>(kernel); }

/*** LOWER AND UPPER BOUND ***/

template <bool kUpper>
void Bound_ComputeImpl(SearchOp* op, TF_OpKernelContext* ctx,
                       TF_Status* status) {
  ScopedTensor sorted, values;
  TF_GetInput(ctx, 0, sorted.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> sorted_shape = ShapeOf(sorted.get());
  const std::vector<int64_t> values_shape = ShapeOf(values.get());
  if (sorted_shape.size() != 2 || values_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: LowerBound expects rank-2 inputs.");
    return;
  }
  const int64_t batch = values_shape[0];
  const int64_t n_values = values_shape[1];
  const int64_t n_sorted = sorted_shape[1];

  const int64_t count = batch * n_values;
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->out_dtype, values_shape.data(), 2,
      static_cast<size_t>(count) * TF_DataTypeSize(op->out_dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype, out_mps;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;
  if (!MPSTypeFor(op->out_dtype, &out_mps, status)) return;

  std::string key = kUpper ? "UpperBound" : "LowerBound";
  AppendShapeToKey(sorted_shape, &key);
  AppendShapeToKey(values_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  key.append("/o").append(std::to_string(static_cast<int>(op->out_dtype)));

  // Compared as [batch, values, 1] against [batch, 1, sorted], which
  // broadcasts to the full pairwise grid without materialising it twice.
  const std::vector<int64_t> v_col = {batch, n_values, 1};
  const std::vector<int64_t> s_row = {batch, 1, n_sorted};

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* s = [g placeholderWithShape:MPSShape(sorted_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* v = [g placeholderWithShape:MPSShape(values_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* vc = [g reshapeTensor:v
                                    withShape:MPSShape(v_col)
                                         name:nil];
        MPSGraphTensor* sr = [g reshapeTensor:s
                                    withShape:MPSShape(s_row)
                                         name:nil];
        // Lower bound counts entries strictly below the value; upper bound
        // counts those at or below it. That single difference is the whole
        // distinction between the two ops.
        MPSGraphTensor* below =
            kUpper ? [g lessThanOrEqualToWithPrimaryTensor:sr
                                           secondaryTensor:vc
                                                      name:nil]
                   : [g lessThanWithPrimaryTensor:sr
                                  secondaryTensor:vc
                                             name:nil];
        MPSGraphTensor* counted =
            [g reductionSumWithTensor:[g castTensor:below
                                             toType:out_mps
                                               name:nil]
                                 axis:2
                                 name:nil];
        [out->inputs addObject:s];
        [out->inputs addObject:v];
        [out->outputs addObject:[g reshapeTensor:counted
                                       withShape:MPSShape(values_shape)
                                            name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* s_data =
      TensorDataForTensor(sorted.get(), op->dtype, device, status);
  if (s_data == nil) return;
  MPSGraphTensorData* v_data =
      TensorDataForTensor(values.get(), op->dtype, device, status);
  if (v_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->out_dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ s_data, v_data ], @[ o_data ], status);
}

/*** HISTOGRAM ***/

// HistogramFixedWidth buckets every value into nbins equal intervals over
// [range[0], range[1]] and counts them. The bucket index is arithmetic, and
// the count is a one-hot reduction, so no scatter is needed.
void Histogram_ComputeImpl(SearchOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor values, range_t, nbins_t;
  TF_GetInput(ctx, 0, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, range_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, nbins_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const void* nb = TF_TensorData(nbins_t.get());
  if (nb == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "Metal: nbins has no data.");
    return;
  }
  const int64_t nbins = *static_cast<const int32_t*>(nb);
  if (nbins < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "Metal: nbins must be positive.");
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(values.get());
  const int64_t total = ElementCount(in_shape);
  const std::vector<int64_t> out_shape = {nbins};

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->out_dtype, out_shape.data(), 1,
      static_cast<size_t>(nbins) * TF_DataTypeSize(op->out_dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (total == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype, out_mps;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;
  if (!MPSTypeFor(op->out_dtype, &out_mps, status)) return;

  std::string key = "HistogramFixedWidth";
  AppendShapeToKey(in_shape, &key);
  key.append("/n").append(std::to_string(nbins));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  key.append("/o").append(std::to_string(static_cast<int>(op->out_dtype)));
  const std::vector<int64_t> flat = {total};
  const std::vector<int64_t> range_shape = {2};

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* v = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* r = [g placeholderWithShape:MPSShape(range_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* lo = [g sliceTensor:r dimension:0 start:0 length:1
                                       name:nil];
        MPSGraphTensor* hi = [g sliceTensor:r dimension:0 start:1 length:1
                                       name:nil];
        MPSGraphTensor* span =
            [g subtractionWithPrimaryTensor:hi secondaryTensor:lo name:nil];
        MPSGraphTensor* flat_v =
            [g reshapeTensor:v withShape:MPSShape(flat) name:nil];
        MPSGraphTensor* scaled = [g
            multiplicationWithPrimaryTensor:
                [g divisionWithPrimaryTensor:
                       [g subtractionWithPrimaryTensor:flat_v
                                       secondaryTensor:lo
                                                  name:nil]
                             secondaryTensor:span
                                        name:nil]
                            secondaryTensor:[g constantWithScalar:(double)nbins
                                                         dataType:mps_dtype]
                                       name:nil];
        // Values on or past the top edge belong to the last bin, and values
        // below the bottom to the first, which is what the clamp does.
        MPSGraphTensor* idx = [g
            clampWithTensor:[g floorWithTensor:scaled name:nil]
             minValueTensor:[g constantWithScalar:0.0 dataType:mps_dtype]
             maxValueTensor:[g constantWithScalar:(double)(nbins - 1)
                                         dataType:mps_dtype]
                       name:nil];
        MPSGraphTensor* hot =
            [g oneHotWithIndicesTensor:[g castTensor:idx
                                              toType:MPSDataTypeInt32
                                                name:nil]
                                 depth:static_cast<NSUInteger>(nbins)
                                  axis:1
                              dataType:out_mps
                               onValue:1.0
                              offValue:0.0
                                  name:nil];
        MPSGraphTensor* counted =
            [g reductionSumWithTensor:hot axes:@[ @0 ] name:nil];
        [out->inputs addObject:v];
        [out->inputs addObject:r];
        [out->outputs addObject:[g reshapeTensor:counted
                                       withShape:MPSShape(out_shape)
                                            name:nil]];
      },
      status);
  if (cached == nullptr) return;

  BufferSlice range_slice;
  if (!SliceForTensor(range_t.get(), &range_slice, status)) return;
  MPSGraphTensorData* v_data =
      TensorDataForTensor(values.get(), op->dtype, device, status);
  if (v_data == nil) return;
  MPSGraphTensorData* r_data =
      TensorDataFor(range_slice, range_shape, op->dtype, device, status);
  if (r_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->out_dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ v_data, r_data ], @[ o_data ], status);
}

/*** TOP K, THE ATTRIBUTE FORM ***/

void TopK_ComputeImpl(SearchOp* op, TF_OpKernelContext* ctx,
                      TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  if (in_shape.empty() || op->k < 0 || op->k > in_shape.back()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "Metal: TopK k is out of range.");
    return;
  }
  std::vector<int64_t> out_shape = in_shape;
  out_shape.back() = op->k;
  const int64_t count = ElementCount(out_shape);

  ScopedTensor values, indices;
  values.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  indices.reset(TF_AllocateOutput(
      ctx, 1, TF_INT32, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_INT32), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "TopK";
  AppendShapeToKey(in_shape, &key);
  key.append("/k").append(std::to_string(op->k));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSUInteger k = static_cast<NSUInteger>(op->k);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        NSArray<MPSGraphTensor*>* r =
            [out->graph topKWithSourceTensor:x k:k name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:r[0]];
        [out->outputs addObject:[out->graph castTensor:r[1]
                                                toType:MPSDataTypeInt32
                                                  name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* v_data =
      TensorDataForTensor(values.get(), op->dtype, device, status);
  if (v_data == nil) return;
  MPSGraphTensorData* i_data =
      TensorDataForTensor(indices.get(), TF_INT32, device, status);
  if (i_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ v_data, i_data ], status);
}

/*** APPROXIMATE TOP K ***/

// ApproxTopK is allowed to trade recall for speed, and returning the exact
// answer satisfies any recall target it is given. That is what this does: the
// same sort the exact op uses, which on this hardware is fast enough that an
// approximation would buy little and cost a guarantee.
//
// The smallest-k form negates the input, takes the largest, and negates the
// values back, which leaves the indices already correct.
void ApproxTopK_ComputeImpl(SearchOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  if (op->k < 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: ApproxTopK reduces over the last dimension only.");
    return;
  }
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  if (in_shape.empty() || op->k > in_shape.back()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ApproxTopK k is out of range.");
    return;
  }
  std::vector<int64_t> out_shape = in_shape;
  out_shape.back() = op->k;
  const int64_t count = ElementCount(out_shape);

  ScopedTensor values, indices;
  values.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  indices.reset(TF_AllocateOutput(
      ctx, 1, TF_INT32, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_INT32), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "ApproxTopK";
  AppendShapeToKey(in_shape, &key);
  key.append("/k").append(std::to_string(op->k));
  key.append(op->is_max_k ? "/max" : "/min");
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSUInteger k = static_cast<NSUInteger>(op->k);
  const bool is_max_k = op->is_max_k;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* source =
            is_max_k ? x : [g negativeWithTensor:x name:nil];
        NSArray<MPSGraphTensor*>* r = [g topKWithSourceTensor:source
                                                            k:k
                                                         name:nil];
        MPSGraphTensor* v =
            is_max_k ? r[0] : [g negativeWithTensor:r[0] name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:v];
        [out->outputs addObject:[g castTensor:r[1]
                                       toType:MPSDataTypeInt32
                                         name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* v_data =
      TensorDataForTensor(values.get(), op->dtype, device, status);
  if (v_data == nil) return;
  MPSGraphTensorData* i_data =
      TensorDataForTensor(indices.get(), TF_INT32, device, status);
  if (i_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ v_data, i_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<SearchOp*>(kernel);                                \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(LowerBound_Compute, Bound_ComputeImpl<false>)
METAL_COMPUTE(UpperBound_Compute, Bound_ComputeImpl<true>)
METAL_COMPUTE(Histogram_Compute, Histogram_ComputeImpl)
METAL_COMPUTE(TopK_Compute, TopK_ComputeImpl)
METAL_COMPUTE(ApproxTopK_Compute, ApproxTopK_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, std::vector<const char*> host_args,
              const char* attr2 = nullptr, TF_DataType dtype2 = TF_INT32) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &SearchOp_Create, compute, &SearchOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
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

void RegisterMetalSearchKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kOutTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kOutSuffixes[] = {"Int32", "Int64"};

  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    Register("TopK", &TopK_Compute, t, "MetalTopK" + s, {});
    Register("ApproxTopK", &ApproxTopK_Compute, t,
             "MetalApproxTopK" + s, {});
    for (int j = 0; j < 2; ++j) {
      Register("LowerBound", &LowerBound_Compute, t,
               "MetalLowerBound" + s + kOutSuffixes[j], {}, "out_type",
               kOutTypes[j]);
      Register("UpperBound", &UpperBound_Compute, t,
               "MetalUpperBound" + s + kOutSuffixes[j], {}, "out_type",
               kOutTypes[j]);
      // nbins sizes the output, so it is read on the host.
      Register("HistogramFixedWidth", &Histogram_Compute, t,
               "MetalHistogramFixedWidth" + s + kOutSuffixes[j], {"nbins"},
               "dtype", kOutTypes[j]);
    }
  }
}

}  // namespace metal
}  // namespace tensorflow
