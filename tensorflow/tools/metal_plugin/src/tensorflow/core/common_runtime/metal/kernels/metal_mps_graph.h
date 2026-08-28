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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_MPS_GRAPH_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_MPS_GRAPH_H_

// Objective-C++ only.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {

// Bridge from this backend's tensors to MPSGraph.
//
// MPSGraph is what makes convolutions, pooling, normalisation, reductions and
// their gradients tractable: Apple ships kernels tuned per GPU generation for
// all of them, and reimplementing that by hand would be both enormous and
// slower.
//
// The reason it is usable at all is MPSNDArray's
// initWithBuffer:offset:descriptor:, which aliases an existing MTLBuffer at a
// byte offset. Core's BFC allocator places tensors at arbitrary offsets inside
// a shared allocation, so an interface that assumed a tensor started at the
// beginning of its buffer would force a copy of every operand. Going through
// MPSNDArray keeps the whole path zero-copy, in both directions: results are
// written straight into the output tensor's storage.

// Maps a TF dtype to its MPSGraph equivalent. Fails `status` for dtypes this
// backend does not handle.
bool MPSTypeFor(TF_DataType dtype, MPSDataType* out, TF_Status* status);

// Shape in the form MPSGraph expects. Autoreleased.
NSArray<NSNumber*>* MPSShape(const std::vector<int64_t>& shape);

// Zero-copy view of a tensor's storage as MPSGraph input or output data.
// Autoreleased; nil on failure, with `status` set.
MPSGraphTensorData* TensorDataFor(const BufferSlice& slice,
                                  const std::vector<int64_t>& shape,
                                  TF_DataType dtype, id<MTLDevice> device,
                                  TF_Status* status);

// Convenience: resolve a TF_Tensor and wrap it in one step.
MPSGraphTensorData* TensorDataForTensor(TF_Tensor* tensor, TF_DataType dtype,
                                        id<MTLDevice> device,
                                        TF_Status* status);

// A compiled graph plus the placeholders to feed and the tensors to read.
//
// Building an MPSGraph is expensive relative to running one, and a kernel runs
// the same shape over and over across training steps, so graphs are cached by
// a key the caller derives from everything that changes their structure
// (shapes, dtype, strides, padding, and so on).
struct CachedGraph {
  MPSGraph* graph = nil;
  // Placeholders, in the order the caller feeds them.
  NSMutableArray<MPSGraphTensor*>* inputs = nil;
  // Results, in the order the caller supplies output storage.
  NSMutableArray<MPSGraphTensor*>* outputs = nil;
};

// Returns the cached graph for `key`, building it with `builder` on first use.
// `builder` receives a fresh graph and must fill in `inputs` and `outputs`.
// Returns nullptr with `status` set if the builder failed.
//
// The returned pointer is owned by the cache and lives for the process.
const CachedGraph* LookupOrBuildGraph(const std::string& key,
                                      void (^builder)(CachedGraph* out),
                                      TF_Status* status);

// Encodes `cached` onto `stream`, feeding `input_data` and writing results
// directly into `output_data`.
//
// Handles the interaction between MPSGraph and this backend's stream ordering:
// MPSGraph may call commitAndContinue and replace the underlying command
// buffer, so the ordering signal is attached to whichever buffer is live at
// the end rather than the one we started with.
bool RunGraph(SP_Stream stream, const CachedGraph& cached,
              NSArray<MPSGraphTensorData*>* input_data,
              NSArray<MPSGraphTensorData*>* output_data, TF_Status* status);

// Appends a shape to a cache key, for callers building keys by hand.
void AppendShapeToKey(const std::vector<int64_t>& shape, std::string* key);

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_MPS_GRAPH_H_
