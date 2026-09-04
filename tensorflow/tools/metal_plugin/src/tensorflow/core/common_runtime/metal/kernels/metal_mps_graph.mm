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

#include "tensorflow/core/common_runtime/metal/kernels/metal_mps_graph.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/metal_profiler.h"

namespace tensorflow {
namespace metal {
namespace {

// Cache of built graphs, keyed by the caller's configuration string.
//
// Entries are never evicted. A kernel builds one graph per distinct shape and
// configuration, and a training loop repeats the same handful of shapes for
// its whole run, so the cache reaches a small fixed size and stays there. An
// eviction policy would cost more than it saves.
class GraphCache {
 public:
  static GraphCache& Global() {
    static GraphCache* cache = new GraphCache();
    return *cache;
  }

  const CachedGraph* LookupOrBuild(const std::string& key,
                                   void (^builder)(CachedGraph* out),
                                   TF_Status* status) {
    {
      absl::MutexLock lock(&mu_);
      auto it = entries_.find(key);
      if (it != entries_.end()) return it->second;
    }

    // Built outside the lock: graph construction is slow and calls into MPS,
    // and two threads racing on the same key simply build twice, with one
    // result discarded. That is cheaper than serialising every first use.
    auto* built = new CachedGraph();
    built->graph = [[MPSGraph alloc] init];
    built->inputs = [[NSMutableArray alloc] init];
    built->outputs = [[NSMutableArray alloc] init];
    builder(built);

    // Every result goes through an identity before it is published.
    //
    // MPSGraph can answer for a tensor that is a pure view of an input, a
    // transpose or a reshape among them, without doing any work: the value is
    // already in memory, just read differently. encodeToCommandBuffer is then
    // asked to deliver that view into a caller-supplied MPSGraphTensorData and
    // encodes nothing at all, so the output tensor keeps whatever it was
    // allocated with, which is zeros. Transpose returned zeros for every
    // permutation but the identity, and the identity was right for exactly the
    // reason the others were wrong.
    //
    // An identity gives the result a producing operation, so there is
    // something to encode into the destination. When the result already had
    // one, this is folded away and costs nothing.
    for (NSUInteger i = 0; i < [built->outputs count]; ++i) {
      built->outputs[i] = [built->graph identityWithTensor:built->outputs[i]
                                                      name:nil];
    }

    if (built->graph == nil || [built->inputs count] == 0 ||
        [built->outputs count] == 0) {
      TF_SetStatus(status, TF_INTERNAL,
                   ("Metal: failed to build an MPSGraph for " + key).c_str());
      [built->graph release];
      [built->inputs release];
      [built->outputs release];
      delete built;
      return nullptr;
    }

    absl::MutexLock lock(&mu_);
    auto [it, inserted] = entries_.emplace(key, built);
    if (!inserted) {
      // Lost the race; keep the entry already published so every caller sees
      // the same graph object.
      [built->graph release];
      [built->inputs release];
      [built->outputs release];
      delete built;
    }
    return it->second;
  }

 private:
  GraphCache() = default;

  absl::Mutex mu_;
  absl::flat_hash_map<std::string, CachedGraph*> entries_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

bool MPSTypeFor(TF_DataType dtype, MPSDataType* out, TF_Status* status) {
  switch (dtype) {
    case TF_FLOAT:
      *out = MPSDataTypeFloat32;
      return true;
    case TF_HALF:
      *out = MPSDataTypeFloat16;
      return true;
    case TF_BFLOAT16:
      *out = MPSDataTypeBFloat16;
      return true;
    case TF_INT32:
      *out = MPSDataTypeInt32;
      return true;
    case TF_INT64:
      *out = MPSDataTypeInt64;
      return true;
    case TF_BOOL:
      *out = MPSDataTypeBool;
      return true;
    case TF_UINT8:
      *out = MPSDataTypeUInt8;
      return true;
    case TF_INT8:
      *out = MPSDataTypeInt8;
      return true;
    default:
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   ("Metal: dtype " + std::to_string(static_cast<int>(dtype)) +
                    " is not supported by the MPSGraph path.")
                       .c_str());
      return false;
  }
}

NSArray<NSNumber*>* MPSShape(const std::vector<int64_t>& shape) {
  NSMutableArray<NSNumber*>* result =
      [NSMutableArray arrayWithCapacity:shape.size()];
  for (int64_t dim : shape) {
    [result addObject:@(static_cast<NSInteger>(dim))];
  }
  return result;
}

MPSGraphTensorData* TensorDataFor(const BufferSlice& slice,
                                  const std::vector<int64_t>& shape,
                                  TF_DataType dtype, id<MTLDevice> device,
                                  TF_Status* status) {
  MPSDataType mps_dtype;
  if (!MPSTypeFor(dtype, &mps_dtype, status)) return nil;

  MPSNDArrayDescriptor* descriptor =
      [MPSNDArrayDescriptor descriptorWithDataType:mps_dtype
                                             shape:MPSShape(shape)];
  if (descriptor == nil) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: could not describe a tensor for MPSGraph.");
    return nil;
  }

  // By default MPSNDArray rounds the innermost dimension's row up to a
  // multiple of 16 bytes and then rejects a tightly packed buffer as too
  // small, with an assertion that takes the process down rather than an error
  // that can be reported. TensorFlow tensors are always tightly packed, and
  // three float32 channels is 12 bytes, so the very common RGB case would hit
  // this on the first convolution. preferPackedRows tells MPS to use the tight
  // stride, which is what keeps the alias valid and the path zero-copy.
  const size_t element_size = TF_DataTypeSize(dtype);
  const size_t innermost = shape.empty() ? 1 : static_cast<size_t>(shape.back());
  const size_t row_bytes = innermost * element_size;
  if ([descriptor respondsToSelector:@selector(setPreferPackedRows:)]) {
    descriptor.preferPackedRows = YES;
  } else if (row_bytes % 16 != 0) {
    // preferPackedRows arrived in macOS 15. Older systems would assert inside
    // MPS, so refuse here with something a user can act on.
    TF_SetStatus(
        status, TF_UNIMPLEMENTED,
        ("Metal: a tensor whose innermost dimension is " +
         std::to_string(innermost) + " elements (" + std::to_string(row_bytes) +
         " bytes) cannot be aliased for MPSGraph on this macOS version; "
         "packed MPSNDArray rows require macOS 15 or later.")
            .c_str());
    return nil;
  }

  // The alias that makes the whole path zero-copy: the array points into the
  // existing allocation at the tensor's own offset, so neither inputs nor
  // outputs are ever staged through a separate buffer.
  MPSNDArray* array = [[[MPSNDArray alloc] initWithBuffer:slice.buffer
                                                   offset:slice.offset
                                               descriptor:descriptor]
      autorelease];
  if (array == nil) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: could not alias a tensor as an MPSNDArray.");
    return nil;
  }

  MPSGraphTensorData* data =
      [[[MPSGraphTensorData alloc] initWithMPSNDArray:array] autorelease];
  if (data == nil) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: could not wrap an MPSNDArray for MPSGraph.");
    return nil;
  }
  return data;
}

MPSGraphTensorData* TensorDataForTensor(TF_Tensor* tensor, TF_DataType dtype,
                                        id<MTLDevice> device,
                                        TF_Status* status) {
  BufferSlice slice;
  if (!SliceForTensor(tensor, &slice, status)) return nil;
  return TensorDataFor(slice, ShapeOf(tensor), dtype, device, status);
}

const CachedGraph* LookupOrBuildGraph(const std::string& key,
                                      void (^builder)(CachedGraph* out),
                                      TF_Status* status) {
  return GraphCache::Global().LookupOrBuild(key, builder, status);
}

bool RunGraph(SP_Stream stream, const CachedGraph& cached,
              NSArray<MPSGraphTensorData*>* input_data,
              NSArray<MPSGraphTensorData*>* output_data, TF_Status* status) {
  if ([input_data count] != [cached.inputs count] ||
      [output_data count] != [cached.outputs count]) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: MPSGraph feed or result count does not match the "
                 "cached graph.");
    return false;
  }

  MPSGraphTensorDataDictionary* feeds = [NSMutableDictionary dictionary];
  for (NSUInteger i = 0; i < [cached.inputs count]; ++i) {
    [(NSMutableDictionary*)feeds setObject:input_data[i]
                                    forKey:cached.inputs[i]];
  }
  // Results are written straight into the output tensors' own storage, so no
  // copy happens on the way out either.
  MPSGraphTensorDataDictionary* results = [NSMutableDictionary dictionary];
  for (NSUInteger i = 0; i < [cached.outputs count]; ++i) {
    [(NSMutableDictionary*)results setObject:output_data[i]
                                      forKey:cached.outputs[i]];
  }

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for an MPSGraph "
                 "operation.");
    return false;
  }

  MPSCommandBuffer* mps_buffer =
      [MPSCommandBuffer commandBufferWithCommandBuffer:command_buffer.get()];

  // From here the command buffer belongs to MPS: encodeToCommandBuffer may
  // call commitAndContinue, committing the buffer we started with and moving
  // to a fresh one. So the ordering signal cannot go on the buffer we opened.
  const OrderedCommandBuffer::ExternalCommit commit =
      command_buffer.ReleaseForExternalCommit();

  [cached.graph encodeToCommandBuffer:mps_buffer
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:results
                  executionDescriptor:nil];

  // rootCommandBuffer is whichever MTLCommandBuffer is live now, which is not
  // necessarily the one handed to MPS above.
  id<MTLCommandBuffer> live = mps_buffer.rootCommandBuffer;
  [live encodeSignalEvent:commit.stream->order_event value:commit.signal_value];
  // MPSGraph's own buffer carries neither the label nor the handler the one we
  // opened had, so both are put back: without them an MPSGraph failure is
  // never reported and every op that goes through MPSGraph, which is most of
  // them, is missing from a profile.
  if (!CurrentOpName().empty() && live.label == nil) {
    live.label = [NSString stringWithUTF8String:CurrentOpName().c_str()];
  }
  SP_Stream owner = commit.stream;
  [live addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    NoteCommandBufferCompletion(owner, completed);
  }];
  [mps_buffer commit];
  if (SynchronousMode()) WaitForStream(stream);
  return true;
}

void AppendShapeToKey(const std::vector<int64_t>& shape, std::string* key) {
  key->push_back('[');
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) key->push_back(',');
    key->append(std::to_string(shape[i]));
  }
  key->push_back(']');
}

}  // namespace metal
}  // namespace tensorflow
