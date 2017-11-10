/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EAGER_TAPE_H_
#define TENSORFLOW_C_EAGER_TAPE_H_

// Language-agnostic gradient tape. Does not perform backpropagation, just
// maintains the data structures required to do so.

#include <unordered_map>
#include <vector>
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace eager {

// Information about a tensor.
struct TapeTensor {
  int64 id;  // Expected to be unique in the lifetime of this process.
  DataType dtype;
  TensorShape shape;
};

// Represents an entry in the tape.
struct OpTapeEntry {
  string op_type;
  std::vector<TapeTensor> output_tensor_info;
  std::vector<int64> input_tensor_id;

  // TODO(apassos) consider narrowing down this interface.
  void* backward_function;

  // Should be called before deleting the backward function. TODO(apassos) use
  // unique_ptrs to ensure this happens.
  std::function<void()> backward_function_deleter;
};

// Map from tensor_id to internally-defined operation-id of the operation which
// produced this tensor. A value of -1 means that the tensor was directly
// watched and not the result of any operation in the tape.
using TensorTape = std::unordered_map<int64, int64>;

// Map from operation-id to tape entry.
using OpTape = std::unordered_map<int64, OpTapeEntry>;

// Traces the execution of operations, doing eager garbage collection, and
// exporting a full trace so other code can do backpropagation. Not thread-safe.
class GradientTape {
 public:
  GradientTape() {}

  bool ShouldRecord(gtl::ArraySlice<int64> tensor_ids);

  void Watch(int64 tensor_id);

  void RecordOperation(const string& op_type,
                       gtl::ArraySlice<TapeTensor> output_tensors,
                       gtl::ArraySlice<int64> input_tensor_id,
                       void* backward_function,
                       const std::function<void()>& backward_function_deleter);

  void DeleteTrace(int64 tensor_id);

  // Note: it is only valid to call Export once per tape, and after calling
  // export the tape is no longer valid (i.e. calls to ShouldRecord, Watch,
  // Record, and Delete have undefined behavior).
  std::pair<TensorTape, OpTape> Export();

 private:
  TensorTape tensor_tape_;
  OpTape op_tape_;
  int64 next_op_id_{0};

  // Map from tensor id to number of remaining usages (i.e. how many entries in
  // the tape refer to it); to aid in tape garbage collection.
  std::unordered_map<int64, int64> tensor_usage_;
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_TAPE_H_
