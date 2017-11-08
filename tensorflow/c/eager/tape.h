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

// Operations the tape needs to perform on tensors to do backpropagation. Named
// "vspace" because a subset of these are related to a vector space, such as
// adding gradients, getting zeroes, etc. Currently cannot be implemented
// without using tensorflow python code, hence left unspecified here.
//
// We currently use void* for tensors, backward functions, and gradients (which
// can be but are not required to be tensors). TODO(apassos) replace this first
// with templates to allow for pyobject specialization in the client followed by
// a TFE_TensorHandle specialization, which is blocked by quite a few things
// still.
class VSpace {
 public:
  virtual ~VSpace() {}

  // Returns the number of elements in the tensor.
  virtual int64 NumElements(void* tensor) const = 0;

  // Consumes references to the tensors in the gradient_tensors list and returns
  // a tensor with the result.
  virtual void* AggregateGradients(
      gtl::ArraySlice<void*> gradient_tensors) const = 0;

  // Returns a tensor of the right shape and dtype filled with zeros.
  virtual void* Zeros(TensorShape shape, DataType dtype) const = 0;

  // Returns a Tensor which is filled with ones and like the input.
  virtual void* OnesLike(void*) const = 0;

  // Returns an integer which is a unique-to-within-this-program handle for this
  // tensor.
  virtual int64 TensorId(void* tensor) const = 0;

  // Calls the passed-in backward function.
  virtual Status CallBackwardFunction(void* backward_function,
                                      gtl::ArraySlice<void*> output_gradients,
                                      std::vector<void*>* result) const = 0;

  // Deletes the input tensor.
  virtual void DeleteTensor(void* tensor) const = 0;
};

// Traces the execution of operations, doing eager garbage collection, and
// exporting a full trace so other code can do backpropagation. Not thread-safe.
class GradientTape {
 public:
  GradientTape() {}
  ~GradientTape() {
    for (const auto& pair : op_tape_) {
      pair.second.backward_function_deleter();
    }
  }

  bool ShouldRecord(gtl::ArraySlice<int64> tensor_ids);

  void Watch(int64 tensor_id);

  void RecordOperation(const string& op_type,
                       gtl::ArraySlice<TapeTensor> output_tensors,
                       gtl::ArraySlice<int64> input_tensor_id,
                       void* backward_function,
                       const std::function<void()>& backward_function_deleter);

  void DeleteTrace(int64 tensor_id);

  // Consumes the internal state of the tape (so cannot be called more than
  // once) and produces the gradient of the target tensors with respect to the
  // source tensors. The output gradients are used if not empty and not
  // null. The result is populated with one tensor per target element.
  Status Gradient(const VSpace& vspace, gtl::ArraySlice<void*> target,
                  gtl::ArraySlice<void*> sources,
                  gtl::ArraySlice<void*> output_gradients,
                  std::vector<void*>* result);

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
