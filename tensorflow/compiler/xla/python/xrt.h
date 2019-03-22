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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_XRT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_XRT_H_

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {
namespace swig {

// Represents a reference to literals that live in a device-allocated buffer via
// XRT. Specifically, wraps an int64 handle produced by running the allocation
// graph, and an XLA shape to track the referent's shape.
class XrtAllocation {
 public:
  // Accepts a `session_target` argument, used in constructing the
  // `tensorflow::ClientSession` instance in which allocation and deallocation
  // graphs are run.
  static StatusOr<XrtAllocation*> FromLiteral(const Literal& argument,
                                              const string& session_target);

  XrtAllocation(int64 handle, Shape shape, const string& session_target);
  ~XrtAllocation();
  StatusOr<Literal> ToLiteral() const;
  const Shape& shape() const;
  const int64 handle() const;

 private:
  const int64 handle_;
  const Shape shape_;
  const string session_target_;
};

// Result of a tuple destructuring operation on an XrtAllocation.
class XrtAllocationTuple {
 public:
  // Note: any XrtAllocation elements that are not Release()'d will be
  // deallocated in the destructor.
  explicit XrtAllocationTuple(std::vector<XrtAllocation*> elements);

  ~XrtAllocationTuple();

  // Releases the ith element to the caller. Further attempts to release the ith
  // element will return an invalid argument error.
  StatusOr<XrtAllocation*> Release(int i);

  // Returns the number of elements in the destructured tuple.
  int64 size() const;

 private:
  std::vector<XrtAllocation*> elements_;
};

// Destructures a tuple-valued XrtAllocation into its constituent elements
// in XrtAllocationTuple form.
//
// Accepts a `session_target` argument, used in constructing the
// `tensorflow::ClientSession` instance in which the sub-tupling graph is run,
// and passed along in constructing each constituent XrtAllocation.
StatusOr<XrtAllocationTuple*> DestructureXrtAllocationTuple(
    XrtAllocation* allocation, const string& session_target);

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. Specifically, wraps an XRT computation handle.
class XrtExecutable {
 public:
  // Accepts a `session_target` argument, used in constructing the
  // `tensorflow::ClientSession` instance in which the compilation graph is run.
  static StatusOr<XrtExecutable*> CompileForXrt(
      const string& hlo_module_proto, const std::vector<Shape>& argument_shapes,
      const Shape& result_shape, const string& session_target);

  // Accepts a `session_target` argument, used in constructing the
  // `tensorflow::ClientSession` instance in which the execution graph is run.
  XrtExecutable(const ProgramShape& program_shape, int64 handle,
                const string& session_target);
  ~XrtExecutable();

  std::vector<int> DeviceOrdinals() const { return {0}; }

  StatusOr<XrtAllocation*> Execute(
      absl::Span<XrtAllocation* const> argument_handles);

  const ProgramShape& program_shape() const;
  int64 handle() const;

 private:
  const ProgramShape program_shape_;
  const int64 handle_;
  const string session_target_;
};

// Functions for freeing resources from the Python side.
void DeleteXrtAllocation(XrtAllocation* allocation);
void DeleteXrtExecutable(XrtExecutable* computation);

}  // namespace swig
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_XRT_H_
