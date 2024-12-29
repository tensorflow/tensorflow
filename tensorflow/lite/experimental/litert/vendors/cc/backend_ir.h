// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_BACKEND_IR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_BACKEND_IR_H_

#include <functional>
#include <string>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

namespace litert {

// Interfaces and types for managing backend IR to be targeted by LiteRt for
// compilation.

// Memory Management
//===---------------------------------------------------------------------------

// Callable for allocating a new instance of a backend IR type. This facilitates
// external memory management for the backend IR implementented by the backend.
// It is encouraged for implementations provide pointer stability (consider
// std::list for storage).
template <class BackendIr, class... Args>
using BackendIrAllocator = std::function<BackendIr*(Args&&... args)>;

// Allocator for backend tensors.
template <class BackendTensor>
using TensorAllocator = BackendIrAllocator<BackendTensor>;

// Allocator for backend ops.
template <class BackendOp>
using OpAllocator = BackendIrAllocator<BackendOp>;

// Graph Construction
//===---------------------------------------------------------------------------

// Wrapper for an in memory graph for a particular backend. Implementations
// should contain an instance of a backend graph that can be iteratively
// constructed via calls to this interface.
template <class BackendOp, class BackendTensor>
class BackendGraphBuilder {
 public:
  // Hook called to initialize state for a new backend graph with a name. This
  // will be called once per-instance before any other method.
  virtual void InitGraph(std::string graph_name) = 0;

  // Hook called to register a backend tensor once it
  // has been converted. This will be called once per tensor.
  virtual LiteRtStatus RegisterTensor(BackendTensor& tensor) = 0;

  // Hook called to register a backend op once it has been converted. This will
  // be called once per op (in a toplogogical order). All input/output tensors
  // will have been registered before called.
  virtual LiteRtStatus RegisterOp(BackendOp& op) = 0;

  // Hook called to register a graph when graph
  // conversion is completed. Backend graph context should be stored as internal
  // state. This will be called once per instance after all ops/tensors have
  // been finalized.
  virtual LiteRtStatus FinalizeGraph() = 0;

  virtual ~BackendGraphBuilder() = default;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_BACKEND_IR_H_
