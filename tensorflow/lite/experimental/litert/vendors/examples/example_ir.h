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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_IR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_IR_H_

#include <cstdint>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/lite/experimental/litert/vendors/cc/backend_ir.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/ir_types.h"

namespace litert::example {

// Example IR wrapper types for an imaginary backend.

// Example backend knows only float and int 32.
enum class ExampleTensorType {
  FLOAT,
  INT,
};

// Example backend tensor wrapper that stores the type and shape and unique ID.
struct ExampleTensor {
  using Id = int32_t;
  ExampleTensorType type;
  std::vector<uint32_t> dims;
  std::string name;
  Id id = -1;
};

// Example backend knows only a few simple ops.
enum class ExampleOpType {
  ADD,
  MUL,
  RELU,
};

// Example backend op that stores op type as well as input and output tensor
// IDs and names.
struct ExampleOp {
  ExampleOpType op_code;
  std::vector<ExampleTensor::Id> inputs;
  std::vector<std::string> input_names;
  std::vector<ExampleTensor::Id> outputs;
  std::vector<std::string> output_names;
};

// Simple allocator(s) for example example IR types that provides pointer
// stability.
template <class E>
class ExampleIrAllocatorBase {
 public:
  ExampleIrAllocatorBase(const ExampleIrAllocatorBase&) = delete;
  ExampleIrAllocatorBase& operator=(const ExampleIrAllocatorBase&) = delete;
  ExampleIrAllocatorBase() = default;

 protected:
  std::list<E> ir_;
};

// Allocator for example tensors that provides pointer stability and unique IDs.
class ExampleTensorAllocator : public ExampleIrAllocatorBase<ExampleTensor> {
 private:
  using Alloc = BackendIrAllocator<ExampleTensor>;

 public:
  ExampleTensor* operator()() {
    auto& tensor = this->ir_.emplace_back();
    tensor.id = this->next_id_++;
    return &tensor;
  }

  // Return lambda instead of implicit copy construction when converting to
  // function type.
  // NOLINTNEXTLINE
  operator Alloc() {
    return [this]() { return this->operator()(); };
  }

  ExampleTensorAllocator(const ExampleTensorAllocator&) = delete;
  ExampleTensorAllocator& operator=(const ExampleTensorAllocator&) = delete;
  ExampleTensorAllocator() = default;

 private:
  uint32_t next_id_ = 0;
};

// Allocator for example ops that provides pointer stability.
class ExampleOpAllocator : public ExampleIrAllocatorBase<ExampleOp> {
 private:
  using Alloc = BackendIrAllocator<ExampleOp>;

 public:
  ExampleOp* operator()() { return &this->ir_.emplace_back(); }

  // Return lambda instead of implicit copy construction when converting to
  // function type.
  // NOLINTNEXTLINE
  operator Alloc() {
    return [this]() { return this->operator()(); };
  }

  ExampleOpAllocator(const ExampleOpAllocator&) = delete;
  ExampleOpAllocator& operator=(const ExampleOpAllocator&) = delete;
  ExampleOpAllocator() = default;
};

// Builder for graph conversion to example IR. The internal example IR graph is
// simply a string representation of the graph.
class ExampleGraphBuilder
    : public BackendGraphBuilder<ExampleOp, ExampleTensor> {
 public:
  // Prefixes ir string.
  void InitGraph(std::string graph_name) override;

  // Registers tensor into the currrent graph by simply appending its string
  // representation.
  LiteRtStatus RegisterTensor(ExampleTensor& tensor) override;

  // Registers op into the currrent graph by simply appending its string
  // representation.
  LiteRtStatus RegisterOp(ExampleOp& op) override;

  // Simply appends tag to IR string.
  LiteRtStatus FinalizeGraph() override;

  // Gets the serialized IR representation.
  std::string Serialize() const;

 private:
  std::stringstream example_graph_;
};

using ExampleTypes = IrTypes<ExampleOp, ExampleTensor>;

}  // namespace litert::example

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_IR_H_
