/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_REMAT_REMATERIALIZER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_REMAT_REMATERIALIZER_H_

// This file declares the Rematerializer class, which is used by an MLIR-based
// set of transformations for TFLite IR that lower memory usage by redoing
// operations with small inputs and large outputs instead of keeping the result
// in memory. This class allows us to compactly and efficiently represent the
// (idealized) memory profile of a TFLite graph and simulate the effect of
// re-inserting operations on that memory profile.

#include <algorithm>
#include <cinttypes>
#include <tuple>
#include <vector>

namespace mlir {
namespace TFL {

// A class that
// (1) Encodes in concise form the memory requirements of a computational graph
// (2) Allows for the fast simulation of changes to the peak memory requirement
//     under rematerialization of intermediate results in the graph
class Rematerializer {
 public:
  Rematerializer() = default;
  virtual ~Rematerializer() = default;

  // The type used for memory sizes (in bytes) and differences thereof.
  using SizeT = int64_t;

  // The memory profile: The i-th element gives the amount of memory
  // that is needed when performing the i-th operation. This is the
  // sum of the sizes of
  //
  // (1) input tensors of that operation,
  // (2) output tensors of that operation,
  // (3) output tensors of preceding operations that are input tensors
  //     of subsequent operations.
  using MemProfile = std::vector<SizeT>;

  // Used for specifying memory consumption at a certain operation in the
  // computational graph.
  struct MemSpec {
    int op_index;  // The index of the operation
    SizeT size;    // The amount of memory needed in order to execute this
                   // operation, i.e., the sum of input and output sizes and the
                   // sizes of outputs of previous operations that are needed as
                   // inputs of subsequent operations.
    explicit MemSpec(int op_index = 0, SizeT size = 0)
        : op_index(op_index), size(size) {}
  };

  static bool BySize(const MemSpec& a, const MemSpec& b) {
    return std::tie(a.size, a.op_index) < std::tie(b.size, b.op_index);
  }

  static bool ByOpIndex(const MemSpec& a, const MemSpec& b) {
    return std::tie(a.op_index, a.size) < std::tie(b.op_index, b.size);
  }

  // Gives the peak memory location and size. Ties are broken towards
  // later locations.
  MemSpec GetPeakMemory() const;

  // Gives memory profile. Mostly used for tests.
  MemProfile GetMemProfile() const;

 protected:
  // The next protected methods are to be used by derived classes to create the
  // low-level (this class) representation from a high-level one.

  // Creates a new tensor-like object that takes `size` bytes. Returns a
  // contiguous increasing index for each new object, starting at 0.
  int AddTensor(SizeT size);

  // Creates an operation. Returns a contiguous increasing index for each new
  // object, starting at 0.
  int AddOperation();

  // The operator with index `ioperation` will be assumed to produce and/or
  // consume the tensor with index `itensor`. NoOp if that's already the case.
  // The arguments must be valid indices (i.e., obtained with
  // `AddOperation`/`AddTensor`).
  void AddUse(int ioperation, int itensor);

  // Undoes an AddUse(ioperation, itensor). NoOp if there was no prior `AddUse`.
  // The arguments must be valid indices (i.e., obtained with
  // `AddOperation`/`AddTensor`).
  void DelUse(int ioperation, int itensor);

  // The memory objects.
  struct Tensor {
    SizeT size;                   // The size of the object (in bytes.)
    std::vector<int> operations;  // The operations it is used in. This vector
                                  // is kept sorted + unique.

    // The operation that makes the first use of this tensor.
    int first_use() const { return *operations.begin(); }

    // The operation that makes the last use of this tensor.
    int last_use() const { return *operations.rbegin(); }
  };

  // The operators.
  struct Operation {
    std::vector<int> tensors;  // The tensors that are used (input or output) by
                               // this operation. They needn't correspond to
                               // tensors in the TF graph -- we may add fake
                               // tensors to model memory consumed in addition
                               // to input and output tensors. This vector is
                               // kept sorted + unique.

    SizeT alloc = 0;    // The number of bytes that need to be allocated before
                        // this operation.
    SizeT dealloc = 0;  // The number of bytes that can be deallocated after
                        // this operation.
  };

  // Helper template: Iterates through all `MemSpec`s (i.e., operation
  // index/memory usage pairs) for the current graph in operation order and
  // calls `mapper` on them. This is an optimization -- by instantiating with an
  // appropriate `Mapper`, it allows us to e.g. compute the peak memory without
  // having to instantiate an actual memory profile vector.
  template <class Mapper>
  void MapMem(const Mapper& mapper) const {
    for (MemSpec m; m.op_index < operations_.size(); ++m.op_index) {
      m.size += operations_[m.op_index].alloc;
      mapper(m);
      m.size -= operations_[m.op_index].dealloc;
    }
  }

  std::vector<Operation> operations_;
  std::vector<Tensor> tensors_;
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_REMAT_REMATERIALIZER_H_
