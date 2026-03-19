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
#include <cstdint>
#include <tuple>
#include <vector>

namespace mlir {
namespace TFL {

// A class that
// (1) Encodes in concise form the memory requirements of a computational graph
// (2) Allows for the fast simulation of changes to the peak memory requirement
//     under rematerialization of intermediate results in the graph
// (3) Implements a greedy algorithm for finding rematerializations of
//     intermediate results in that graph to lower peak memory requirements.
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

  // Specifies an elementary rematerialization operation: The operations in
  // operations [`begin`, `end`) will be rescheduled before operation `insert`.
  // A valid `RematSpec` requires begin <= end <= insert <= number of
  // operations. Note that (1) `end` is exclusive -- begin == end signifies a
  // trivial RematSpec (no operation will be rescheduled), (2) the
  // zero-initialized RematSpec {} is trivial and always valid.
  struct RematSpec {
    int begin;
    int end;
    int insert;
  };

  // Gives the peak memory location and size after inserting operations
  // according to `remat` (but doesn't actually insert them.)  Ties are broken
  // towards later locations. `remat` must be valid (see above).
  MemSpec GetPeakMemory(const RematSpec& remat = {}) const;

  // Gives memory profile after inserting operations according to `remat` (but
  // doesn't actually insert them). `remat` must be valid (see above).
  MemProfile GetMemProfile(const RematSpec& remat = {}) const;

  // Runs the greedy incremental block algorithm: Finds a sequence of
  // rematerializations of block size up to max_block_length, each reducing peak
  // memory by at least min_savings. If max_cost >= 0, at most max_cost
  // operations will be re-inserted. For each rematerialization found,
  // ApplyRemat is invoked (which can be used to apply the rematerialization to
  // the higher- level representation, e.g., MLIR, flatbuffer, ...)
  void RunGreedyAlgorithm(int max_cost, int max_block_length,
                          SizeT min_savings);

  virtual void ApplyRemat(const RematSpec& remat) {}

 protected:
  // Rematerializes the outputs of the operations [`remat.begin`, `remat.end`)
  // before operation remat.insert by copying that operation range before
  // remat.insert and updating tensor references so that any operation that can
  // will make use of the rematerialized outputs rather than the original ones.
  // `remat` must be valid (see above).
  void Remat(const RematSpec& remat);

  // The protected methods below are to be used by derived classes to create the
  // low-level (this class) representation from a high-level one.

  // Creates a new tensor-like object that takes `size` bytes. Returns a
  // contiguous increasing index for each new object, starting at 0.
  int AddTensor(SizeT size);

  // Creates an operation. If `is_stateful`, the operation (and any block of
  // operations containing it) will never be considered for rematerialization.
  // Returns a contiguous increasing index for each new object, starting at 0.
  int AddOperation(bool is_stateful);

  // The operator with index `ioperation` will be assumed to produce and/or
  // consume the tensor with index `itensor`. NoOp if that's already the case.
  // The arguments must be valid indices (i.e., obtained with
  // `AddOperation`/`AddTensor`).
  void AddUse(int ioperation, int itensor);

  // Undoes an AddUse(ioperation, itensor). NoOp if there was no prior `AddUse`.
  // The arguments must be valid indices (i.e., obtained with
  // `AddOperation`/`AddTensor`).
  void DelUse(int ioperation, int itensor);

 private:
  // Find the best remat operation that saves at least `min_savings` bytes for a
  // block of operators with a length is between [`begin_len`, `end_len`).
  // 'Best' means with the highest savings, ties are broken towards shorter
  // blocks.
  std::tuple<SizeT, RematSpec> FindBestRemat(SizeT min_savings, int begin_len,
                                             int end_len) const;

  // Optimization: Estimate (from above) the remat savings of instruction block
  // [begin, end) after operation `peak_location`
  SizeT MaxSavings(int begin, int end, int peak_loc) const;

  // If I want to remat ops [begin, end) after the op at operation `peak_loc`,
  // find the latest point at which to reinsert them (the op before which to
  // insert.)
  int FindBestRematPoint(int begin, int end, int peak_loc) const;

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
    bool is_stateful = false;  // Results of an Operation can be rematerialized
                               // only if `!is_stateful`. This probably should
                               // be replaced with a more-fine grained
                               // approach--for example, the results of a "read
                               // resource variable" operation can be
                               // rematerialized as long as this doesn't happen
                               // after the corresponding "write resource
                               // variable" operation.

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

  // Given the current state of `operations_` and `tensors_`, return a vector of
  // corrections that transform the current memory profile into the one that we
  // would get after applying `remat`.
  //
  // The memory profile of a sequence of operations is the partial sum of the
  // sizes of the allocations that are necessary before an operation and the
  // negative sizes of the deallocations that are possible after the previous
  // operation.
  //
  // If we modify the operation sequence by cloning an operation range, that
  // memory profile will change--cloning makes it necessary to extend the
  // lifetime of some tensors, while other tensors can be deallocated early and
  // rematerialized later.
  //
  // This method represents these changes in compact form: It returns a vector
  // of (position of operation, delta) pairs in lexicographic order; one
  // obtains the memory profile after `remat` by adding the deltas from any
  // entries (i, delta) to the i-th entry of the partial sum.
  //
  // This allows us to efficiently compute the change to the peak of a memory
  // profile due to cloning an operation range without having to actually clone
  // that range and without having to build a profile vector.
  //
  // The returned vector has at most 2 entries for each tensor referenced in
  // [remat.begin, remat.end). There may be multiple entries for a single
  // operation position; operation positions refer to the sequence *after*
  // cloning [`remat.begin`, `remat.end`) before `remat.insert`.
  std::vector<MemSpec> GetDeltas(const RematSpec& remat) const;

  // Helper template: Iterates through all `MemSpec`s (i.e., operation
  // index/memory usage pairs) for the current graph in operation order and
  // calls `mapper` on them. This is an optimization -- by instantiating with an
  // appropriate `Mapper`, it allows us to e.g. compute the peak memory without
  // having to instantiate an actual memory profile vector.
  template <class Mapper>
  void MapMem(const Mapper& mapper, const RematSpec& remat) const {
    const auto deltas = GetDeltas(remat);
    const auto len = (remat.end - remat.begin);
    auto idelta = deltas.begin();

    for (MemSpec m; m.op_index < operations_.size() + len; ++m.op_index) {
      // Are we in the cloned portion of the new operation sequence?
      // Then all alloc/dealloc information must come from deltas.
      const bool patch =
          (m.op_index >= remat.insert) && (m.op_index < remat.insert + len);
      // Are we past the insertion portion of the new operation sequence?
      // Then we need to convert indices back to the original sequence.
      const int shift = (m.op_index >= remat.insert + len) ? len : 0;
      m.size += patch ? 0 : operations_[m.op_index - shift].alloc;
      // deltas is sorted by location; apply any corrections to the current
      // operator.
      for (; idelta != deltas.end() && idelta->op_index == m.op_index;
           ++idelta) {
        m.size += idelta->size;
      }
      mapper(m);
      m.size -= patch ? 0 : operations_[m.op_index - shift].dealloc;
    }
  }

  std::vector<Operation> operations_;
  std::vector<Tensor> tensors_;
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_REMAT_REMATERIALIZER_H_
