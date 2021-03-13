/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_EMITTER_H_

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace gpu {

// Implements handling of GPU execution for HLO operations that are handed off
// to specialized thunks that do not require code generation. Intended to be
// mixed into GPU emitters.
class ThunkEmitter {
 public:
  class EmissionContext {
   public:
    virtual void AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) = 0;
    virtual StatusOr<BufferAllocation::Slice> MaybeGetAllocationSlice(
        const HloInstruction& hlo, const ShapeIndex& index) const = 0;
    virtual int64 ByteSizeOf(const Shape& shape) const = 0;
    virtual absl::string_view platform_name() const = 0;
    virtual Thunk::ThunkInfo GetThunkInfo(const HloInstruction* hlo) const;

    virtual ~EmissionContext() = default;
  };

  explicit ThunkEmitter(EmissionContext* context) : context_(context) {}

  Status HandleTriangularSolve(HloInstruction* hlo);

 private:
  EmissionContext* context_;

  void AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) {
    return context_->AddThunkToThunkSequence(std::move(thunk));
  }

  StatusOr<BufferAllocation::Slice> MaybeGetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index) const {
    return context_->MaybeGetAllocationSlice(hlo, index);
  }

  int64 ByteSizeOf(const Shape& shape) { return context_->ByteSizeOf(shape); }

  absl::string_view platform_name() const { return context_->platform_name(); }

  BufferAllocation::Slice GetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index = {}) const {
    return MaybeGetAllocationSlice(hlo, index).ValueOrDie();
  }

  // Returns a CholeskyThunk that calls cuSolver to implement `inst`.
  std::unique_ptr<Thunk> BuildCholeskyThunk(const HloInstruction* inst);

  // Returns a GemmThunk that calls gemm to implement `inst`. The caller needs
  // to make sure `inst` outlives the lifetime of the returned Thunk object.
  std::unique_ptr<Thunk> BuildGemmThunk(const HloInstruction* inst);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_EMITTER_H_
