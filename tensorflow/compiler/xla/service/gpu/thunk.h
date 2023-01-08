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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_

#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {
namespace gpu {

class GpuExecutable;

// Thunk acts as the bridge between IrEmitter and GpuExecutable. It stores the
// metadata IrEmitter generates for GpuExecutable to invoke an HloInstruction.
//
// Thunk provides the Initialize and ExecuteOnStream interface for GpuExecutable
// to initialize and execute the invocation respectively. Its subclasses are
// supposed to override these interfaces to launch a generated kernel or call an
// external library function (such as operations in cuBLAS).
//
// This is thread-compatible.
class Thunk {
 public:
  enum Kind {
    kCholesky,
    kConditional,
    kConvolution,
    kCopy,
    kCublasLtMatmul,
    kCustomCall,
    kFft,
    kFor,
    kGemm,
    kInfeed,
    kKernel,
    kMemset32BitValue,
    kMemzero,
    kNcclAllGather,
    kNcclAllReduce,
    kNcclAllReduceStart,
    kNcclAllReduceDone,
    kNcclCollectivePermute,
    kNcclCollectivePermuteStart,
    kNcclCollectivePermuteDone,
    kNcclReduceScatter,
    kNcclAllToAll,
    kOutfeed,
    kReplicaId,
    kPartitionId,
    kSequential,
    kTriangularSolve,
    kWhile,
  };

  struct ThunkInfo {
    explicit ThunkInfo(mlir::Operation* op) : op(op) {}
    std::optional<int64_t> profile_index;
    std::string profile_annotation;
    mlir::Operation* op;
  };

  // The hlo_instruction argument is meant to be the instruction this thunk was
  // generated from, but Thunk never uses this argument other than to save it
  // to Thunk::hlo_instruction, so it can be null.
  Thunk(Kind kind, ThunkInfo thunk_info)
      : kind_(kind),
        profile_index_(thunk_info.profile_index),
        profile_annotation_(thunk_info.profile_annotation),
        op_(thunk_info.op) {}
  virtual ~Thunk() = default;
  Thunk(const Thunk&) = delete;
  Thunk& operator=(const Thunk&) = delete;

  virtual std::string ToStringExtra(int indent) const { return ""; }
  Kind kind() const { return kind_; }
  std::string profile_annotation() const { return profile_annotation_; }
  // Only valid during compilation, i.e., lowering thunks to kernel-launch
  // related XLA runtime custom calls). nullptr at runtime. MLIR codegen will
  // cease the practice of lowering thunks to XLA runtime custom calls.
  mlir::Operation* op() { return op_; }

  // Prepares the thunk for execution on the given StreamExecutor.
  //
  // This may be called multiple times.  Its main purpose is to give us a chance
  // to do initialization outside of ExecuteOnStream() so that the
  // time spent initializing doesn't count towards our execution profile.
  virtual Status Initialize(const GpuExecutable& /*executable*/,
                            se::StreamExecutor* /*executor*/) {
    return OkStatus();
  }

  // Parameters passed to ExecuteOnStream.  Encapsulated in a struct so that
  // when we add something we don't have to change every subclass of Thunk.
  struct ExecuteParams {
    ExecuteParams(const ServiceExecutableRunOptions& run_options,
                  const BufferAllocations& buffer_allocations,
                  se::Stream* stream, se::Stream* async_comms_stream);

    const BufferAllocations* buffer_allocations;  // never null
    se::Stream* stream;
    se::Stream* async_comms_stream;
    NcclExecuteParams nccl_params;
  };

  // Execute the kernel for the thunk on the given stream. This method must be
  // called after Initialize and can be called multiple times over Thunk's
  // lifetime.
  //
  // Precondition: Initialize(stream->parent()) has been called.
  virtual Status ExecuteOnStream(const ExecuteParams& params) = 0;

  // Clears metadata that is only valid during compile time.
  virtual void ClearCompileTimeInfo() { op_ = nullptr; }

  static absl::string_view KindToString(Thunk::Kind kind);

 protected:
  std::optional<int64_t> profile_index() const { return profile_index_; }

 private:
  Kind kind_;
  std::optional<int64_t> profile_index_;
  std::string profile_annotation_;
  mlir::Operation* op_;
};

// A sequence of thunks.
class ThunkSequence : public std::vector<std::unique_ptr<Thunk>> {
 public:
  std::string ToString(int indent = 0,
                       std::function<std::string(const Thunk*)>
                           get_thunk_annotation = nullptr) const;
};

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind);

// A struct that defines a shaped slice, i.e., a BufferAllocation::Slice and its
// shape.
struct ShapedSlice {
  BufferAllocation::Slice slice;
  Shape shape;
};

// Returns if the thunk implements a reduction collective (all-reduce or
// reduce-scatter).
bool IsReductionCollective(Thunk::Kind kind);
}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_
