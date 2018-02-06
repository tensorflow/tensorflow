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

#include "tensorflow/compiler/xla/service/cpu/parallel_cpu_executable.h"

#include <stdint.h>
#include <algorithm>
#include <deque>
#include <iterator>
#include <list>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace cpu {

ParallelCpuExecutable::ParallelCpuExecutable(
    std::unique_ptr<SimpleOrcJIT> jit,
    std::unique_ptr<const BufferAssignment> assignment,
    std::unique_ptr<const HloModule> hlo_module,
    std::unique_ptr<const HloInstructionMap<string>> function_names,
    std::unordered_map<const HloInstruction*, std::unique_ptr<unsigned char[]>>
        aligned_constants,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)),
      jit_(std::move(jit)),
      assignment_(std::move(assignment)),
      function_names_(std::move(function_names)),
      aligned_constants_(std::move(aligned_constants)) {}

// Type of the computation function we expect in the JIT.
using ComputeFunctionType = void (*)(void*, const void*, const void**, void**,
                                     int64*, int64*);

// Given a pointer to an output buffer (following the CPU JIT calling
// conventions), mark addresses that are "live". The initial pointer itself is
// trivially live. If the shape of the buffer is a tuple, this analysis looks
// into the tuple's elements and marks them live as well (since tuples keep
// pointers to buffers) and also works recursively.
// address is an in-memory buffer address that contains some runtime XLA object.
// shape is its shape. marked_addresses is the set of live addresses to
// populate.
static void MarkLiveAddressesInOutput(
    const void* address, const Shape& shape,
    std::unordered_set<const void*>* marked_addresses) {
  marked_addresses->insert(address);
  const uintptr_t* address_buffer = static_cast<const uintptr_t*>(address);
  if (ShapeUtil::IsTuple(shape)) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      const uintptr_t* element_address = address_buffer + i;
      const void* element = reinterpret_cast<const void*>(*element_address);
      MarkLiveAddressesInOutput(
          element, ShapeUtil::GetTupleElementShape(shape, i), marked_addresses);
    }
  }
}

namespace {

// Executor manages the concurrent execution of 'functions' for instructions
// in 'pending' on 'thread_pool' (storing resulting data in 'results').
class Executor {
 public:
  Executor(const HloInstructionMap<ComputeFunctionType>& functions,
           const ServiceExecutableRunOptions* run_options,
           std::list<HloInstruction*>* pending,
           HloInstructionMap<const void*>* results, void** temps_array,
           int64* profile_counters_array, const BufferAssignment* assignment)
      : functions_(functions),
        run_options_(run_options),
        pending_(pending),
        results_(results),
        temps_array_(temps_array),
        profile_counters_array_(profile_counters_array),
        thread_pool_(CHECK_NOTNULL(run_options_->xla_intra_op_thread_pool())),
        assignment_(assignment) {}

  // Executes pending list of instructions on thread pool.
  // Returns OK status on success, error status otherwise.
  Status Run();

 private:
  // Schedules a parallel invocation of compute function for 'instruction' on
  // 'thread_pool_', storing result in 'result_buffer'.
  // If 'partition_buffers' is non-null, parallel task will be invoked on
  // per-dimension partition [start, limit) values stored in
  // 'partition_buffers'.
  void Schedule(HloInstruction* instruction, int64* partition_buffers,
                void* result_buffer);

  // Returns true if 'instruction' has been assigned parallel tasks (returns
  // false otherwise).
  bool HasParallelTasks(HloInstruction* instruction);

  // Returns in 'partition_buffers' the partition [size, limit) for each
  // dimension.
  int64* GetPartitionBuffers(
      const std::vector<std::pair<int64, int64>>& partition);

  // Returns array of result buffers for all operands in 'instruction'.
  const void** GetOperandBuffers(HloInstruction* instruction);

  // Arguments passed into Executor.
  const HloInstructionMap<ComputeFunctionType>& functions_;
  const ServiceExecutableRunOptions* run_options_;
  std::list<HloInstruction*>* pending_;
  HloInstructionMap<const void*>* results_;
  void** temps_array_;
  int64* profile_counters_array_;
  tensorflow::thread::ThreadPool* thread_pool_;
  const BufferAssignment* assignment_;

  // Members used to manage instruction execution.
  tensorflow::mutex completion_queue_lock_;
  tensorflow::condition_variable completion_queue_cv_;
  std::deque<HloInstruction*> completion_queue_;
  int64 instructions_in_flight_ = 0;
  std::unordered_map<const HloInstruction*, int64> tasks_in_flight_;
};

Status Executor::Run() {
  while (!pending_->empty() || instructions_in_flight_ > 0) {
    auto pending_it = pending_->begin();
    while (pending_it != pending_->end()) {
      HloInstruction* instruction = *pending_it;
      // Skip pending instructions whose operands aren't ready.
      if (std::any_of(instruction->operands().begin(),
                      instruction->operands().end(),
                      [&](HloInstruction* operand) {
                        return !ContainsKey(*results_, operand);
                      })) {
        ++pending_it;
        continue;
      }

      // Get 'result_buffer' reference to result buffer for 'instruction'.
      TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                          assignment_->GetUniqueTopLevelSlice(instruction));
      void* result_buffer =
          static_cast<char*>(temps_array_[result_slice.index()]) +
          result_slice.offset();

      if (HasParallelTasks(instruction)) {
        // 'instruction' has been assigned parallel task partitions.
        CHECK_EQ(HloOpcode::kCall, instruction->opcode());
        HloInstruction* root = instruction->to_apply()->root_instruction();

        // Create ShapePartitionIterator to iterate through all outer dimension
        // partitions of 'instruction'.
        ShapePartitionIterator partition_iterator(
            root->shape(), root->outer_dimension_partitions());

        const int64 partition_count =
            partition_iterator.GetTotalPartitionCount();

        // Record total parallel task count for 'instruction' before dispatch.
        {
          tensorflow::mutex_lock l(completion_queue_lock_);
          tasks_in_flight_.insert(std::make_pair(instruction, partition_count));
          VLOG(2) << "Schedule PARALLEL"
                  << " instruction: " << instruction->name()
                  << " instruction.callee: "
                  << instruction->to_apply()->root_instruction()->name()
                  << " partition_count: " << partition_count;
        }

        for (int64 i = 0; i < partition_count; ++i) {
          // Get partition [start, limit) for each dimension.
          auto partition_buffers =
              GetPartitionBuffers(partition_iterator.GetPartition(i));
          Schedule(instruction, partition_buffers, result_buffer);
        }

      } else {
        // Set tasks in-flight to '1' for sequential instruction execution.
        {
          tensorflow::mutex_lock l(completion_queue_lock_);
          tasks_in_flight_.insert(std::make_pair(instruction, 1));
          VLOG(2) << "Schedule SEQUENTIAL"
                  << " instruction: " << instruction->name()
                  << " instruction.callee: "
                  << instruction->to_apply()->root_instruction()->name();
        }
        Schedule(instruction, nullptr, result_buffer);
      }

      ++instructions_in_flight_;
      pending_it = pending_->erase(pending_it);
    }
    // Wait for a completed HLO instruction to be present in the queue.  We will
    // pop it out of the queue and make the result available to its users.
    HloInstruction* instruction;
    do {
      tensorflow::mutex_lock l(completion_queue_lock_);
      if (completion_queue_.empty()) {
        completion_queue_cv_.wait(l);
      }
      if (!completion_queue_.empty()) {
        instruction = completion_queue_.front();
        completion_queue_.pop_front();
        break;
      }
    } while (true);
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                        assignment_->GetUniqueTopLevelSlice(instruction));
    void* result_buffer =
        static_cast<char*>(temps_array_[result_slice.index()]) +
        result_slice.offset();
    InsertOrDie(results_, instruction, result_buffer);
    --instructions_in_flight_;
  }
  return Status::OK();
}

void Executor::Schedule(HloInstruction* instruction, int64* partition_buffers,
                        void* result_buffer) {
  // The thread pool entry takes ownership of |operand_buffers|.
  auto operand_buffers = GetOperandBuffers(instruction);

  auto function = FindOrDie(functions_, instruction);
  const auto* exec_run_options = &run_options_->run_options();
  thread_pool_->Schedule([this, instruction, result_buffer, operand_buffers,
                          partition_buffers, exec_run_options, function]() {
    function(result_buffer, exec_run_options, operand_buffers, temps_array_,
             partition_buffers, profile_counters_array_);

    delete[] operand_buffers;
    delete[] partition_buffers;
    // Push the completed HLO instruction on the queue, the main
    // thread will pop it off and potentially launch more work which
    // uses the result.
    // TODO(b/27458679) Consider alternative task scheduling and synchronization
    // schemes. For example, we could avoid the overhead associate with the
    // condvar here if the thread just dequed the next instruction to execute
    // on completion.
    {
      tensorflow::mutex_lock l(completion_queue_lock_);
      // Decrement in-flight task count for this completion.
      if (--FindOrDie(tasks_in_flight_, instruction) == 0) {
        completion_queue_.push_back(instruction);
        completion_queue_cv_.notify_all();
        tasks_in_flight_.erase(instruction);
      }
    }
  });
}

int64* Executor::GetPartitionBuffers(
    const std::vector<std::pair<int64, int64>>& partition) {
  // Return in 'partition_buffers' partition [size, limit) for each dimension.
  auto partition_buffers = new int64[partition.size() * 2];
  for (int i = 0; i < partition.size(); ++i) {
    partition_buffers[2 * i + 0] = partition[i].first;
    partition_buffers[2 * i + 1] = partition[i].first + partition[i].second;
  }
  return partition_buffers;
}

bool Executor::HasParallelTasks(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kCall &&
         !instruction->to_apply()
              ->root_instruction()
              ->outer_dimension_partitions()
              .empty();
}

const void** Executor::GetOperandBuffers(HloInstruction* instruction) {
  // We cannot use a move-only RAII type like std::unique_ptr because the
  // list of operands is allocated on the main thread and transferred to the
  // worker via the lambda passed to enqueue_function.  In order for the
  // lambda to take ownership, we would need to use generalized lambda
  // capture which is a feature new to C++14.
  // TODO(b/27458679) Avoid dynamic allocations in Executor.
  auto operand_buffers = new const void*[instruction->operand_count()];
  std::transform(instruction->operands().begin(), instruction->operands().end(),
                 operand_buffers, [this](HloInstruction* operand) {
                   return FindOrDie(*results_, operand);
                 });
  return operand_buffers;
}

}  // namespace

Status ParallelCpuExecutable::AllocateBuffers(
    DeviceMemoryAllocator* memory_allocator, int device_ordinal,
    std::vector<perftools::gputools::DeviceMemoryBase>* buffers) {
  CHECK_EQ(buffers->size(), assignment_->Allocations().size());
  VLOG(3) << "Allocating " << assignment_->Allocations().size()
          << " allocations for module " << module().name();
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    auto& allocation = assignment_->GetAllocation(i);

    VLOG(3) << allocation.ToString();

    if (allocation.is_entry_computation_parameter()) {
      VLOG(3) << "allocation #" << i << " is a parameter";
      continue;
    }

    if (allocation.is_thread_local()) {
      VLOG(3) << "buffer #" << i << " is thread-local";
      continue;
    }

    int64 buffer_size = allocation.size();
    if (!(*buffers)[i].is_null()) {
      VLOG(3) << "buffer #" << i
              << " is in the preallocated result ShapedBuffer";
    } else {
      TF_ASSIGN_OR_RETURN((*buffers)[i], memory_allocator->Allocate(
                                             device_ordinal, buffer_size));

      VLOG(3) << "buffer #" << i << " allocated " << buffer_size << " bytes ["
              << (*buffers)[i].opaque() << "]";
    }

    // Since the output buffer and all the temporary buffers were written into
    // by the JITed code, msan has no way of knowing their memory was
    // initialized. Mark them initialized so that msan doesn't flag loads from
    // these buffers.
    TF_ANNOTATE_MEMORY_IS_INITIALIZED((*buffers)[i].opaque(), buffer_size);
  }

  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                      assignment_->GetUniqueTopLevelOutputSlice());
  VLOG(3) << "result index: " << result_slice.index();

  return Status::OK();
}

Status ParallelCpuExecutable::ExecuteComputeFunctions(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> buffers,
    HloExecutionProfile* hlo_execution_profile) {
  // Allocate profiling counters for each hlo instruction that we would like to
  // profile.
  std::vector<int64>* profile_counters = nullptr;
  if (hlo_execution_profile) {
    profile_counters = hlo_execution_profile->mutable_profile_counters();
  }

  std::vector<void*> buffer_pointers;
  buffer_pointers.reserve(buffers.size());
  for (auto device_allocation : buffers) {
    buffer_pointers.push_back(device_allocation.opaque());
  }

  // Resolve functions for all the HLO instructions ahead of time.
  HloInstructionMap<ComputeFunctionType> functions;
  for (auto& entry : *function_names_) {
    tensorflow::mutex_lock lock(jit_mutex_);
    HloInstruction* instruction = entry.first;
    llvm::JITSymbol sym = jit_->FindSymbol(entry.second);
    TF_RET_CHECK(sym);
    InsertOrDie(
        &functions, instruction,
        reinterpret_cast<ComputeFunctionType>(cantFail(sym.getAddress())));
  }

  // Map containing pointers to result buffers for each instruction.
  HloInstructionMap<const void*> results;

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  std::list<HloInstruction*> pending;

  // Call the function for each HLO instruction in topological order.
  const HloComputation& entry_computation = *module().entry_computation();
  for (auto* instruction : entry_computation.MakeInstructionPostOrder()) {
    // Parameters and constants have no functions associated with them. Instead
    // just copy the existing buffer into the map containing instruction
    // results..
    if (instruction->opcode() == HloOpcode::kParameter) {
      InsertOrDie(
          &results, instruction,
          arguments[instruction->parameter_number()]->root_buffer().opaque());
    } else if (instruction->opcode() == HloOpcode::kConstant) {
      unsigned char* aligned_data =
          FindOrDie(aligned_constants_, instruction).get();
      InsertOrDie(&results, instruction, aligned_data);
    } else {
      TF_RET_CHECK(instruction->opcode() == HloOpcode::kCall);
      pending.push_back(instruction);
    }
  }

  // TODO(b/27458679) Manage scheduling based on in-flight concurrency limits.
  // For example, if we expect a library conv/matmul call to run at max
  // concurrency, we should not dispatch runnable instructions until the
  // library call is finished (to avoid expensive cache invalidation).
  Executor executor(
      functions, run_options, &pending, &results, buffer_pointers.data(),
      profile_counters ? profile_counters->data() : nullptr, assignment_.get());

  TF_RETURN_IF_ERROR(executor.Run());

  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  {
    tensorflow::mutex_lock lock(mutex_);
    double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));
  }

  return Status::OK();
}

StatusOr<std::unique_ptr<ShapedBuffer>> ParallelCpuExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  if (GetRootPointsToSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  se::Stream* stream = run_options->stream();
  DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  std::vector<se::DeviceMemoryBase> buffers(assignment_->Allocations().size());

  auto result_buffer = MakeUnique<ShapedBuffer>(
      /*on_host_shape=*/result_shape(), /*on_device_shape=*/result_shape(),
      stream->parent()->platform(), stream->parent()->device_ordinal());

  TF_RETURN_IF_ERROR(AllocateBuffers(
      memory_allocator, stream->parent()->device_ordinal(), &buffers));

  TF_RETURN_IF_ERROR(ExecuteComputeFunctions(run_options, arguments, buffers,
                                             hlo_execution_profile));

  // Copy DeviceMemoryBase values which into the respective location in
  // ShapedBuffer which is returned to the caller.
  std::vector<bool> buffers_in_result(assignment_->Allocations().size(), false);
  TF_RETURN_IF_ERROR(result_buffer->buffers().ForEachMutableElementWithStatus(
      [&](const ShapeIndex& index, se::DeviceMemoryBase* device_memory) {
        const auto& sources = this->GetRootPointsToSet().element(index);

        // The points to set is unambiguous so the set should be a singleton.
        CHECK_EQ(1, sources.size());
        const LogicalBuffer* buffer_source = sources[0];
        HloInstruction* src = buffer_source->instruction();

        // The source for this result buffer can be a nested buffer such as a
        // tuple element. The source instruction should have a non-parameter
        // buffer assigned.
        TF_ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            this->assignment_->GetUniqueSlice(src, buffer_source->index()));
        CHECK(!slice.allocation()->is_entry_computation_parameter());

        const BufferAllocation::Index buffer_index = slice.index();
        const se::DeviceMemoryBase& buffer = buffers[buffer_index];
        CHECK(!buffer.is_null() || buffer.size() == 0);
        *device_memory = buffer;
        buffers_in_result[buffer_index] = true;
        return Status::OK();
      }));

  // Free all buffers not in the result.
  for (size_t i = 0; i < buffers.size(); ++i) {
    se::DeviceMemoryBase alloc = buffers[i];
    if (!buffers_in_result[i] && !alloc.is_null()) {
      VLOG(3) << "CpuExecutable deallocating buffer #" << i << " ["
              << alloc.opaque() << "]";
      TF_RETURN_IF_ERROR(memory_allocator->Deallocate(
          stream->parent()->device_ordinal(), &alloc));
    }
  }

  return std::move(result_buffer);
}

StatusOr<std::unique_ptr<ShapedBuffer>>
ParallelCpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) {
  // TODO(b/30671675): Implement asynchronous execution mode.
  return Unimplemented(
      "Asynchronous execution on stream is not yet supported on CPU.");
}

const PointsToSet& ParallelCpuExecutable::GetRootPointsToSet() const {
  return assignment_->points_to_analysis().GetPointsToSet(
      module().entry_computation()->root_instruction());
}

}  // namespace cpu
}  // namespace xla
