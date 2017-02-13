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

#include "external/llvm/include/llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
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
    std::unique_ptr<BufferAssignment> assignment,
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloModuleConfig> module_config,
    std::unique_ptr<std::map<HloInstruction*, string>> function_names,
    std::unordered_map<const HloInstruction*, size_t> hlo_to_profile_idx,
    std::unordered_map<const HloInstruction*, std::unique_ptr<unsigned char[]>>
        aligned_constants)
    : Executable(std::move(hlo_module), std::move(module_config)),
      jit_(std::move(jit)),
      assignment_(std::move(assignment)),
      functions_names_(std::move(function_names)),
      hlo_to_profile_idx_(std::move(hlo_to_profile_idx)),
      aligned_constants_(std::move(aligned_constants)) {}

// Type of the computation function we expect in the JIT.
using ComputeFunctionType = void (*)(void*, const void*, const void**, void**,
                                     uint64*);

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

StatusOr<perftools::gputools::DeviceMemoryBase>
ParallelCpuExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();
  DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  VLOG(3) << "ExecuteOnStream arg size: " << arguments.size();
  if (!arguments.empty()) {
    VLOG(3) << "ExecuteOnStream arg[0]: " << arguments.at(0).opaque();
  }

  // Allocate the temporary buffers required for the computation.
  se::StreamExecutor* stream_executor = stream->parent();
  int device_ordinal = stream_executor->device_ordinal();
  int64 buffer_count = assignment_->Allocations().size();
  VLOG(3) << "temp buffer count: " << buffer_count;

  std::vector<se::DeviceMemoryBase> device_allocations;
  for (BufferAllocation::Index i = 0; i < buffer_count; ++i) {
    auto& allocation = assignment_->GetAllocation(i);
    if (allocation.is_entry_computation_parameter()) {
      // Buffers do not need to be allocated for parameters.
      device_allocations.push_back(se::DeviceMemoryBase(nullptr));
      continue;
    }

    if (allocation.is_thread_local()) {
      // Buffers do not need to be allocated for thread-local temporaries.
      device_allocations.push_back(se::DeviceMemoryBase(nullptr));
      continue;
    }

    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase device_allocation,
        memory_allocator->Allocate(device_ordinal, allocation.size()));

    if (VLOG_IS_ON(3)) {
      VLOG(3) << "ParallelCpuExecutable allocating " << allocation.size()
              << " bytes for allocation #" << i << " ["
              << device_allocation.opaque() << "]";
      std::vector<string> parts;
      for (const auto& buffer_offset_size : allocation.assigned_buffers()) {
        const LogicalBuffer& buffer = *buffer_offset_size.first;
        parts.push_back(tensorflow::strings::StrCat(
            buffer.instruction()->parent()->name(), "::", buffer.ToString()));
      }
      VLOG(3) << "  " << tensorflow::str_util::Join(parts, ", ");
    }

    device_allocations.push_back(device_allocation);
    // Since the output buffer and all the temporary buffers were written into
    // by the JITed code, msan has no way of knowing their memory was
    // initialized. Mark them initialized so that msan doesn't flag loads from
    // these buffers.
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(device_allocation.opaque(),
                                      allocation.size());
  }

  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                      assignment_->GetUniqueTopLevelOutputSlice());
  const BufferAllocation::Index result_index = result_slice.index();
  VLOG(3) << "result index: " << result_index;

  // Allocate profiling counters for each hlo instruction that we would like to
  // profile.  Allocate an additional profile counter for the entire
  // computation.
  std::vector<uint64> profile_counters(hlo_to_profile_idx_.size() + 1);

  std::vector<void*> buffer_pointers;
  for (auto& device_allocation : device_allocations) {
    buffer_pointers.push_back(device_allocation.opaque());
  }

  // Resolve functions for all the HLO instructions ahead of time.
  std::map<HloInstruction*, ComputeFunctionType> functions;
  for (auto& entry : *functions_names_) {
    tensorflow::mutex_lock lock(jit_mutex_);
    HloInstruction* instruction = entry.first;
    llvm::JITSymbol sym = jit_->FindSymbol(entry.second);
    TF_RET_CHECK(sym);
    InsertOrDie(&functions, instruction,
                reinterpret_cast<ComputeFunctionType>(sym.getAddress()));
  }

  // Map containing pointers to result buffers for each instruction.
  std::map<HloInstruction*, const void*> results;

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  std::list<HloInstruction*> pending;

  // Call the function for each HLO instruction in topological order.
  for (auto* instruction :
       module().entry_computation()->MakeInstructionPostOrder()) {
    // Parameters and constants have no functions associated with them. Instead
    // just copy the existing buffer into the map containing instruction
    // results..
    if (instruction->opcode() == HloOpcode::kParameter) {
      InsertOrDie(&results, instruction,
                  arguments[instruction->parameter_number()].opaque());
    } else if (instruction->opcode() == HloOpcode::kConstant) {
      unsigned char* aligned_data =
          FindOrDie(aligned_constants_, instruction).get();
      InsertOrDie(&results, instruction, aligned_data);
    } else {
      TF_RET_CHECK(instruction->opcode() == HloOpcode::kCall);
      pending.push_back(instruction);
    }
  }

  void** temps_array = buffer_pointers.data();
  uint64* profile_counters_array = profile_counters.data();
  auto* thread_pool = CHECK_NOTNULL(run_options->inter_op_thread_pool());
  tensorflow::mutex completion_queue_lock;
  tensorflow::condition_variable completion_queue_cv;
  std::deque<HloInstruction*> completion_queue;
  int64 instructions_in_flight = 0;
  while (!pending.empty() || instructions_in_flight > 0) {
    auto pending_it = pending.begin();
    while (pending_it != pending.end()) {
      HloInstruction* instruction = *pending_it;
      // Skip pending instructions whose operands aren't ready.
      if (std::any_of(instruction->operands().begin(),
                      instruction->operands().end(),
                      [&](HloInstruction* operand) {
                        return !ContainsKey(results, operand);
                      })) {
        ++pending_it;
        continue;
      }

      TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                          assignment_->GetUniqueTopLevelSlice(instruction));
      void* result_buffer =
          static_cast<char*>(temps_array[result_slice.index()]) +
          result_slice.offset();
      // We cannot use a move-only RAII type like std::unique_ptr because the
      // list of operands is allocated on the main thread and transferred to the
      // worker via the lambda passed to enqueue_function.  In order for the
      // lambda to take ownership, we would need to use generalized lambda
      // capture which is a feature new to C++14.
      auto operand_buffers = new const void*[instruction->operand_count()];
      std::transform(instruction->operands().begin(),
                     instruction->operands().end(), operand_buffers,
                     [&results](HloInstruction* operand) {
                       return FindOrDie(results, operand);
                     });
      auto function = FindOrDie(functions, instruction);
      // The thread pool entry takes ownership of |operand_buffers|.
      thread_pool->Schedule([instruction, &completion_queue,
                             &completion_queue_lock, &completion_queue_cv,
                             result_buffer, run_options, operand_buffers,
                             temps_array, profile_counters_array, function] {
        function(result_buffer, run_options, operand_buffers, temps_array,
                 profile_counters_array);
        delete[] operand_buffers;
        // Push the completed HLO instruction on the queue, the main thread
        // will pop it off and potentially launch more work which uses the
        // result.
        {
          tensorflow::mutex_lock l(completion_queue_lock);
          completion_queue.push_back(instruction);
          completion_queue_cv.notify_all();
        }
      });

      ++instructions_in_flight;
      pending_it = pending.erase(pending_it);
    }
    // Wait for a completed HLO instruction to be present in the queue.  We will
    // pop it out of the queue and make the result available to its users.
    HloInstruction* instruction;
    do {
      tensorflow::mutex_lock l(completion_queue_lock);
      if (completion_queue.empty()) {
        completion_queue_cv.wait(l);
      }
      if (!completion_queue.empty()) {
        instruction = completion_queue.front();
        completion_queue.pop_front();
        break;
      }
    } while (1);
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                        assignment_->GetUniqueTopLevelSlice(instruction));
    void* result_buffer =
        static_cast<char*>(temps_array[result_slice.index()]) +
        result_slice.offset();
    InsertOrDie(&results, instruction, result_buffer);
    --instructions_in_flight;
  }
  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  {
    tensorflow::mutex_lock lock(mutex_);
    double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));
    // The last profile counter is used for the computation as a whole.
    execution_profile_.set_compute_cycle_count(profile_counters.back());
  }
  if (hlo_execution_profile != nullptr) {
    hlo_execution_profile->set_total_cycles_executed(profile_counters.back());

    for (auto hlo_prof_idx : hlo_to_profile_idx_) {
      const HloInstruction* hlo = hlo_prof_idx.first;
      uint64 cycles_taken = profile_counters[hlo_prof_idx.second];
      hlo_execution_profile->AddProfileResult(hlo, cycles_taken);
    }
  }

  // Mark the buffers that are actually live (used in the output) when the
  // computation finishes executing.
  std::unordered_set<const void*> marked_addresses;
  MarkLiveAddressesInOutput(device_allocations[result_index].opaque(),
                            result_shape(), &marked_addresses);

  VLOG(3) << "Live addresses in output marking found "
          << marked_addresses.size() << " addresses:\n"
          << tensorflow::str_util::Join(
                 marked_addresses, ", ", [](string* out, const void* address) {
                   tensorflow::strings::StrAppend(
                       out, tensorflow::strings::Printf("%p", address));
                 });

  // Computation is done - deallocate temp buffers. Keep those marked
  // live because they are referenced by the output of the computation
  // and are needed by the service. They will be deallocated by the
  // service.
  for (auto i = 0; i < device_allocations.size(); ++i) {
    auto alloc = device_allocations[i];
    if (marked_addresses.count(alloc.opaque()) == 0 &&
        alloc.opaque() != nullptr) {
      VLOG(3) << "ParallelCpuExecutable deallocating buffer #" << i << " ["
              << alloc.opaque() << "]";
      TF_RETURN_IF_ERROR(memory_allocator->Deallocate(device_ordinal, &alloc));
    }
  }

  return device_allocations[result_index];
}

StatusOr<std::unique_ptr<ShapedBuffer>> ParallelCpuExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return Unimplemented(
      "ParallelCpuExecutable not supported yet with LocalService execution");
}

Status ParallelCpuExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    ShapedBuffer* result_buffer, HloExecutionProfile* hlo_execution_profile) {
  return Unimplemented(
      "preallocated result buffer not supported with ParallelCpuExecutable");
}

StatusOr<perftools::gputools::DeviceMemoryBase>
ParallelCpuExecutable::ExecuteAsyncOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments) {
  // TODO(b/30671675): Implement asynchronous execution mode.
  return Unimplemented(
      "Asynchronous execution on stream is not yet supported on CPU.");
}

}  // namespace cpu
}  // namespace xla
