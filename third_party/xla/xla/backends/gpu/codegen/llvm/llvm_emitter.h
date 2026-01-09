/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_EMITTER_H_
#define XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_EMITTER_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla::gpu {

// Emit a constant with a given number of element, given byte size of the
// element, given symbol name and content.
GpuExecutable::ConstantInfo AppendGlobalConstant(llvm::Module* module,
                                                 int64_t num_elements,
                                                 int64_t bytes_per_element,
                                                 absl::string_view symbol_name,
                                                 int allocation_idx,
                                                 DenseDataIntermediate content);

absl::StatusOr<ThunkSequence> EmitBitonicSortLLVMIR(
    const HloSortInstruction* sort, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context);

// Input = {static array, dynamic_dim0, dynamic_dim1}
// Output = {dynamic array(with dynamic dimension meta data at the end)}
// For a tensor with static dimension [2][<=5] and dynamic dimension [2][3]
// (`_` stands for padding)
// Input = {{1,2,3,_,_,4,5,6_,_}, 2, 3}
// Output = {{1,2,3,4,5,6,_,_,_,_,2,3}}

// pseudo code for padToStatic on a 2d array
//   ```
// void padToStatic(int** input, int** output, int threads_per_block,
//                  int meta_data_offset, int max_num_element,
//                  int static_dim0_size, int static_dim1_size) {
//   int* source_array = input[0];
//   int* dest_array = output[0];

//   // extract the dynamic dimension from the source array's metadata
//   int* dyn_dim0_size = source_array + meta_data_offset;
//   int* dyn_dim1_size = source_array + meta_data_offset + sizeof(int);

//   // only one thread need to store the dynamic index
//   int thread_id = GetThreadId();
//   int block_id = GetBlockId();
//   if (thread_id == 0 && block_id == 0) {
//     *output[1] = *dyn_dim0_size;
//     *output[2] = *dyn_dim1_size;
//   }

//   int dyn_element_total = 1;
//   dyn_element_total *= *dyn_dim0_size;
//   dyn_element_total *= *dyn_dim1_size;
//   linear_index = block_id * threads_per_block + thread_id;
//   if (linear_index < max_num_element) {
//     Index static_index =
//         delinerized(linerized_index, static_dim0_size, static_dim1_size);
//     if (linerized_index < dyn_element_total) {
//       Index dyn_index =
//           delinerized(linerized_index, *dyn_dim0_size, *dyn_dim1_size);
//       dest_array[dyn_index.dim0][dyn_index.dim1] =
//           source_array[static_index.dim0][static_index.dim1];
//     }
//   }
//   return;
// }
//   ```
absl::StatusOr<ThunkSequence> EmitPadToStaticLLVMIR(
    const HloCustomCallInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context);

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
// For a tensor with static dimension [2][<=5] and dynamic dimension [2][3]
// (`_` stands for padding)
// Input = {{1,2,3,4,5,6,_,_,_,_,2,3}}
// Output = {{1,2,3,_,_,4,5,6_,_}, 2, 3}

// pseudo code for sliceToDynamic on a 2d array
//   ```
// void sliceToDynamic(int** input, int** output, int threads_per_block,
//                  int meta_data_offset, int max_num_element,
//                  int static_dim0_size, int static_dim1_size) {
//   int* source_array = input[0];
//   int* dest_array = output[0];

//   // calculate the location where metadata needs to be inserted
//   int* dyn_dim0_size = dest_array + meta_data_offset;
//   int* dyn_dim1_size = dest_array + meta_data_offset + sizeof(int);

//   // only one thread need to store the dynamic index
//   int thread_id = GetThreadId();
//   int block_id = GetBlockId();
//   if (thread_id == 0 && block_id == 0) {
//     *dyn_dim0_size = *output[1];
//     *dyn_dim1_size = *output[2];
//   }

//   int dyn_element_total = 1;
//   dyn_element_total *= *dyn_dim0_size;
//   dyn_element_total *= *dyn_dim1_size;
//   linear_index = block_id * threads_per_block + thread_id;
//   if (linear_index < max_num_element) {
//     Index static_index =
//         delinerized(linerized_index, static_dim0_size, static_dim1_size);
//     if (linerized_index < dyn_element_total) {
//       Index dyn_index =
//           delinerized(linerized_index, *dyn_dim0_size, *dyn_dim1_size);
//       dest_array[static_index.dim0][static_index.dim1] =
//           source_array[dyn_index.dim0][dyn_index.dim1];
//     }
//   }
//   return;
// }
//   ```
absl::StatusOr<ThunkSequence> EmitSliceToDynamicLLVMIR(
    const HloCustomCallInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context);

absl::StatusOr<ThunkSequence> EmitRngGetAndUpdateStateLLVMIR(
    const HloRngGetAndUpdateStateInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_EMITTER_H_
