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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_ALIAS_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_ALIAS_ANALYSIS_H_

#include <unordered_map>

#include "external/llvm/include/llvm/IR/Module.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace llvm_ir {

// Helper functionality used to augment the LLVM IR emitted with alias-scope
// metadata.
class AliasAnalysis {
 public:
  AliasAnalysis(const HloModule& module, const BufferAssignment& assignment,
                llvm::LLVMContext* context)
      : module_(module), assignment_(assignment), context_(context) {}

  // Augments IrArray with aliasing information.
  void AddAliasingInformationToIrArray(const HloInstruction& hlo,
                                       llvm_ir::IrArray* array);

 private:
  // Returns a unique alias domain for this emitter.
  llvm::MDNode* GetAliasDomain();

  // Returns an alias.scope metadata node corresponding to a given buffer index.
  llvm::MDNode* GetAliasScopeMetadataForBuffer(
      BufferAllocation::Index buffer_index, llvm::MDNode* domain);

  // Returns a noalias metadata node corresponding to a given buffer index.
  //
  // |buffer_index| is the buffer index.
  //
  // |domain| corresponds to the alias scope domain as documented at
  // http://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
  //
  // |hlo| is the instruction we are computing a noalias set for.
  llvm::MDNode* GetNoaliasMetadataForBuffer(
      BufferAllocation::Index buffer_index, llvm::MDNode* domain,
      const BufferAssignment& assignment, const HloInstruction& hlo);

  // The HLO module we are compiling for.
  const HloModule& module_;

  // Assignment of the temporary buffers needed by the computation and their
  // shape information.
  const BufferAssignment& assignment_;

  // The LLVM context which we are using for IR emission.
  llvm::LLVMContext* context_;

  // Holds the alias domain for this computation.
  llvm::MDNode* alias_domain_ = nullptr;

  // Index in alias_scope_metadata_ and noalias_metadata_ for parameters
  // of the entry computation which have special aliasing properties.
  static constexpr int kParameterAliasSet = -1;

  // A map from a buffer index to metadata corresponding to its alias.scope
  // metadata.  The index kParameterAliasSet is used to hold aliasing
  // information for parameters.
  std::unordered_map<int, llvm::MDNode*> alias_scope_metadata_;

  // A map from a buffer index to metadata corresponding to its noalias
  // metadata.
  std::unordered_map<int, llvm::MDNode*> noalias_metadata_;
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_ALIAS_ANALYSIS_H_
