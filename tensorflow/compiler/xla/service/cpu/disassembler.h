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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DISASSEMBLER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DISASSEMBLER_H_

#include <memory>
#include <string>

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace cpu {

struct DisassemblerResult {
  DisassemblerResult(const string& text, size_t code_size_bytes)
      : text(text), code_size_bytes(code_size_bytes) {}

  // The dissassembled text sections of the object file.
  string text;
  // The total number of bytes of executable code in the object file.
  uint64_t code_size_bytes;
};

// Class for disassembling object files (and potentially other constructs) into
// x86/ARM assembly. Builds all the LLVM disassembly and instruction printing
// constructs from a given TargetMachine.
class Disassembler {
 public:
  explicit Disassembler(const llvm::TargetMachine& target_machine);

  // Returns a DisassemblerResult for the given object file, containing the
  // disassembled code.
  //
  // If we couldnt' retrieve a disassembler for this platform, an error status
  // is returned.
  StatusOr<DisassemblerResult> DisassembleObjectFile(
      const llvm::object::ObjectFile& object_file) const;

 private:
  const llvm::MCSubtargetInfo& subtarget_info_;
  std::unique_ptr<llvm::MCObjectFileInfo> objfile_info_;
  std::unique_ptr<llvm::MCContext> mc_context_;
  std::unique_ptr<llvm::MCDisassembler> disassembler_;
  std::unique_ptr<llvm::MCInstPrinter> inst_printer_;
  std::unique_ptr<llvm::MCInstrAnalysis> inst_analysis_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DISASSEMBLER_H_
