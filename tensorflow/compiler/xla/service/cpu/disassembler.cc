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

#include "tensorflow/compiler/xla/service/cpu/disassembler.h"

#include <stdint.h>
#include <algorithm>
// IWYU pragma: no_include <system_error>
#include <type_traits>
#include <vector>

#include "absl/strings/str_format.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace cpu {

Disassembler::Disassembler(const llvm::TargetMachine& target_machine)
    : subtarget_info_(*target_machine.getMCSubtargetInfo()) {
  objfile_info_.reset(new llvm::MCObjectFileInfo());
  mc_context_.reset(new llvm::MCContext(target_machine.getMCAsmInfo(),
                                        target_machine.getMCRegisterInfo(),
                                        objfile_info_.get()));
  disassembler_.reset(target_machine.getTarget().createMCDisassembler(
      subtarget_info_, *mc_context_));
  inst_printer_.reset(target_machine.getTarget().createMCInstPrinter(
      target_machine.getTargetTriple(),
      /*SyntaxVariant=*/1,  // Use Intel syntax.
      *target_machine.getMCAsmInfo(), *target_machine.getMCInstrInfo(),
      *target_machine.getMCRegisterInfo()));
  inst_analysis_.reset(target_machine.getTarget().createMCInstrAnalysis(
      target_machine.getMCInstrInfo()));
}

// This code is based on llvm-objdump in llvm/tools.
StatusOr<DisassemblerResult> Disassembler::DisassembleObjectFile(
    const llvm::object::ObjectFile& object_file) const {
  if (disassembler_ == nullptr) {
    return NotFound("could not find a disassembler for this platform");
  }

  std::string buffer_string;
  llvm::raw_string_ostream ostream(buffer_string);
  uint64_t code_size_bytes = 0;

  // Iterate through sections. Disassemble symbols of the text section(s).
  for (auto& section : object_file.sections()) {
    if (!section.isText()) {
      continue;
    }

    // Gather symbols from the section.
    std::vector<llvm::object::SymbolRef> symbols;
    for (auto& symbol : object_file.symbols()) {
      if (section.containsSymbol(symbol)) {
        symbols.push_back(symbol);
      }
    }

    // Sort the symbols in increasing address order.
    absl::c_sort(symbols, [](const llvm::object::SymbolRef& a,
                             const llvm::object::SymbolRef& b) {
      // getAddress returns a Expected object. Assert there is no error
      // before extracting the address.
      llvm::Expected<uint64_t> a_address_or_error = a.getAddress();
      CHECK(a_address_or_error);
      llvm::Expected<uint64_t> b_address_or_error = b.getAddress();
      CHECK(b_address_or_error);
      return a_address_or_error.get() < b_address_or_error.get();
    });

    // Construct ArrayRef pointing to section contents.
    llvm::StringRef section_content_string;
    if (section.getContents(section_content_string)) {
      continue;
    }
    llvm::ArrayRef<uint8_t> section_content_bytes(
        reinterpret_cast<const uint8*>(section_content_string.data()),
        section_content_string.size());

    // Use int types from LLVM (eg, uint64_t) for values passed to and returned
    // from the LLVM API. These values map to different types in LLVM and
    // XLA (unsigned long vs unsigned long long).
    uint64_t section_address = section.getAddress();
    uint64_t section_size = section.getSize();

    // Iterate through symbols in increasing address order and disassemble each
    // one.
    for (int i = 0; i < symbols.size(); ++i) {
      auto symbol = symbols[i];
      llvm::Expected<uint64_t> address = symbol.getAddress();
      CHECK(address);
      uint64_t start_index = address.get() - section_address;

      // End of symbol is either the end of the section or the start of the next
      // symbol.
      uint64_t end_index;
      if (i < symbols.size() - 1) {
        llvm::Expected<uint64_t> next_address = symbols[i + 1].getAddress();
        CHECK(next_address);
        end_index = std::min(section_size, next_address.get());
      } else {
        end_index = section_size;
      }

      // Skip zero-length symbols.
      if (start_index == end_index) {
        continue;
      }

      llvm::Expected<llvm::StringRef> name_or_error = symbol.getName();
      TF_RET_CHECK(name_or_error);
      ostream << name_or_error.get().str() << ":\n";

      // Update the code size statistic.
      code_size_bytes += end_index - start_index;

      // Disassemble symbol instruction-by-instruction.
      uint64_t index = start_index;
      while (index < end_index) {
        llvm::MCInst instruction;
        uint64_t size;
        llvm::MCDisassembler::DecodeStatus decode_status =
            disassembler_->getInstruction(instruction, size,
                                          section_content_bytes.slice(index),
                                          /*Address=*/section_address + index,
                                          /*VStream=*/llvm::nulls(),
                                          /*CStream=*/llvm::nulls());
        // If we fail to disassemble, then we must skip past this address.
        if (size == 0) {
          size = 1;
        }

        ostream << absl::StrFormat("0x%08lx", index) << " ";

        if (decode_status == llvm::MCDisassembler::Success) {
          // For branches, try to determine the actual address and emit it as an
          // annotation.
          string annotation;
          if (inst_analysis_ &&
              (inst_analysis_->isUnconditionalBranch(instruction) ||
               inst_analysis_->isConditionalBranch(instruction))) {
            uint64_t target;
            if (inst_analysis_->evaluateBranch(
                    instruction, section_address + index, size, target)) {
              annotation = absl::StrFormat("[0x%08lx]", target);
            }
          }
          inst_printer_->printInst(&instruction, ostream, annotation.c_str(),
                                   subtarget_info_);
        } else {
          ostream << " <unknown>";
        }

        ostream << "\n";
        index += size;
      }
    }
  }

  ostream.flush();
  return DisassemblerResult{
      string(buffer_string.data(), buffer_string.length()), code_size_bytes};
}

}  // namespace cpu
}  // namespace xla
