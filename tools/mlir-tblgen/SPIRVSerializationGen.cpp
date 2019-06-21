//===- SPIRVSerializationGen.cpp - SPIR-V serialization utility generator -===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// SPIRVSerializationGen generates common utility functions for SPIR-V
// serialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::formatv;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::tblgen::Operator;

// Writes the following function to `os`:
//   inline uint32_t getOpcode(<op-class-name>) { return <opcode>; }
static void emitGetOpcodeFunction(const llvm::Record &record,
                                  const Operator &op, raw_ostream &os) {
  if (llvm::isa<llvm::UnsetInit>(record.getValueInit("opcode")))
    return;

  os << formatv("inline uint32_t getOpcode({0}) {{ return {1}u; }\n",
                op.getQualCppClassName(), record.getValueAsInt("opcode"));
}

static bool emitSerializationUtils(const RecordKeeper &recordKeeper,
                                   raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Serialization Utilities", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("SPV_Op");
  for (const auto *def : defs) {
    Operator op(def);
    emitGetOpcodeFunction(*def, op, os);
  }

  return false;
}

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDefs("gen-spirv-serial",
                "Generate SPIR-V serialization utility definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return emitSerializationUtils(records, os);
                });
