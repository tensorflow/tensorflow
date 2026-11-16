/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// A TableGen driver that dumps the records of a .td file as JSON, the same
// output as `llvm-tblgen --dump-json`. mlir-tblgen has no JSON backend, and
// llvm-tblgen is not visible outside LLVM, so this binary stands in for it.
// The operator documentation generator reads the TFL op definitions through
// it; see the tensorflow_lite_ops_json target in BUILD.

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

namespace {

bool DumpJson(llvm::raw_ostream& os, const llvm::RecordKeeper& records) {
  llvm::EmitJSON(records, os);
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return llvm::TableGenMain(argv[0], &DumpJson);
}
