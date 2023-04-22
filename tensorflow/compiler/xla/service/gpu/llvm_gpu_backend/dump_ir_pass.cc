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

#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/dump_ir_pass.h"

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

// Pass which dumps the IR of a module into a file.
//
// Because it is implemented as a FunctionPass (IR is dumped
// function-by-function) rather than as a ModulePass the resulting IR is not
// valid (missing metadata, for example) but is still useful for inspection.
// The pass needs to be a FunctionPass rather than a ModulePass because
// inserting ModulePasses is disruptive to LLVM's pass manager.  For sequential
// FunctionPasses (also SCC passes, etc) the pass manager executes the passes
// sequentially on each function (SCC, etc).  Inserting a ModulePass between
// FunctionPasses acts as a barrier forcing the FunctionPasses to execute fully
// across all functions prior to advancing to the next pass.  For some reason
// this results in different generated code resulting in an undesirable
// Heisenberg effect when dumping the IR.
class DumpIrPass : public llvm::FunctionPass {
 public:
  explicit DumpIrPass(const string &output_filename)
      : llvm::FunctionPass(id_), output_filename_(output_filename) {}

  bool doInitialization(llvm::Module &M) override {
    out_.reset(new llvm::raw_fd_ostream(llvm::StringRef(output_filename_), ec_,
                                        llvm::sys::fs::OF_None));
    if (ec_) {
      LOG(FATAL) << "Unable to open " << output_filename_
                 << " to dump LLVM IR: " << ec_.message();
    }
    return false;
  }

  bool runOnFunction(llvm::Function &Function) override {
    Function.print(*out_);
    return false;
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool doFinalization(llvm::Module &M) override {
    out_->close();
    return false;
  }

 private:
  static char id_;
  string output_filename_;
  std::error_code ec_;
  std::unique_ptr<llvm::raw_fd_ostream> out_;
};

char DumpIrPass::id_ = 0;

void IrDumpingPassManager::run(llvm::Module &module) {
  for (int i = 0; i < passes_.size(); ++i) {
    llvm::Pass *P = passes_[i];
    if (dump_ir_) {
      const llvm::PassInfo *PI =
          llvm::PassRegistry::getPassRegistry()->getPassInfo(P->getPassID());
      const string basename = ReplaceFilenameExtension(
          absl::string_view(tensorflow::io::Basename(input_filename_)),
          absl::StrFormat(
              "pass-%02d.before.%s.ll", i,
              absl::string_view(PI == nullptr ? "unknown"
                                              : PI->getPassArgument().data())));
      llvm::legacy::PassManager::add(
          new DumpIrPass(tensorflow::io::JoinPath(output_dir_, basename)));
    }
    llvm::legacy::PassManager::add(P);
  }
  passes_.clear();
  llvm::legacy::PassManager::run(module);
}

}  // namespace gpu
}  // namespace xla
