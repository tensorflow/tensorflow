/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/codegen/ir_printing.h"

#include <memory>
#include <string>
#include <system_error>  // NOLINT
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/path.h"

namespace xla {

namespace {

class IRPrinterConfig : public mlir::PassManager::IRPrinterConfig {
 public:
  IRPrinterConfig(mlir::MLIRContext* context,
                  std::unique_ptr<llvm::raw_fd_ostream> log_stream);

  ~IRPrinterConfig() override;

  void printBeforeIfEnabled(mlir::Pass* pass, mlir::Operation* operation,
                            PrintCallbackFn printCallback) override;

  void printAfterIfEnabled(mlir::Pass* pass, mlir::Operation* operation,
                           PrintCallbackFn printCallback) override;

 private:
  std::unique_ptr<llvm::raw_fd_ostream> log_stream_;
  mlir::DiagnosticEngine::HandlerID callback_id_ = 0;
  mlir::MLIRContext* context_;
};

IRPrinterConfig::IRPrinterConfig(
    mlir::MLIRContext* context,
    std::unique_ptr<llvm::raw_fd_ostream> log_stream)
    : mlir::PassManager::IRPrinterConfig(
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/true,
          mlir::OpPrintingFlags()
              .enableDebugInfo(true, true)
              .printGenericOpForm()),
      log_stream_(std::move(log_stream)),
      context_(context) {
  context_->disableMultithreading();
  context_->printOpOnDiagnostic(true);
  context_->printStackTraceOnDiagnostic(true);

  callback_id_ =
      context_->getDiagEngine().registerHandler([this](mlir::Diagnostic& diag) {
        if (log_stream_ != nullptr) {
          absl::string_view severity;
          switch (diag.getSeverity()) {
            case mlir::DiagnosticSeverity::Error:
              severity = "ERROR";
              break;
            case mlir::DiagnosticSeverity::Warning:
              severity = "WARNING";
              break;
            case mlir::DiagnosticSeverity::Remark:
              severity = "REMARK";
              break;
            case mlir::DiagnosticSeverity::Note:
              severity = "NOTE";
              break;
            default:
              severity = "UNKNOWN";
              break;
          }

          *log_stream_ << "// " << severity << ": " << diag.str() << "\n";
        }
      });
}

IRPrinterConfig::~IRPrinterConfig() {
  if (callback_id_ != 0) {
    context_->getDiagEngine().eraseHandler(callback_id_);
  }
}

void IRPrinterConfig::printBeforeIfEnabled(mlir::Pass* pass,
                                           mlir::Operation* operation,
                                           PrintCallbackFn printCallback) {
  if (log_stream_) {
    printCallback(*log_stream_);
  }
}

void IRPrinterConfig::printAfterIfEnabled(mlir::Pass* pass,
                                          mlir::Operation* operation,
                                          PrintCallbackFn printCallback) {
  if (log_stream_) {
    printCallback(*log_stream_);
  }
}

}  // namespace

void EnableIRPrintingIfRequested(mlir::PassManager& pass_manager,
                                 mlir::MLIRContext* context,
                                 const HloModule& hlo_module,
                                 absl::string_view kernel_name,
                                 absl::string_view pass_manager_name) {
  const HloModuleConfig& hlo_config = hlo_module.config();
  bool should_dump_mlir_passes =
      hlo_config.debug_options().xla_enable_dumping() &&
      DumpingEnabledForHloModule(hlo_module) &&
      DumpingEnabledForEmitter(pass_manager_name, hlo_config.debug_options());
  if (!should_dump_mlir_passes) {
    return;
  }
  std::string outputs_dir = hlo_config.debug_options().xla_dump_to();
  if (outputs_dir == "sponge") {
    if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
      LOG(ERROR) << "Logging to sponge was requested, but failed to get test "
                    "undeclared outputs dir. Lets skip dumping triton passes.";
      LOG(ERROR) << "emitter " << pass_manager_name
                 << " is allowed to be dumped, but neither the environment "
                    "variable TEST_UNDECLARED_OUTPUTS_DIR nor the flag "
                    "--xla_dump_to is set, so the llvm dumps are disabled.";
      return;
    }
  }

  if (outputs_dir.empty()) {
    return;
  }
  const std::string basename =
      absl::StrCat(absl::string_view(tsl::io::Basename(hlo_module.name())), ".",
                   kernel_name, ".", pass_manager_name, ".txt");
  std::string path = tsl::io::JoinPath(outputs_dir, basename);
  std::error_code err;
  auto log_stream =
      std::make_unique<llvm::raw_fd_ostream>(path, err, llvm::sys::fs::OF_None);
  if (err) {
    LOG(ERROR) << "Failed to open the stream for triton passes logging to "
               << path << ": " << err.message();
    return;
  }

  auto config =
      std::make_unique<IRPrinterConfig>(context, std::move(log_stream));
  pass_manager.enableIRPrinting(std::move(config));
}

}  // namespace xla
