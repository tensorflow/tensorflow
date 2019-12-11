//===- IRPrinting.cpp -----------------------------------------------------===//
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

#include "PassDetail.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SHA1.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
//===----------------------------------------------------------------------===//
// OperationFingerPrint
//===----------------------------------------------------------------------===//

/// A unique fingerprint for a specific operation, and all of it's internal
/// operations.
class OperationFingerPrint {
public:
  OperationFingerPrint(Operation *topOp) {
    llvm::SHA1 hasher;

    // Hash each of the operations based upon their mutable bits:
    topOp->walk([&](Operation *op) {
      //   - Operation pointer
      addDataToHash(hasher, op);
      //   - Attributes
      addDataToHash(hasher,
                    op->getAttrList().getDictionary().getAsOpaquePointer());
      //   - Blocks in Regions
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          addDataToHash(hasher, &block);
          for (BlockArgument *arg : block.getArguments())
            addDataToHash(hasher, arg);
        }
      }
      //   - Location
      addDataToHash(hasher, op->getLoc().getAsOpaquePointer());
      //   - Operands
      for (Value *operand : op->getOperands())
        addDataToHash(hasher, operand);
      //   - Successors
      for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i)
        addDataToHash(hasher, op->getSuccessor(i));
    });
    hash = hasher.result();
  }

  bool operator==(const OperationFingerPrint &other) const {
    return hash == other.hash;
  }
  bool operator!=(const OperationFingerPrint &other) const {
    return !(*this == other);
  }

private:
  template <typename T> void addDataToHash(llvm::SHA1 &hasher, const T &data) {
    hasher.update(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&data), sizeof(T)));
  }

  SmallString<20> hash;
};

//===----------------------------------------------------------------------===//
// IRPrinter
//===----------------------------------------------------------------------===//

class IRPrinterInstrumentation : public PassInstrumentation {
public:
  IRPrinterInstrumentation(std::unique_ptr<PassManager::IRPrinterConfig> config)
      : config(std::move(config)) {}

private:
  /// Instrumentation hooks.
  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;

  /// Configuration to use.
  std::unique_ptr<PassManager::IRPrinterConfig> config;

  /// The following is a set of fingerprints for operations that are currently
  /// being operated on in a pass. This field is only used when the
  /// configuration asked for change detection.
  DenseMap<Pass *, OperationFingerPrint> beforePassFingerPrints;
};
} // end anonymous namespace

/// Returns true if the given pass is hidden from IR printing.
static bool isHiddenPass(Pass *pass) {
  return isAdaptorPass(pass) || isa<VerifierPass>(pass);
}

static void printIR(Operation *op, bool printModuleScope, raw_ostream &out,
                    OpPrintingFlags flags) {
  // Check to see if we are printing the top-level module.
  auto module = dyn_cast<ModuleOp>(op);
  if (module && !op->getBlock())
    return module.print(out << "\n", flags);

  // Otherwise, check to see if we are not printing at module scope.
  if (!printModuleScope)
    return op->print(out << "\n", flags.useLocalScope());

  // Otherwise, we are printing at module scope.
  out << " ('" << op->getName() << "' operation";
  if (auto symbolName =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    out << ": @" << symbolName.getValue();
  out << ")\n";

  // Find the top-level module operation.
  auto *topLevelOp = op;
  while (auto *parentOp = topLevelOp->getParentOp())
    topLevelOp = parentOp;

  // Check to see if the top-level operation is actually a module in the case of
  // invalid-ir.
  if (auto module = dyn_cast<ModuleOp>(topLevelOp))
    module.print(out, flags);
  else
    topLevelOp->print(out, flags);
}

/// Instrumentation hooks.
void IRPrinterInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  if (isHiddenPass(pass))
    return;
  // If the config asked to detect changes, record the current fingerprint.
  if (config->shouldPrintAfterOnlyOnChange())
    beforePassFingerPrints.try_emplace(pass, op);

  config->printBeforeIfEnabled(pass, op, [&](raw_ostream &out) {
    out << formatv("*** IR Dump Before {0} ***", pass->getName());
    printIR(op, config->shouldPrintAtModuleScope(), out, OpPrintingFlags());
    out << "\n\n";
  });
}

void IRPrinterInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  if (isHiddenPass(pass))
    return;
  // If the config asked to detect changes, compare the current fingerprint with
  // the previous.
  if (config->shouldPrintAfterOnlyOnChange()) {
    auto fingerPrintIt = beforePassFingerPrints.find(pass);
    assert(fingerPrintIt != beforePassFingerPrints.end() &&
           "expected valid fingerprint");
    // If the fingerprints are the same, we don't print the IR.
    if (fingerPrintIt->second == OperationFingerPrint(op)) {
      beforePassFingerPrints.erase(fingerPrintIt);
      return;
    }
    beforePassFingerPrints.erase(fingerPrintIt);
  }

  config->printAfterIfEnabled(pass, op, [&](raw_ostream &out) {
    out << formatv("*** IR Dump After {0} ***", pass->getName());
    printIR(op, config->shouldPrintAtModuleScope(), out, OpPrintingFlags());
    out << "\n\n";
  });
}

void IRPrinterInstrumentation::runAfterPassFailed(Pass *pass, Operation *op) {
  if (isAdaptorPass(pass))
    return;
  if (config->shouldPrintAfterOnlyOnChange())
    beforePassFingerPrints.erase(pass);

  config->printAfterIfEnabled(pass, op, [&](raw_ostream &out) {
    out << formatv("*** IR Dump After {0} Failed ***", pass->getName());
    printIR(op, config->shouldPrintAtModuleScope(), out,
            OpPrintingFlags().printGenericOpForm());
    out << "\n\n";
  });
}

//===----------------------------------------------------------------------===//
// IRPrinterConfig
//===----------------------------------------------------------------------===//

/// Initialize the configuration.
PassManager::IRPrinterConfig::IRPrinterConfig(bool printModuleScope,
                                              bool printAfterOnlyOnChange)
    : printModuleScope(printModuleScope),
      printAfterOnlyOnChange(printAfterOnlyOnChange) {}
PassManager::IRPrinterConfig::~IRPrinterConfig() {}

/// A hook that may be overridden by a derived config that checks if the IR
/// of 'operation' should be dumped *before* the pass 'pass' has been
/// executed. If the IR should be dumped, 'printCallback' should be invoked
/// with the stream to dump into.
void PassManager::IRPrinterConfig::printBeforeIfEnabled(
    Pass *pass, Operation *operation, PrintCallbackFn printCallback) {
  // By default, never print.
}

/// A hook that may be overridden by a derived config that checks if the IR
/// of 'operation' should be dumped *after* the pass 'pass' has been
/// executed. If the IR should be dumped, 'printCallback' should be invoked
/// with the stream to dump into.
void PassManager::IRPrinterConfig::printAfterIfEnabled(
    Pass *pass, Operation *operation, PrintCallbackFn printCallback) {
  // By default, never print.
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

namespace {
/// Simple wrapper config that allows for the simpler interface defined above.
struct BasicIRPrinterConfig : public PassManager::IRPrinterConfig {
  BasicIRPrinterConfig(
      std::function<bool(Pass *, Operation *)> shouldPrintBeforePass,
      std::function<bool(Pass *, Operation *)> shouldPrintAfterPass,
      bool printModuleScope, bool printAfterOnlyOnChange, raw_ostream &out)
      : IRPrinterConfig(printModuleScope, printAfterOnlyOnChange),
        shouldPrintBeforePass(shouldPrintBeforePass),
        shouldPrintAfterPass(shouldPrintAfterPass), out(out) {
    assert((shouldPrintBeforePass || shouldPrintAfterPass) &&
           "expected at least one valid filter function");
  }

  void printBeforeIfEnabled(Pass *pass, Operation *operation,
                            PrintCallbackFn printCallback) final {
    if (shouldPrintBeforePass && shouldPrintBeforePass(pass, operation))
      printCallback(out);
  }

  void printAfterIfEnabled(Pass *pass, Operation *operation,
                           PrintCallbackFn printCallback) final {
    if (shouldPrintAfterPass && shouldPrintAfterPass(pass, operation))
      printCallback(out);
  }

  /// Filter functions for before and after pass execution.
  std::function<bool(Pass *, Operation *)> shouldPrintBeforePass;
  std::function<bool(Pass *, Operation *)> shouldPrintAfterPass;

  /// The stream to output to.
  raw_ostream &out;
};
} // end anonymous namespace

/// Add an instrumentation to print the IR before and after pass execution,
/// using the provided configuration.
void PassManager::enableIRPrinting(std::unique_ptr<IRPrinterConfig> config) {
  addInstrumentation(
      std::make_unique<IRPrinterInstrumentation>(std::move(config)));
}

/// Add an instrumentation to print the IR before and after pass execution.
void PassManager::enableIRPrinting(
    std::function<bool(Pass *, Operation *)> shouldPrintBeforePass,
    std::function<bool(Pass *, Operation *)> shouldPrintAfterPass,
    bool printModuleScope, bool printAfterOnlyOnChange, raw_ostream &out) {
  enableIRPrinting(std::make_unique<BasicIRPrinterConfig>(
      std::move(shouldPrintBeforePass), std::move(shouldPrintAfterPass),
      printModuleScope, printAfterOnlyOnChange, out));
}
