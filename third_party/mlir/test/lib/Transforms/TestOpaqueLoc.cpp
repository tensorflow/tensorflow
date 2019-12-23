//===- TestOpaqueLoc.cpp - Pass to test opaque locations ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// Pass that changes locations to opaque locations for each operation.
/// It also takes all operations that are not function operations or
/// terminators and clones them with opaque locations which store the initial
/// locations.
struct TestOpaqueLoc : public ModulePass<TestOpaqueLoc> {

  /// A simple structure which is used for testing as an underlying location in
  /// OpaqueLoc.
  struct MyLocation {
    MyLocation() : id(42) {}
    MyLocation(int id) : id(id) {}
    int getId() { return id; }

    int id;
  };

  void runOnModule() override {
    std::vector<std::unique_ptr<MyLocation>> myLocs;
    int last_it = 0;

    getModule().walk([&](Operation *op) {
      myLocs.push_back(std::make_unique<MyLocation>(last_it++));

      Location loc = op->getLoc();

      /// Set opaque location without fallback location to test the
      /// corresponding get method.
      op->setLoc(
          OpaqueLoc::get<MyLocation *>(myLocs.back().get(), &getContext()));

      if (isa<FuncOp>(op) || op->isKnownTerminator())
        return;

      OpBuilder builder(op);

      /// Add the same operation but with fallback location to test the
      /// corresponding get method and serialization.
      Operation *op_cloned_1 = builder.clone(*op);
      op_cloned_1->setLoc(
          OpaqueLoc::get<MyLocation *>(myLocs.back().get(), loc));

      /// Add the same operation but with void* instead of MyLocation* to test
      /// getUnderlyingLocationOrNull method.
      Operation *op_cloned_2 = builder.clone(*op);
      op_cloned_2->setLoc(OpaqueLoc::get<void *>(nullptr, loc));
    });

    ScopedDiagnosticHandler diagHandler(&getContext(), [](Diagnostic &diag) {
      auto &os = llvm::outs();
      if (diag.getLocation().isa<OpaqueLoc>()) {
        MyLocation *loc = OpaqueLoc::getUnderlyingLocationOrNull<MyLocation *>(
            diag.getLocation());
        if (loc)
          os << "MyLocation: " << loc->id;
        else
          os << "nullptr";
      }
      os << ": " << diag << '\n';
      os.flush();
    });

    getModule().walk([&](Operation *op) { op->emitOpError(); });
  }
};

} // end anonymous namespace

static PassRegistration<TestOpaqueLoc>
    pass("test-opaque-loc", "Changes all leaf locations to opaque locations");
