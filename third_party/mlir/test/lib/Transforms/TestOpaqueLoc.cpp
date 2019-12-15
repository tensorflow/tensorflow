//===- TestOpaqueLoc.cpp - Pass to test opaque locations ------------------===//
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
