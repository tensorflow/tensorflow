//===- Location.cpp - MLIR Location Classes -------------------------------===//
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

#include "mlir/IR/Location.h"
#include "LocationDetail.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Location
//===----------------------------------------------------------------------===//

Location::Kind Location::getKind() const { return loc->kind; }

//===----------------------------------------------------------------------===//
// FileLineColLoc
//===----------------------------------------------------------------------===//

FileLineColLoc FileLineColLoc::get(StringRef filename, unsigned line,
                                   unsigned column, MLIRContext *context) {
  return get(Identifier::get(filename.empty() ? "-" : filename, context), line,
             column, context);
}

StringRef FileLineColLoc::getFilename() const {
  return static_cast<ImplType *>(loc)->filename;
}
unsigned FileLineColLoc::getLine() const {
  return static_cast<ImplType *>(loc)->line;
}
unsigned FileLineColLoc::getColumn() const {
  return static_cast<ImplType *>(loc)->column;
}

//===----------------------------------------------------------------------===//
// NameLoc
//===----------------------------------------------------------------------===//

NameLoc NameLoc::get(Identifier name, MLIRContext *context) {
  return get(name, UnknownLoc::get(context), context);
}

/// Return the name identifier.
Identifier NameLoc::getName() const {
  return static_cast<ImplType *>(loc)->name;
}

/// Return the child location.
Location NameLoc::getChildLoc() const {
  return static_cast<ImplType *>(loc)->child;
}

//===----------------------------------------------------------------------===//
// CallSiteLoc
//===----------------------------------------------------------------------===//

CallSiteLoc CallSiteLoc::get(Location name, ArrayRef<Location> frames,
                             MLIRContext *context) {
  assert(!frames.empty() && "required at least 1 frames");
  Location caller = frames.back();
  for (auto frame : llvm::reverse(frames.drop_back()))
    caller = CallSiteLoc::get(frame, caller, context);
  return CallSiteLoc::get(name, caller, context);
}

Location CallSiteLoc::getCallee() const {
  return static_cast<ImplType *>(loc)->callee;
}

Location CallSiteLoc::getCaller() const {
  return static_cast<ImplType *>(loc)->caller;
}

//===----------------------------------------------------------------------===//
// FusedLoc
//===----------------------------------------------------------------------===//

Location FusedLoc::get(ArrayRef<Location> locs, MLIRContext *context) {
  return get(locs, Attribute(), context);
}

ArrayRef<Location> FusedLoc::getLocations() const {
  return static_cast<ImplType *>(loc)->getLocations();
}

Attribute FusedLoc::getMetadata() const {
  return static_cast<ImplType *>(loc)->metadata;
}
