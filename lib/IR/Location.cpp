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

Location::Kind Location::getKind() const { return loc->kind; }

UnknownLoc::UnknownLoc(Location::ImplType *ptr) : Location(ptr) {}

FileLineColLoc::FileLineColLoc(Location::ImplType *ptr) : Location(ptr) {}

StringRef FileLineColLoc::getFilename() const {
  return static_cast<ImplType *>(loc)->filename.getRef();
}
unsigned FileLineColLoc::getLine() const {
  return static_cast<ImplType *>(loc)->line;
}
unsigned FileLineColLoc::getColumn() const {
  return static_cast<ImplType *>(loc)->column;
}

NameLoc::NameLoc(Location::ImplType *ptr) : Location(ptr) {}

Identifier NameLoc::getName() const {
  return static_cast<ImplType *>(loc)->name;
}

CallSiteLoc::CallSiteLoc(Location::ImplType *ptr) : Location(ptr) {}

Location CallSiteLoc::getCallee() const {
  return static_cast<ImplType *>(loc)->callee;
}

Location CallSiteLoc::getCaller() const {
  return static_cast<ImplType *>(loc)->caller;
}

FusedLoc::FusedLoc(Location::ImplType *ptr) : Location(ptr) {}

ArrayRef<Location> FusedLoc::getLocations() const {
  return static_cast<ImplType *>(loc)->getLocations();
}

Attribute FusedLoc::getMetadata() const {
  return static_cast<ImplType *>(loc)->metadata;
}
