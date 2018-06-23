//===- MLIRContext.h - MLIR Global Context Class ----------------*- C++ -*-===//
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

#ifndef MLIR_IR_MLIRCONTEXT_H
#define MLIR_IR_MLIRCONTEXT_H

#include <memory>

namespace mlir {
  class MLIRContextImpl;

/// MLIRContext is the top-level object for a collection of MLIR modules.  It
/// holds immortal uniqued objects like types, and the tables used to unique
/// them.
///
/// MLIRContext gets a redundant "MLIR" prefix because otherwise it ends up with
/// a very generic name ("Context") and because it is uncommon for clients to
/// interact with it.
///
class MLIRContext {
  const std::unique_ptr<MLIRContextImpl> impl;
  MLIRContext(const MLIRContext&) = delete;
  void operator=(const MLIRContext&) = delete;
public:
  explicit MLIRContext();
  ~MLIRContext();

  // This is effectively private given that only MLIRContext.cpp can see the
  // MLIRContextImpl type.
  MLIRContextImpl &getImpl() const { return *impl.get(); }
};
} // end namespace mlir

#endif  // MLIR_IR_MLIRCONTEXT_H
