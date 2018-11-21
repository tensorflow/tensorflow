//===- Translation.cpp - Translation registry -------------------*- C++ -*-===//
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
// Definitions of the translation registry.
//
//===----------------------------------------------------------------------===//

#include "mlir/Translation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;

// Get the mutable static map between translations registered and the
// TranslateFunctions that perform those translations.
static llvm::StringMap<TranslateFunction> &getMutableTranslationRegistry() {
  static llvm::StringMap<TranslateFunction> translationRegistry;
  return translationRegistry;
}

TranslateRegistration::TranslateRegistration(
    StringRef name, const TranslateFunction &function) {
  auto &translationRegistry = getMutableTranslationRegistry();
  if (translationRegistry.find(name) != translationRegistry.end())
    llvm::report_fatal_error("Attempting to overwrite an existing function");
  assert(function && "Attempting to register an empty translate function");
  translationRegistry[name] = function;
}

// Merely add the const qualifier to the mutable registry so that external users
// cannot modify it.
const llvm::StringMap<TranslateFunction> &mlir::getTranslationRegistry() {
  return getMutableTranslationRegistry();
}
