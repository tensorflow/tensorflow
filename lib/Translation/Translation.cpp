//===- Translation.cpp - Translation registry -----------------------------===//
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

// Get the mutable static map between registered "to MLIR" translations and the
// TranslateToMLIRFunctions that perform those translations.
static llvm::StringMap<TranslateToMLIRFunction> &
getMutableTranslationToMLIRRegistry() {
  static llvm::StringMap<TranslateToMLIRFunction> translationToMLIRRegistry;
  return translationToMLIRRegistry;
}
// Get the mutable static map between registered "from MLIR" translations and
// the TranslateFromMLIRFunctions that perform those translations.
static llvm::StringMap<TranslateFromMLIRFunction> &
getMutableTranslationFromMLIRRegistry() {
  static llvm::StringMap<TranslateFromMLIRFunction> translationFromMLIRRegistry;
  return translationFromMLIRRegistry;
}

TranslateToMLIRRegistration::TranslateToMLIRRegistration(
    StringRef name, const TranslateToMLIRFunction &function) {
  auto &translationToMLIRRegistry = getMutableTranslationToMLIRRegistry();
  if (translationToMLIRRegistry.find(name) != translationToMLIRRegistry.end())
    llvm::report_fatal_error(
        "Attempting to overwrite an existing <to> function");
  assert(function && "Attempting to register an empty translate <to> function");
  translationToMLIRRegistry[name] = function;
}

TranslateFromMLIRRegistration::TranslateFromMLIRRegistration(
    StringRef name, const TranslateFromMLIRFunction &function) {
  auto &translationFromMLIRRegistry = getMutableTranslationFromMLIRRegistry();
  if (translationFromMLIRRegistry.find(name) !=
      translationFromMLIRRegistry.end())
    llvm::report_fatal_error(
        "Attempting to overwrite an existing <from> function");
  assert(function && "Attempting to register an empty translate <to> function");
  translationFromMLIRRegistry[name] = function;
}

// Merely add the const qualifier to the mutable registry so that external users
// cannot modify it.
const llvm::StringMap<TranslateToMLIRFunction> &
mlir::getTranslationToMLIRRegistry() {
  return getMutableTranslationToMLIRRegistry();
}

const llvm::StringMap<TranslateFromMLIRFunction> &
mlir::getTranslationFromMLIRRegistry() {
  return getMutableTranslationFromMLIRRegistry();
}
