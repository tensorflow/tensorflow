/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_HLO_TRANSLATE_MHLO_TO_HLO_TRANSLATE_REGISTRATION_H_
#define XLA_HLO_TRANSLATE_MHLO_TO_HLO_TRANSLATE_REGISTRATION_H_

#include "llvm/Support/CommandLine.h"

// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_use_tuple_arg(
    "emit-use-tuple-args",
    llvm::cl::desc(
        "Emit HLO modules using tuples as args for the entry computation"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_return_tuple(
    "emit-return-tuple",
    llvm::cl::desc("Emit HLO modules with entry computations returning tuple"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> with_layouts(
    "with-layouts",
    llvm::cl::desc("Propagate layouts when translating MHLO->XLA HLO"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_layouts(
    "print-layouts", llvm::cl::desc("Print layouts in the generated HLO text"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_large_constants(
    "print-large-constants",
    llvm::cl::desc("Print large constants in the generated HLO text"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_sugar(
    "print-sugar",
    llvm::cl::desc(
        "Print async ops using syntactic sugar in the generated HLO text"),
    llvm::cl::init(true));

// NOLINTNEXTLINE
llvm::cl::opt<bool> via_builder(
    "via-builder", llvm::cl::desc("Translate MHLO->XLA HLO via XLA Builder"),
    llvm::cl::init(false));

#endif  // XLA_HLO_TRANSLATE_MHLO_TO_HLO_TRANSLATE_REGISTRATION_H_
