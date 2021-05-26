/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/xla_mlir_translate_cl.h"

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
llvm::cl::opt<bool> optimize_xla_hlo(
    "optimize-xla-hlo",
    llvm::cl::desc("Enable optimizations when translating XLA HLO -> LHLO"),
    llvm::cl::init(true));

// NOLINTNEXTLINE
llvm::cl::opt<bool> prefer_tf2xla(
    "prefer-tf2xla",
    llvm::cl::desc("Prefer tf2xla fallback for legalization to HLO over MLIR "
                   "legalizations."),
    llvm::cl::init(false));
