/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_FLOAT8_FNUZ_IR_EMITTER_H_
#define XLA_SERVICE_FLOAT8_FNUZ_IR_EMITTER_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace float8_fnuz_ir_emitter {

// Convert the given floating point input to the output type. input_type must
// be one of BF16, F16, F32, and F64. output_type must be one of F8E4M3FNUZ and
// F8E5M2FNUZ.
StatusOr<llvm::Value*> EmitFloatingToF8fnuz(PrimitiveType input_type,
                                            llvm::Value* input_value,
                                            PrimitiveType output_type,
                                            llvm::IRBuilder<>* b);

// Convert the given floating point input to the output type. input_type must
// be one of F8E4M3FNUZ and F8E5M2FNUZ. output_type must be one of BF16, F16,
// F32, and F64.
StatusOr<llvm::Value*> EmitF8fnuzToFloating(PrimitiveType input_type,
                                            llvm::Value* f8_value,
                                            PrimitiveType output_type,
                                            llvm::IRBuilder<>* b,
                                            llvm::Module* module);
}  // namespace float8_fnuz_ir_emitter
}  // namespace xla

#endif  // XLA_SERVICE_FLOAT8_FNUZ_IR_EMITTER_H_
