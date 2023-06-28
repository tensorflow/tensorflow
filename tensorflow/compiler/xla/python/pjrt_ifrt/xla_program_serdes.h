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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_XLA_PROGRAM_SERDES_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_XLA_PROGRAM_SERDES_H_

#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/python/ifrt/serdes.h"

namespace xla {
namespace ifrt {

// Library that provides stable serialization and deserialization of
// `xla::ifrt::XlaProgram`. Both serialization and deserialization require
// linking in this library.
//
// Serialization:
// ```
// TF_ASSIGN_OR_RETURN(Serialized serialized, Serialize(xla_program));
// ```
//
// Deserialization:
// ```
// auto deserialize_options =
//     std::make_unique<XlaDeserializeProgramOptions>(&mlir_context);
// TF_ASSIGN_OR_RETURN(
//     auto deserialized,
//     Deserialize(serialized, std::move(deserialize_options)));
// auto xla_program = llvm::dyn_cast<XlaProgram>(deserialized);
// ```

struct XlaDeserializeProgramOptions
    : llvm::RTTIExtends<XlaDeserializeProgramOptions, DeserializeOptions> {
  XlaDeserializeProgramOptions() = default;
  explicit XlaDeserializeProgramOptions(mlir::MLIRContext* mlir_context)
      : mlir_context(mlir_context) {}

  mlir::MLIRContext* mlir_context;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_XLA_PROGRAM_SERDES_H_
