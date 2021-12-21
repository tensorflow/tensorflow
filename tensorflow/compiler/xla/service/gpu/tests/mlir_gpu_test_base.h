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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TESTS_MLIR_GPU_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TESTS_MLIR_GPU_TEST_BASE_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

class MlirGpuTestBase : public HloTestBase {
 public:
  MlirGpuTestBase();

  StatusOr<std::vector<std::vector<uint8_t>>> RunMlirTextWithHostBuffers(
      absl::string_view module_text,
      std::vector<absl::Span<uint8_t>> arguments);

  StatusOr<std::unique_ptr<Executable>> CompileMlirText(
      absl::string_view module_text);

  template <typename T>
  static absl::Span<uint8_t> ToUint8Span(std::vector<T>* v) {
    return absl::Span<uint8_t>(reinterpret_cast<uint8_t*>(v->data()),
                               v->size() * sizeof(T));
  }

  template <typename T>
  static absl::Span<const T> FromUint8Span(absl::Span<const uint8_t> span) {
    CHECK_EQ(0, span.size() % sizeof(T));
    return absl::Span<const T>(reinterpret_cast<const T*>(span.data()),
                               span.size() / sizeof(T));
  }

 private:
  StatusOr<std::vector<std::vector<uint8_t>>> RunMlirModuleWithHostBuffers(
      mlir::ModuleOp module, std::vector<absl::Span<uint8_t>> arguments);

  StatusOr<std::unique_ptr<Executable>> CompileMlirModule(mlir::ModuleOp module,
                                                          se::Stream* stream);

  StatusOr<ExecutionOutput> RunMlirModule(
      mlir::ModuleOp module, se::Stream* stream,
      absl::Span<const se::DeviceMemoryBase> arguments);

  StatusOr<mlir::OwningModuleRef> ParseMlirModule(absl::string_view module_text,
                                                  mlir::MLIRContext& context);

  std::unique_ptr<xla::Backend> backend_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TESTS_MLIR_GPU_TEST_BASE_H_
