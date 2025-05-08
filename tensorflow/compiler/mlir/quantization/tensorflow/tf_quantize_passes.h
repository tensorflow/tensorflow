/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_TF_QUANTIZE_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_TF_QUANTIZE_PASSES_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace tensorflow {
namespace quantization {

// mlir_dump_file_prefix is an optional field that is used for debugging to save
// mlir dump files.
void AddQuantizeQatPasses(mlir::OpPassManager &pm,
                          const QuantizationOptions &quantization_options,
                          std::optional<const absl::string_view>
                              mlir_dump_file_prefix = std::nullopt);

void AddQuantizePtqDynamicRangePasses(
    mlir::OpPassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix =
        std::nullopt);

void AddQuantizeWeightOnlyPasses(
    mlir::OpPassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix =
        std::nullopt);

void AddQuantizePtqPreCalibrationPasses(
    mlir::OpPassManager &pm, const QuantizationOptions &quantization_options);

void AddQuantizePtqPostCalibrationPasses(
    mlir::OpPassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix =
        std::nullopt);

}  // namespace quantization
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_TF_QUANTIZE_PASSES_H_
