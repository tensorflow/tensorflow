/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_CONSTANTS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_CONSTANTS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/export/payloads.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"

namespace mlir::TFL::fb_export {

// How the elements of a constant are laid out in its TFLite buffer.
enum class ConstantEncoding {
  // Elements as stored by MLIR, one or more whole bytes each.
  kRaw,
  // 4 or 2 bit elements, stored one per byte by MLIR, bit-packed.
  kPacked4Bit,
  kPacked2Bit,
  // f16 or bf16 elements widened to f32. Used to fold a `tfl.cast` to f32 of
  // a constant while writing, without materializing the f32 data in memory.
  kWidenedToF32,
};

// A constant value to be written to a TFLite buffer.
struct Constant {
  ElementsAttr value;
  ConstantEncoding encoding = ConstantEncoding::kRaw;
};

// Returns the constant `op` produces, if it is a constant op or a `tfl.cast`
// from a f16/bf16 constant to f32 (which folds into a f32 constant). The
// encoding of sub-byte types is left as `kRaw`; see `GetEncoding`.
std::optional<Constant> GetConstant(Operation* op);

// Returns true if `op` is a constant whose uses all fold into f32 constants,
// so that it doesn't need to be exported itself.
bool IsOnlyUsedByFoldedCasts(Operation* op);

// Returns the encoding of a non-folded constant stored as `type`.
ConstantEncoding GetEncoding(tflite::TensorType type);

// Returns a fingerprint of the bytes `constant` is written as.
uint64_t GetFingerprint(const Constant& constant);

// Returns the number of bytes `constant` is written as.
uint64_t GetByteSize(const Constant& constant);

// Returns the resource blob holding the data of `constant`, if any.
AsmResourceBlob* GetResourceBlob(const Constant& constant);

// Returns an error if `constant` can't be written, e.g. because its storage
// doesn't match its type. Checking upfront keeps writing, which releases
// resource blobs, from failing halfway.
absl::Status CheckConstant(const Constant& constant);

// Writes the bytes of `constant` to `sink`.
absl::Status WriteConstant(const Constant& constant, ByteSink sink);

// A parameter tensor (scales or zero points) of a blockwise quantized tensor.
struct BlockwiseParameter {
  tflite::TensorType type;
  std::vector<int32_t> shape;
  std::string data;
};

// Serializes blockwise quantization scales. f32, f16 and bf16 scales are kept
// as is; other float types (e.g. f8E8M0FNU), which TFLite has no tensor type
// for, are widened to f32, which represents all their values exactly.
absl::StatusOr<BlockwiseParameter> SerializeBlockwiseScales(
    ElementsAttr scales);

// Serializes blockwise quantization zero points: integer ones as i32 and
// float ones (e.g. the -0.5 of a grid centered between integers) as f32.
absl::StatusOr<BlockwiseParameter> SerializeBlockwiseZeroPoints(
    ElementsAttr zero_points);

}  // namespace mlir::TFL::fb_export

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_CONSTANTS_H_
