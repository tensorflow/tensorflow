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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_DEVICE_TARGET_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_DEVICE_TARGET_H_

#include <functional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/numerical_utils.h"

namespace mlir {
namespace quant {

class QuantizeContext;

using AdjacentOperations = llvm::SmallVectorImpl<Operation*>;
using QuantizedMultipliers = llvm::SmallVector<QuantizedMultiplier, 4>;
using QuantizedRanges = llvm::SmallVector<QuantizedRange, 4>;
using ScaleFn = std::function<LogicalResult(QuantizeContext*, Operation*,
                                            AdjacentOperations*, bool*)>;

using ScaleDecomposeFn =
    std::function<LogicalResult(Operation*, QuantizedMultipliers*,
                                QuantizedMultipliers*, QuantizedRanges*)>;

static const QuantizedMultiplier kUnitQuantizedMultiplier{1, 0};

enum class ScaleConstraintType {
  OutputInputSameScale,
  OutputInputFreeScale,
  CustomScale,
};

// Each kernel signature has its own specification for scales.
struct KernelSpec {
  // Scale constraint
  ScaleConstraintType type;

  // Custom function to derive the scales. Only available when the scale
  // constraint is `CustomScale`.
  ScaleFn scale_fn;
};

class KernelSpecs {
 public:
  using Signature = llvm::SmallVector<quant::AnyQuantizedType, 4>;

  // Returns the kernel specification for the kernel signature.
  Optional<KernelSpec> Find(const Signature& signature) const {
    auto spec_it = all_signatures_.find(signature);
    if (spec_it != all_signatures_.end()) {
      return spec_it->second;
    } else {
      return llvm::None;
    }
  }

  ScaleDecomposeFn GetDecomposeFn() const { return decompose_fn_; }

  // Adds the kernel signature with the kernel specification.
  LogicalResult Add(const Signature& signature, const KernelSpec& spec) {
    if (all_signatures_.insert({signature, spec}).second) return success();
    return failure();
  }

  KernelSpecs& WithSignature(const KernelSpecs::Signature& signature,
                             const ScaleFn& fn) {
    Add(signature, {ScaleConstraintType::CustomScale, fn});
    return *this;
  }

  KernelSpecs& WithImpl(const ScaleDecomposeFn& dfn) {
    decompose_fn_ = dfn;
    return *this;
  }

 private:
  // The signature is pattern match based.
  struct SignatureInfo : public llvm::DenseMapInfo<Signature> {
    static inline Signature getEmptyKey() { return {}; }
    static inline Signature getTombstoneKey() { return {nullptr}; }
    static unsigned getHashValue(Signature val) {
      return llvm::hash_combine_range(val.begin(), val.end());
    }
    static bool isEqual(Signature LHS, Signature RHS) {
      if (RHS == getEmptyKey()) return LHS == getEmptyKey();
      if (RHS == getTombstoneKey()) return LHS == getTombstoneKey();
      if (LHS.size() != RHS.size()) return false;
      for (auto arg : llvm::zip(LHS, RHS)) {
        if (std::get<0>(arg) != std::get<1>(arg)) return false;
      }
      return true;
    }
  };

  // Maps the signature to the kernel spec. Note that the matching is
  // pattern match based.
  llvm::DenseMap<Signature, KernelSpec, SignatureInfo> all_signatures_;

  // A method to compute the effective multipliers. This is independent on the
  // bits of the ports, thus all the signature shares the same here.
  ScaleDecomposeFn decompose_fn_;
};

class DeviceTarget {
 public:
  explicit DeviceTarget(MLIRContext* ctx);

  // Retrieves the kernel spec for the quant region op.
  Optional<KernelSpec> GetKernelSpec(
      llvm::StringRef kernel, const KernelSpecs::Signature& signature) const;

  // Retrieves the scale decomposition function for the quant region op.
  ScaleDecomposeFn GetDecomposeFn(quant::QuantizeRegionOp op) const;

  // converts specification to signature:
  // - UniformedQuantizedType -> AnyQuantizedType
  // - AnyQuantizedType (int) -> AnyQuantizedType
  // - Float -> {}
  static void AppendToSignature(Type spec, KernelSpecs::Signature* signature);

 protected:
  // Adds the kernel spec with the custom scale function for the kernel.
  LogicalResult RegisterKernel(llvm::StringRef kernel,
                               const KernelSpecs::Signature& signature,
                               const ScaleFn& fn, const ScaleDecomposeFn& dfn);

  // Adds the kernel spec with the scale constraint type for the kernel.
  LogicalResult RegisterKernel(llvm::StringRef kernel,
                               const KernelSpecs::Signature& signature,
                               const ScaleConstraintType constraint);

  // Adds the kernel with the name. Retrun an existing one if it has been
  // added before.
  KernelSpecs& RegisterKernel(llvm::StringRef kernel) { return specs_[kernel]; }

  // For "mulmat->add" type of kernels, convert the scales of all the ports to
  // multipliers.
  static LogicalResult DecomposeMultiplyAccumulateScale(
      Operation* op, quant::QuantizedMultipliers* input_multipliers,
      quant::QuantizedMultipliers* output_multipliers,
      quant::QuantizedRanges* output_ranges);

  // For "reshape" type of kernels.
  static LogicalResult DecomposeSameScale(
      Operation* op, quant::QuantizedMultipliers* input_multipliers,
      quant::QuantizedMultipliers* output_multipliers,
      quant::QuantizedRanges* output_ranges);

  // A set of parameters are required to build the signatures.
  FloatType f32_;
  IntegerType i8_, i32_;
  int64_t i8_min_, i8_max_, i32_min_, i32_max_;
  AnyQuantizedType any_, qi8_, qi8n_, qi32_;

 private:
  // Maps the kernel names to all the available kernels.
  llvm::StringMap<KernelSpecs> specs_;

  // Points to the global MLIRContext.
  MLIRContext* ctx_;
};

}  // namespace quant
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_DEVICE_TARGET_H_
