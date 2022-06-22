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

#include "tensorflow/compiler/mlir/lite/quantization/device_target.h"

#include <algorithm>

#include "absl/types/optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/numerical_utils.h"

namespace mlir {
namespace quant {

constexpr int k8Bits = 8;
constexpr int k32Bits = 32;
constexpr unsigned kSigned = quant::QuantizationFlags::Signed;

DeviceTarget::DeviceTarget(MLIRContext* ctx) : ctx_(ctx) {
  f32_ = FloatType::getF32(ctx_);
  i8_ = IntegerType::get(ctx_, k8Bits);
  i8_min_ = QuantizedType::getDefaultMinimumForInteger(kSigned, k8Bits);
  i8_max_ = QuantizedType::getDefaultMaximumForInteger(kSigned, k8Bits);
  i32_ = IntegerType::get(ctx_, k32Bits);
  i32_min_ = QuantizedType::getDefaultMinimumForInteger(kSigned, k32Bits);
  i32_max_ = QuantizedType::getDefaultMaximumForInteger(kSigned, k32Bits);
  any_ = AnyQuantizedType();
  qi8_ = AnyQuantizedType::get(kSigned, i8_, f32_, i8_min_, i8_max_);
  qi8n_ = AnyQuantizedType::get(kSigned, i8_, f32_, i8_min_ + 1, i8_max_);
  qi32_ = AnyQuantizedType::get(kSigned, i32_, f32_, i32_min_, i32_max_);
  assert(qi8n_ == qi8n_);
}

Optional<KernelSpec> DeviceTarget::GetKernelSpec(
    llvm::StringRef kernel, const KernelSpecs::Signature& signature) const {
  auto kernel_specs_it = specs_.find(kernel);
  if (kernel_specs_it == specs_.end()) return llvm::None;
  return kernel_specs_it->getValue().Find(signature);
}

ScaleDecomposeFn DeviceTarget::GetDecomposeFn(QuantizeRegionOp op) const {
  auto kernel_specs_it = specs_.find(op.logical_kernel());
  if (kernel_specs_it == specs_.end()) return ScaleDecomposeFn(nullptr);
  return kernel_specs_it->second.GetDecomposeFn();
}

void DeviceTarget::AppendToSignature(Type spec,
                                     KernelSpecs::Signature* signature) {
  if (auto quant = spec.dyn_cast_or_null<UniformQuantizedType>()) {
    signature->push_back(AnyQuantizedType::get(
        quant.getFlags(), quant.getStorageType(), quant.getExpressedType(),
        quant.getStorageTypeMin(), quant.getStorageTypeMax()));
  } else if (auto any = spec.dyn_cast_or_null<AnyQuantizedType>()) {
    signature->push_back(any);
  } else {  // float
    signature->push_back(AnyQuantizedType());
  }
}

LogicalResult DeviceTarget::RegisterKernel(
    llvm::StringRef kernel, const KernelSpecs::Signature& signature,
    const ScaleFn& fn, const ScaleDecomposeFn& dfn) {
  return specs_[kernel].Add(signature, {ScaleConstraintType::CustomScale, fn});
}

namespace ph = std::placeholders;

LogicalResult DeviceTarget::RegisterKernel(
    llvm::StringRef kernel, const KernelSpecs::Signature& signature,
    const ScaleConstraintType constraint) {
  if (failed(specs_[kernel].Add(signature, {constraint, {}}))) return failure();
  switch (constraint) {
    case ScaleConstraintType::OutputInputSameScale:
      specs_[kernel].WithImpl(std::bind(&DeviceTarget::DecomposeSameScale,
                                        ph::_1, ph::_2, ph::_3, ph::_4));
      return success();
    default:
      return failure();
  }
}

LogicalResult DeviceTarget::DecomposeMultiplyAccumulateScale(
    Operation* op, quant::QuantizedMultipliers* input_multipliers,
    quant::QuantizedMultipliers* output_multipliers,
    quant::QuantizedRanges* output_ranges) {
  auto rop = llvm::dyn_cast<quant::QuantizeRegionOp>(op);
  if (!rop) return failure();

  llvm::SmallVector<Type, 4> input_specs, out_specs;
  for (auto spec : rop.input_specs()) {
    input_specs.push_back(spec.cast<TypeAttr>().getValue());
  }
  for (auto spec : rop.output_specs()) {
    out_specs.push_back(spec.cast<TypeAttr>().getValue());
  }

  auto in_spec = input_specs[0].dyn_cast<quant::UniformQuantizedType>();
  // TODO(fengliuai): handles the PerAxis QuantizedType.
  auto w_spec = input_specs[1].dyn_cast<quant::UniformQuantizedType>();
  auto b_spec = input_specs[2].dyn_cast<quant::UniformQuantizedType>();
  auto o_spec = out_specs[0].dyn_cast<quant::UniformQuantizedType>();
  if (!in_spec || !w_spec || !b_spec || !o_spec) return failure();

  double scale_product = in_spec.getScale() * w_spec.getScale();
  if (fabs(scale_product - b_spec.getScale()) >= 1e-6) return failure();

  // input multipliers
  input_multipliers->append(3, kUnitQuantizedMultiplier);

  // output multipliers
  double real_multiplier = scale_product / o_spec.getScale();
  output_multipliers->push_back(quant::QuantizeMultiplier(real_multiplier));

  // output ranges
  auto min = rop->getAttrOfType<FloatAttr>("min");
  auto max = rop->getAttrOfType<FloatAttr>("max");
  output_ranges->push_back(quant::CalculateQuantizedRange(
      o_spec.getScale(), o_spec.getZeroPoint(),
      (min ? std::optional<double>(min.getValueAsDouble()) : std::nullopt),
      (max ? std::optional<double>(max.getValueAsDouble()) : std::nullopt),
      o_spec.getStorageTypeMin(), o_spec.getStorageTypeMax()));

  return success();
}

LogicalResult DeviceTarget::DecomposeSameScale(
    Operation* op, quant::QuantizedMultipliers* input_multipliers,
    quant::QuantizedMultipliers* output_multipliers,
    quant::QuantizedRanges* output_ranges) {
  auto rop = llvm::dyn_cast<quant::QuantizeRegionOp>(op);
  if (!rop) return failure();

  // input multipliers
  for (int i = 0; i < op->getNumOperands(); ++i) {
    input_multipliers->push_back(kUnitQuantizedMultiplier);
  }

  // output multipliers
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_multipliers->push_back(kUnitQuantizedMultiplier);
  }

  auto o_spec = rop.output_specs()[0]
                    .cast<TypeAttr>()
                    .getValue()
                    .dyn_cast<quant::UniformQuantizedType>();
  if (!o_spec) return failure();

  // output ranges
  auto min = rop->getAttrOfType<FloatAttr>("min");
  auto max = rop->getAttrOfType<FloatAttr>("max");
  output_ranges->push_back(quant::CalculateQuantizedRange(
      o_spec.getScale(), o_spec.getZeroPoint(),
      (min ? std::optional<double>(min.getValueAsDouble()) : std::nullopt),
      (max ? std::optional<double>(max.getValueAsDouble()) : std::nullopt),
      o_spec.getStorageTypeMin(), o_spec.getStorageTypeMax()));

  return success();
}

}  // namespace quant
}  // namespace mlir
