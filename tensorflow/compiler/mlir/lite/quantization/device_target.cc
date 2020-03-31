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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace quant {

constexpr int k8Bits = 8;
constexpr unsigned kSigned = quant::QuantizationFlags::Signed;

DeviceTarget::DeviceTarget(MLIRContext* ctx) : ctx_(ctx) {
  f32_ = FloatType::getF32(ctx_);
  i8_ = IntegerType::get(k8Bits, ctx_);
  i8_min_ = QuantizedType::getDefaultMinimumForInteger(kSigned, k8Bits);
  i8_max_ = QuantizedType::getDefaultMaximumForInteger(kSigned, k8Bits);
  any_ = AnyQuantizedType();
  qi8_ = AnyQuantizedType::get(kSigned, i8_, f32_, i8_min_, i8_max_);
  qi8n_ = AnyQuantizedType::get(kSigned, i8_, f32_, i8_min_ + 1, i8_max_);
  assert(qi8n_ == qi8n_);
}

Optional<KernelSpec> DeviceTarget::Get(QuantizeRegionOp op) const {
  auto kernel_specs_it = specs_.find(op.logical_kernel());
  if (kernel_specs_it == specs_.end()) return llvm::None;

  KernelSpecs::Signature signature;
  signature.reserve(op.input_specs().size() + op.output_specs().size());
  AppendToSignature(op.input_specs(), &signature);
  AppendToSignature(op.output_specs(), &signature);
  return kernel_specs_it->getValue().Find(signature);
}

LogicalResult DeviceTarget::RegisterKernel(
    llvm::StringRef kernel, const KernelSpecs::Signature& signature,
    const ScaleFn& fn) {
  return specs_[kernel].Add(signature, {ScaleConstraintType::CustomScale, fn});
}

LogicalResult DeviceTarget::RegisterKernel(
    llvm::StringRef kernel, const KernelSpecs::Signature& signature,
    const ScaleConstraintType constraint) {
  return specs_[kernel].Add(signature, {constraint, {}});
}

void DeviceTarget::AppendToSignature(ArrayAttr specs_attr,
                                     KernelSpecs::Signature* signature) const {
  for (auto attr : specs_attr) {
    Type spec = attr.cast<TypeAttr>().getValue();
    if (auto quant = spec.dyn_cast<UniformQuantizedType>()) {
      signature->push_back(AnyQuantizedType::get(
          quant.getFlags(), quant.getStorageType(), quant.getExpressedType(),
          quant.getStorageTypeMin(), quant.getStorageTypeMax()));
    } else if (auto any = spec.dyn_cast<AnyQuantizedType>()) {
      signature->push_back(any);
    } else {  // float
      signature->push_back({});
    }
  }
}

}  // namespace quant
}  // namespace mlir
