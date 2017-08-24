/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// static
UnaryVariantOpRegistry* UnaryVariantOpRegistry::Global() {
  static UnaryVariantOpRegistry* global_unary_variant_op_registry =
      new UnaryVariantOpRegistry;
  return global_unary_variant_op_registry;
}

UnaryVariantOpRegistry::VariantShapeFn* UnaryVariantOpRegistry::GetShapeFn(
    const string& type_name) {
  auto found = shape_fns.find(type_name);
  if (found == shape_fns.end()) return nullptr;
  return &found->second;
}

void UnaryVariantOpRegistry::RegisterShapeFn(const string& type_name,
                                             const VariantShapeFn& shape_fn) {
  CHECK(!type_name.empty()) << "Need a valid name for UnaryVariantShape";
  VariantShapeFn* existing = GetShapeFn(type_name);
  CHECK_EQ(existing, nullptr)
      << "Unary VariantShapeFn for type_name: " << type_name
      << " already registered";
  shape_fns.insert(std::pair<string, VariantShapeFn>(type_name, shape_fn));
}

Status GetUnaryVariantShape(Tensor variant_tensor, TensorShape* shape) {
  CHECK_EQ(variant_tensor.dtype(), DT_VARIANT);
  CHECK_EQ(variant_tensor.dims(), 0);
  // Use a mutable Variant because shape_fn will first call
  // MaybeDecodeAndGet, which in turn may mutate the underlying object
  // (if a Decode is called).
  Variant& v = variant_tensor.scalar<Variant>()();
  UnaryVariantOpRegistry::VariantShapeFn* shape_fn =
      UnaryVariantOpRegistry::Global()->GetShapeFn(v.TypeName());
  if (shape_fn == nullptr) {
    return errors::Internal(
        "No unary variant shape function found for Variant type_name: ",
        v.TypeName());
  }
  return (*shape_fn)(&v, shape);
}

}  // namespace tensorflow
