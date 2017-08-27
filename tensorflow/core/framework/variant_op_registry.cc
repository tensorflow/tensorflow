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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"

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

Status GetUnaryVariantShape(const Tensor& variant_tensor, TensorShape* shape) {
  CHECK_EQ(variant_tensor.dtype(), DT_VARIANT);
  CHECK_EQ(variant_tensor.dims(), 0);
  // Use a mutable Variant because shape_fn will first call
  // MaybeDecodeAndGet, which in turn may mutate the underlying object
  // (if a Decode is called).
  const Variant& v = variant_tensor.scalar<Variant>()();
  UnaryVariantOpRegistry::VariantShapeFn* shape_fn =
      UnaryVariantOpRegistry::Global()->GetShapeFn(v.TypeName());
  if (shape_fn == nullptr) {
    return errors::Internal(
        "No unary variant shape function found for Variant type_name: ",
        v.TypeName());
  }
  return (*shape_fn)(v, shape);
}

UnaryVariantOpRegistry::VariantDecodeFn* UnaryVariantOpRegistry::GetDecodeFn(
    const string& type_name) {
  auto found = decode_fns.find(type_name);
  if (found == decode_fns.end()) return nullptr;
  return &found->second;
}

void UnaryVariantOpRegistry::RegisterDecodeFn(
    const string& type_name, const VariantDecodeFn& decode_fn) {
  CHECK(!type_name.empty()) << "Need a valid name for UnaryVariantDecode";
  VariantDecodeFn* existing = GetDecodeFn(type_name);
  CHECK_EQ(existing, nullptr)
      << "Unary VariantDecodeFn for type_name: " << type_name
      << " already registered";
  decode_fns.insert(std::pair<string, VariantDecodeFn>(type_name, decode_fn));
}

bool DecodeUnaryVariant(Variant* variant) {
  UnaryVariantOpRegistry::VariantDecodeFn* decode_fn =
      UnaryVariantOpRegistry::Global()->GetDecodeFn(variant->TypeName());
  if (decode_fn == nullptr) {
    return false;
  }
  return (*decode_fn)(variant);
}

// Add some basic registrations for use by others, e.g., for testing.

namespace {
string MaybeRemoveTFPrefix(const StringPiece& str) {
  return str.starts_with("::tensorflow::") ? str.substr(14).ToString()
                                           : str.ToString();
}
}  // namespace

#define REGISTER_VARIANT_DECODE_TYPE(T) \
  REGISTER_UNARY_VARIANT_DECODE_FUNCTION(T, MaybeRemoveTFPrefix(TF_STR(T)));

// No encode/decode registered for std::complex<> and Eigen::half
// objects yet.
TF_CALL_INTEGRAL_TYPES(REGISTER_VARIANT_DECODE_TYPE);
TF_CALL_float(REGISTER_VARIANT_DECODE_TYPE);
TF_CALL_double(REGISTER_VARIANT_DECODE_TYPE);
TF_CALL_bool(REGISTER_VARIANT_DECODE_TYPE);

#undef REGISTER_VARIANT_DECODE_TYPE

}  // namespace tensorflow
