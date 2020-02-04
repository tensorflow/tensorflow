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

#include "tensorflow/core/framework/variant_op_registry.h"

#include <string>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

std::unordered_set<string>* UnaryVariantOpRegistry::PersistentStringStorage() {
  static std::unordered_set<string>* string_storage =
      new std::unordered_set<string>();
  return string_storage;
}

UnaryVariantOpRegistry::VariantDecodeFn* UnaryVariantOpRegistry::GetDecodeFn(
    StringPiece type_name) {
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
  decode_fns.insert(std::pair<StringPiece, VariantDecodeFn>(
      GetPersistentStringPiece(type_name), decode_fn));
}

bool DecodeUnaryVariant(Variant* variant) {
  CHECK_NOTNULL(variant);
  if (variant->TypeName().empty()) {
    VariantTensorDataProto* t = variant->get<VariantTensorDataProto>();
    if (t == nullptr || !t->metadata().empty() || !t->tensors().empty()) {
      // Malformed variant.
      return false;
    } else {
      // Serialization of an empty Variant.
      variant->clear();
      return true;
    }
  }
  UnaryVariantOpRegistry::VariantDecodeFn* decode_fn =
      UnaryVariantOpRegistry::Global()->GetDecodeFn(variant->TypeName());
  if (decode_fn == nullptr) {
    return false;
  }
  const string type_name = variant->TypeName();
  bool decoded = (*decode_fn)(variant);
  if (!decoded) return false;
  if (variant->TypeName() != type_name) {
    LOG(ERROR) << "DecodeUnaryVariant: Variant type_name before decoding was: "
               << type_name
               << " but after decoding was: " << variant->TypeName()
               << ".  Treating this as a failure.";
    return false;
  }
  return true;
}

// Add some basic registrations for use by others, e.g., for testing.

#define REGISTER_VARIANT_DECODE_TYPE(T) \
  REGISTER_UNARY_VARIANT_DECODE_FUNCTION(T, TF_STR(T));

// No encode/decode registered for std::complex<> and Eigen::half
// objects yet.
REGISTER_VARIANT_DECODE_TYPE(int);
REGISTER_VARIANT_DECODE_TYPE(float);
REGISTER_VARIANT_DECODE_TYPE(bool);
REGISTER_VARIANT_DECODE_TYPE(double);

#undef REGISTER_VARIANT_DECODE_TYPE

Status VariantDeviceCopy(
    const VariantDeviceCopyDirection direction, const Variant& from,
    Variant* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy_fn) {
  UnaryVariantOpRegistry::AsyncVariantDeviceCopyFn* device_copy_fn =
      UnaryVariantOpRegistry::Global()->GetDeviceCopyFn(direction,
                                                        from.TypeId());
  if (device_copy_fn == nullptr) {
    return errors::Internal(
        "No unary variant device copy function found for direction: ",
        direction, " and Variant type_index: ",
        port::MaybeAbiDemangle(from.TypeId().name()));
  }
  return (*device_copy_fn)(from, to, copy_fn);
}

namespace {
template <typename T>
Status DeviceCopyPrimitiveType(
    const T& in, T* out,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copier) {
  // Dummy copy, we don't actually bother copying to the device and back for
  // testing.
  *out = in;
  return Status::OK();
}
}  // namespace

#define REGISTER_VARIANT_DEVICE_COPY_TYPE(T)            \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      T, VariantDeviceCopyDirection::HOST_TO_DEVICE,    \
      DeviceCopyPrimitiveType<T>);                      \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      T, VariantDeviceCopyDirection::DEVICE_TO_HOST,    \
      DeviceCopyPrimitiveType<T>);                      \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      T, VariantDeviceCopyDirection::DEVICE_TO_DEVICE,  \
      DeviceCopyPrimitiveType<T>);

// No zeros_like registered for std::complex<> or Eigen::half objects yet.
REGISTER_VARIANT_DEVICE_COPY_TYPE(int);
REGISTER_VARIANT_DEVICE_COPY_TYPE(float);
REGISTER_VARIANT_DEVICE_COPY_TYPE(double);
REGISTER_VARIANT_DEVICE_COPY_TYPE(bool);

#undef REGISTER_VARIANT_DEVICE_COPY_TYPE

namespace {
template <typename T>
Status ZerosLikeVariantPrimitiveType(OpKernelContext* ctx, const T& t,
                                     T* t_out) {
  *t_out = T(0);
  return Status::OK();
}
}  // namespace

#define REGISTER_VARIANT_ZEROS_LIKE_TYPE(T)                             \
  REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP, \
                                           DEVICE_CPU, T,               \
                                           ZerosLikeVariantPrimitiveType<T>);

// No zeros_like registered for std::complex<> or Eigen::half objects yet.
REGISTER_VARIANT_ZEROS_LIKE_TYPE(int);
REGISTER_VARIANT_ZEROS_LIKE_TYPE(float);
REGISTER_VARIANT_ZEROS_LIKE_TYPE(double);
REGISTER_VARIANT_ZEROS_LIKE_TYPE(bool);

#undef REGISTER_VARIANT_ZEROS_LIKE_TYPE

namespace {
template <typename T>
Status AddVariantPrimitiveType(OpKernelContext* ctx, const T& a, const T& b,
                               T* out) {
  *out = a + b;
  return Status::OK();
}
}  // namespace

#define REGISTER_VARIANT_ADD_TYPE(T)                                           \
  REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_CPU, \
                                            T, AddVariantPrimitiveType<T>);

// No add registered for std::complex<> or Eigen::half objects yet.
REGISTER_VARIANT_ADD_TYPE(int);
REGISTER_VARIANT_ADD_TYPE(float);
REGISTER_VARIANT_ADD_TYPE(double);
REGISTER_VARIANT_ADD_TYPE(bool);

#undef REGISTER_VARIANT_ADD_TYPE

}  // namespace tensorflow
