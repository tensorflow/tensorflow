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

std::unordered_set<string>* UnaryVariantOpRegistry::PersistentStringStorage() {
  static std::unordered_set<string>* string_storage =
      new std::unordered_set<string>();
  return string_storage;
}

// static
UnaryVariantOpRegistry* UnaryVariantOpRegistry::Global() {
  static UnaryVariantOpRegistry* global_unary_variant_op_registry =
      new UnaryVariantOpRegistry;
  return global_unary_variant_op_registry;
}

UnaryVariantOpRegistry::VariantShapeFn* UnaryVariantOpRegistry::GetShapeFn(
    StringPiece type_name) {
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
  shape_fns.insert(std::pair<StringPiece, VariantShapeFn>(
      GetPersistentStringPiece(type_name), shape_fn));
}

Status GetUnaryVariantShape(const Tensor& variant_tensor, TensorShape* shape) {
  CHECK_EQ(variant_tensor.dtype(), DT_VARIANT);
  CHECK_EQ(variant_tensor.dims(), 0);
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

// Add some basic registrations for use by others, e.g., for testing.
namespace {
template <typename T>
Status ScalarShape(const T&, TensorShape* shape) {
  *shape = TensorShape({});
  return Status::OK();
}
}  // namespace

#define REGISTER_VARIANT_SHAPE_TYPE(T) \
  REGISTER_UNARY_VARIANT_SHAPE_FUNCTION(T, TF_STR(T), ScalarShape<T>);

// No encode/shape registered for std::complex<> and Eigen::half
// objects yet.
REGISTER_VARIANT_SHAPE_TYPE(int);
REGISTER_VARIANT_SHAPE_TYPE(float);
REGISTER_VARIANT_SHAPE_TYPE(bool);
REGISTER_VARIANT_SHAPE_TYPE(double);

#undef REGISTER_VARIANT_SHAPE_TYPE

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

UnaryVariantOpRegistry::AsyncVariantDeviceCopyFn*
UnaryVariantOpRegistry::GetDeviceCopyFn(
    const VariantDeviceCopyDirection direction, StringPiece type_name) {
  auto found = device_copy_fns.find(std::make_pair(direction, type_name));
  if (found == device_copy_fns.end()) return nullptr;
  return &found->second;
}

void UnaryVariantOpRegistry::RegisterDeviceCopyFn(
    const VariantDeviceCopyDirection direction, const string& type_name,
    const AsyncVariantDeviceCopyFn& device_copy_fn) {
  CHECK(!type_name.empty()) << "Need a valid name for UnaryVariantDeviceCopy";
  AsyncVariantDeviceCopyFn* existing = GetDeviceCopyFn(direction, type_name);
  CHECK_EQ(existing, nullptr)
      << "UnaryVariantDeviceCopy for direction: " << direction
      << " and type_name: " << type_name << " already registered";
  device_copy_fns.insert(
      std::pair<std::pair<VariantDeviceCopyDirection, StringPiece>,
                AsyncVariantDeviceCopyFn>(
          std::make_pair(direction, GetPersistentStringPiece(type_name)),
          device_copy_fn));
}

Status VariantDeviceCopy(
    const VariantDeviceCopyDirection direction, const Variant& from,
    Variant* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy_fn) {
  UnaryVariantOpRegistry::AsyncVariantDeviceCopyFn* device_copy_fn =
      UnaryVariantOpRegistry::Global()->GetDeviceCopyFn(direction,
                                                        from.TypeName());
  if (device_copy_fn == nullptr) {
    return errors::Internal(
        "No unary variant device copy function found for direction: ",
        direction, " and Variant type_name: ", from.TypeName());
  }
  return (*device_copy_fn)(from, to, copy_fn);
}

// Special casing UnaryOpFn per op and per device.
UnaryVariantOpRegistry::VariantUnaryOpFn* UnaryVariantOpRegistry::GetUnaryOpFn(
    VariantUnaryOp op, StringPiece device, StringPiece type_name) {
  auto found = unary_op_fns.find(std::make_tuple(op, device, type_name));
  if (found == unary_op_fns.end()) return nullptr;
  return &found->second;
}

void UnaryVariantOpRegistry::RegisterUnaryOpFn(
    VariantUnaryOp op, const string& device, const string& type_name,
    const VariantUnaryOpFn& unary_op_fn) {
  CHECK(!type_name.empty()) << "Need a valid name for UnaryVariantUnaryOp";
  VariantUnaryOpFn* existing = GetUnaryOpFn(op, device, type_name);
  CHECK_EQ(existing, nullptr)
      << "Unary VariantUnaryOpFn for type_name: " << type_name
      << " already registered for device type: " << device;
  unary_op_fns.insert(
      std::pair<std::tuple<VariantUnaryOp, StringPiece, StringPiece>,
                VariantUnaryOpFn>(
          std::make_tuple(op, GetPersistentStringPiece(device),
                          GetPersistentStringPiece(type_name)),
          unary_op_fn));
}

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
                                           DEVICE_CPU, T, TF_STR(T),    \
                                           ZerosLikeVariantPrimitiveType<T>);

// No zeros_like registered for std::complex<> or Eigen::half objects yet.
REGISTER_VARIANT_ZEROS_LIKE_TYPE(int);
REGISTER_VARIANT_ZEROS_LIKE_TYPE(float);
REGISTER_VARIANT_ZEROS_LIKE_TYPE(double);
REGISTER_VARIANT_ZEROS_LIKE_TYPE(bool);

#undef REGISTER_VARIANT_ZEROS_LIKE_TYPE

// Special casing BinaryOpFn per op and per device.
UnaryVariantOpRegistry::VariantBinaryOpFn*
UnaryVariantOpRegistry::GetBinaryOpFn(VariantBinaryOp op, StringPiece device,
                                      StringPiece type_name) {
  auto found = binary_op_fns.find(std::make_tuple(op, device, type_name));
  if (found == binary_op_fns.end()) return nullptr;
  return &found->second;
}

void UnaryVariantOpRegistry::RegisterBinaryOpFn(
    VariantBinaryOp op, const string& device, const string& type_name,
    const VariantBinaryOpFn& add_fn) {
  CHECK(!type_name.empty()) << "Need a valid name for UnaryVariantBinaryOp";
  VariantBinaryOpFn* existing = GetBinaryOpFn(op, device, type_name);
  CHECK_EQ(existing, nullptr)
      << "Unary VariantBinaryOpFn for type_name: " << type_name
      << " already registered for device type: " << device;
  binary_op_fns.insert(
      std::pair<std::tuple<VariantBinaryOp, StringPiece, StringPiece>,
                VariantBinaryOpFn>(
          std::make_tuple(op, GetPersistentStringPiece(device),
                          GetPersistentStringPiece(type_name)),
          add_fn));
}

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
                                            T, TF_STR(T),                      \
                                            AddVariantPrimitiveType<T>);

// No add registered for std::complex<> or Eigen::half objects yet.
REGISTER_VARIANT_ADD_TYPE(int);
REGISTER_VARIANT_ADD_TYPE(float);
REGISTER_VARIANT_ADD_TYPE(double);
REGISTER_VARIANT_ADD_TYPE(bool);

#undef REGISTER_VARIANT_ADD_TYPE

}  // namespace tensorflow
