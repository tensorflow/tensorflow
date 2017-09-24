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

#ifndef TENSORFLOW_FRAMEWORK_VARIANT_OP_REGISTRY_H_
#define TENSORFLOW_FRAMEWORK_VARIANT_OP_REGISTRY_H_

#include <string>
#include <unordered_set>
#include <vector>

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {

class OpKernelContext;
// A global UnaryVariantOpRegistry is used to hold callback functions
// for different variant types.  To be used by ShapeOp, RankOp, and
// SizeOp, decoding, etc.

enum VariantUnaryOp {
  INVALID_VARIANT_UNARY_OP = 0,
  ZEROS_LIKE_VARIANT_UNARY_OP = 1,
  CONJ_VARIANT_UNARY_OP = 2,
};

enum VariantBinaryOp {
  INVALID_VARIANT_BINARY_OP = 0,
  ADD_VARIANT_BINARY_OP = 1,
};

class UnaryVariantOpRegistry {
 public:
  typedef std::function<Status(const Variant& v, TensorShape*)> VariantShapeFn;
  typedef std::function<bool(Variant*)> VariantDecodeFn;
  typedef std::function<Status(OpKernelContext*, const Variant&, Variant*)>
      VariantUnaryOpFn;
  typedef std::function<Status(OpKernelContext*, const Variant&, const Variant&,
                               Variant*)>
      VariantBinaryOpFn;

  // Add a shape lookup function to the registry.
  void RegisterShapeFn(const string& type_name, const VariantShapeFn& shape_fn);

  // Returns nullptr if no shape function was found for the given TypeName.
  VariantShapeFn* GetShapeFn(StringPiece type_name);

  // Add a decode function to the registry.
  void RegisterDecodeFn(const string& type_name,
                        const VariantDecodeFn& decode_fn);

  // Returns nullptr if no decode function was found for the given TypeName.
  VariantDecodeFn* GetDecodeFn(StringPiece type_name);

  // Add a unary op function to the registry.
  void RegisterUnaryOpFn(VariantUnaryOp op, const string& device,
                         const string& type_name,
                         const VariantUnaryOpFn& unary_op_fn);

  // Returns nullptr if no unary op function was found for the given
  // op, device, and TypeName.
  VariantUnaryOpFn* GetUnaryOpFn(VariantUnaryOp op, StringPiece device,
                                 StringPiece type_name);

  // Add a binary op function to the registry.
  void RegisterBinaryOpFn(VariantBinaryOp op, const string& device,
                          const string& type_name,
                          const VariantBinaryOpFn& add_fn);

  // Returns nullptr if no binary op function was found for the given
  // op, device and TypeName.
  VariantBinaryOpFn* GetBinaryOpFn(VariantBinaryOp op, StringPiece device,
                                   StringPiece type_name);

  // Get a pointer to a global UnaryVariantOpRegistry object
  static UnaryVariantOpRegistry* Global();

  // Get a pointer to a global persistent string storage object.
  // ISO/IEC C++ working draft N4296 clarifies that insertion into an
  // std::unordered_set does not invalidate memory locations of
  // *values* inside the set (though it may invalidate existing
  // iterators).  In other words, one may safely point a StringPiece to
  // a value in the set without that StringPiece being invalidated by
  // future insertions.
  static std::unordered_set<string>* PersistentStringStorage();

 private:
  std::unordered_map<StringPiece, VariantShapeFn, StringPiece::Hasher>
      shape_fns;
  std::unordered_map<StringPiece, VariantDecodeFn, StringPiece::Hasher>
      decode_fns;

  // Map std::tuple<Op, device, type_name> to function.
  struct TupleHash {
    template <typename Op>
    std::size_t operator()(
        const std::tuple<Op, StringPiece, StringPiece>& x) const {
      // The hash of an enum is just its value as a std::size_t.
      std::size_t ret = static_cast<std::size_t>(std::get<0>(x));
      StringPiece::Hasher sp_hasher;
      ret = Hash64Combine(ret, sp_hasher(std::get<1>(x)));
      ret = Hash64Combine(ret, sp_hasher(std::get<2>(x)));
      return ret;
    }
  };
  std::unordered_map<std::tuple<VariantUnaryOp, StringPiece, StringPiece>,
                     VariantUnaryOpFn, TupleHash>
      unary_op_fns;
  std::unordered_map<std::tuple<VariantBinaryOp, StringPiece, StringPiece>,
                     VariantBinaryOpFn, TupleHash>
      binary_op_fns;

  // Find or insert a string into a persistent string storage
  // container; return the StringPiece pointing to the permanent
  // string location.
  static StringPiece GetPersistentStringPiece(const string& str) {
    const auto string_storage = PersistentStringStorage();
    auto found = string_storage->find(str);
    if (found == string_storage->end()) {
      auto inserted = string_storage->insert(str);
      return StringPiece(*inserted.first);
    } else {
      return StringPiece(*found);
    }
  }
};

// Gets a TensorShape from a Tensor containing a scalar Variant.
// Returns an Internal error if the Variant does not have a registered shape
// function, or if it's a serialized Variant that cannot be decoded.
//
// REQUIRES:
//   variant_tensor.dtype() == DT_VARIANT
//   variant_tensor.dims() == 0
//
Status GetUnaryVariantShape(const Tensor& variant_tensor, TensorShape* shape);

// Decodes the Variant whose data_type has a registered decode
// function.  Returns an Internal error if the Variant does not have a
// registered decode function, or if the decoding function fails.
//
// REQUIRES:
//   variant is not null.
//
bool DecodeUnaryVariant(Variant* variant);

// Sets *v_out = unary_op(v).  The variant v must have a registered
// UnaryOp function for the given Device.  Returns an Internal error
// if v does not have a registered unary_op function for this device, or if
// UnaryOp fails.
//
// REQUIRES:
//   v_out is not null.
//
template <typename Device>
Status UnaryOpVariant(OpKernelContext* ctx, VariantUnaryOp op, const Variant& v,
                      Variant* v_out) {
  const string& device = DeviceName<Device>::value;
  UnaryVariantOpRegistry::VariantUnaryOpFn* unary_op_fn =
      UnaryVariantOpRegistry::Global()->GetUnaryOpFn(op, device, v.TypeName());
  if (unary_op_fn == nullptr) {
    return errors::Internal(
        "No unary variant unary_op function found for unary variant op enum: ",
        op, " Variant type_name: ", v.TypeName(), " for device type: ", device);
  }
  return (*unary_op_fn)(ctx, v, v_out);
}

// Sets *out = binary_op(a, b).  The variants a and b must be the same type
// and have a registered binary_op function for the given Device.  Returns an
// Internal error if a and b are not the same type_name or if
// if a does not have a registered op function for this device, or if
// BinaryOp fails.
//
// REQUIRES:
//   out is not null.
//
template <typename Device>
Status BinaryOpVariants(OpKernelContext* ctx, VariantBinaryOp op,
                        const Variant& a, const Variant& b, Variant* out) {
  if (a.TypeName() != b.TypeName()) {
    return errors::Internal(
        "BianryOpVariants: Variants a and b have different "
        "type names: '",
        a.TypeName(), "' vs. '", b.TypeName(), "'");
  }
  const string& device = DeviceName<Device>::value;
  UnaryVariantOpRegistry::VariantBinaryOpFn* binary_op_fn =
      UnaryVariantOpRegistry::Global()->GetBinaryOpFn(op, device, a.TypeName());
  if (binary_op_fn == nullptr) {
    return errors::Internal(
        "No unary variant binary_op function found for binary variant op "
        "enum: ",
        op, " Variant type_name: '", a.TypeName(),
        "' for device type: ", device);
  }
  return (*binary_op_fn)(ctx, a, b, out);
}

namespace variant_op_registry_fn_registration {

template <typename T>
class UnaryVariantShapeRegistration {
 public:
  typedef std::function<Status(const T& t, TensorShape*)> LocalVariantShapeFn;

  UnaryVariantShapeRegistration(const string& type_name,
                                const LocalVariantShapeFn& shape_fn) {
    auto wrapped_fn = [type_name, shape_fn](const Variant& v,
                                            TensorShape* s) -> Status {
      const T* t = v.get<T>();
      if (t == nullptr) {
        return errors::Internal(
            "VariantShapeFn: Could not access object, type_name: ", type_name);
      }
      return shape_fn(*t, s);
    };
    UnaryVariantOpRegistry::Global()->RegisterShapeFn(type_name, wrapped_fn);
  }
};

template <typename T>
class UnaryVariantDecodeRegistration {
 public:
  UnaryVariantDecodeRegistration(const string& type_name) {
    // The Variant is passed by pointer because it should be
    // mutable: get below may Decode the variant, which
    // is a self-mutating behavior.  The variant is not modified in
    // any other way.
    auto wrapped_fn = [type_name](Variant* v) -> bool {
      CHECK_NOTNULL(v);
      VariantTensorDataProto* t = v->get<VariantTensorDataProto>();
      if (t == nullptr) {
        return false;
      }
      Variant decoded = T();
      VariantTensorData data(*t);
      if (!decoded.Decode(data)) {
        return false;
      }
      *v = std::move(decoded);
      return true;
    };
    UnaryVariantOpRegistry::Global()->RegisterDecodeFn(type_name, wrapped_fn);
  }
};

template <typename T>
class UnaryVariantUnaryOpRegistration {
  typedef std::function<Status(OpKernelContext* ctx, const T& t, T* t_out)>
      LocalVariantUnaryOpFn;

 public:
  UnaryVariantUnaryOpRegistration(VariantUnaryOp op, const string& device,
                                  const string& type_name,
                                  const LocalVariantUnaryOpFn& unary_op_fn) {
    auto wrapped_fn = [type_name, unary_op_fn](OpKernelContext* ctx,
                                               const Variant& v,
                                               Variant* v_out) -> Status {
      CHECK_NOTNULL(v_out);
      *v_out = T();
      if (v.get<T>() == nullptr) {
        return errors::Internal(
            "VariantUnaryOpFn: Could not access object, type_name: ",
            type_name);
      }
      const T& t = *v.get<T>();
      T* t_out = v_out->get<T>();
      return unary_op_fn(ctx, t, t_out);
    };
    UnaryVariantOpRegistry::Global()->RegisterUnaryOpFn(op, device, type_name,
                                                        wrapped_fn);
  }
};

template <typename T>
class UnaryVariantBinaryOpRegistration {
  typedef std::function<Status(OpKernelContext* ctx, const T& a, const T& b,
                               T* out)>
      LocalVariantBinaryOpFn;

 public:
  UnaryVariantBinaryOpRegistration(VariantBinaryOp op, const string& device,
                                   const string& type_name,
                                   const LocalVariantBinaryOpFn& binary_op_fn) {
    auto wrapped_fn = [type_name, binary_op_fn](
                          OpKernelContext* ctx, const Variant& a,
                          const Variant& b, Variant* out) -> Status {
      CHECK_NOTNULL(out);
      *out = T();
      if (a.get<T>() == nullptr) {
        return errors::Internal(
            "VariantBinaryOpFn: Could not access object 'a', type_name: ",
            type_name);
      }
      if (b.get<T>() == nullptr) {
        return errors::Internal(
            "VariantBinaryOpFn: Could not access object 'b', type_name: ",
            type_name);
      }
      const T& t_a = *a.get<T>();
      const T& t_b = *b.get<T>();
      T* t_out = out->get<T>();
      return binary_op_fn(ctx, t_a, t_b, t_out);
    };
    UnaryVariantOpRegistry::Global()->RegisterBinaryOpFn(op, device, type_name,
                                                         wrapped_fn);
  }
};

};  // namespace variant_op_registry_fn_registration

// Register a unary shape variant function with the signature:
//    Status ShapeFn(const T& t, TensorShape* s);
// to Variants having TypeName type_name.
#define REGISTER_UNARY_VARIANT_SHAPE_FUNCTION(T, type_name, shape_function)    \
  REGISTER_UNARY_VARIANT_SHAPE_FUNCTION_UNIQ_HELPER(__COUNTER__, T, type_name, \
                                                    shape_function)

#define REGISTER_UNARY_VARIANT_SHAPE_FUNCTION_UNIQ_HELPER(ctr, T, type_name, \
                                                          shape_function)    \
  REGISTER_UNARY_VARIANT_SHAPE_FUNCTION_UNIQ(ctr, T, type_name, shape_function)

#define REGISTER_UNARY_VARIANT_SHAPE_FUNCTION_UNIQ(ctr, T, type_name,          \
                                                   shape_function)             \
  static variant_op_registry_fn_registration::UnaryVariantShapeRegistration<T> \
      register_unary_variant_op_shape_registration_fn_##ctr(type_name,         \
                                                            shape_function)

// Register a unary decode variant function for the given type.
#define REGISTER_UNARY_VARIANT_DECODE_FUNCTION(T, type_name) \
  REGISTER_UNARY_VARIANT_DECODE_FUNCTION_UNIQ_HELPER(__COUNTER__, T, type_name)

#define REGISTER_UNARY_VARIANT_DECODE_FUNCTION_UNIQ_HELPER(ctr, T, type_name) \
  REGISTER_UNARY_VARIANT_DECODE_FUNCTION_UNIQ(ctr, T, type_name)

#define REGISTER_UNARY_VARIANT_DECODE_FUNCTION_UNIQ(ctr, T, type_name)        \
  static variant_op_registry_fn_registration::UnaryVariantDecodeRegistration< \
      T>                                                                      \
      register_unary_variant_op_decoder_fn_##ctr(type_name)

// Register a unary unary_op variant function with the signature:
//    Status UnaryOpFn(OpKernelContext* ctx, const T& t, T* t_out);
// to Variants having TypeName type_name, for device string device,
// for UnaryVariantOp enum op.
#define REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(op, device, T, type_name, \
                                                 unary_op_function)        \
  REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION_UNIQ_HELPER(                    \
      __COUNTER__, op, device, T, type_name, unary_op_function)

#define REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION_UNIQ_HELPER(                  \
    ctr, op, device, T, type_name, unary_op_function)                          \
  REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION_UNIQ(ctr, op, device, T, type_name, \
                                                unary_op_function)

#define REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION_UNIQ(                         \
    ctr, op, device, T, type_name, unary_op_function)                          \
  static variant_op_registry_fn_registration::UnaryVariantUnaryOpRegistration< \
      T>                                                                       \
      register_unary_variant_op_decoder_fn_##ctr(op, device, type_name,        \
                                                 unary_op_function)

// Register a binary_op variant function with the signature:
//    Status BinaryOpFn(OpKernelContext* ctx, const T& a, const T& b, T* out);
// to Variants having TypeName type_name, for device string device,
// for BinaryVariantOp enum OP.
#define REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(op, device, T, type_name, \
                                                  binary_op_function)       \
  REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION_UNIQ_HELPER(                    \
      __COUNTER__, op, device, T, type_name, binary_op_function)

#define REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION_UNIQ_HELPER( \
    ctr, op, device, T, type_name, binary_op_function)         \
  REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION_UNIQ(              \
      ctr, op, device, T, type_name, binary_op_function)

#define REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION_UNIQ(                     \
    ctr, op, device, T, type_name, binary_op_function)                      \
  static variant_op_registry_fn_registration::                              \
      UnaryVariantBinaryOpRegistration<T>                                   \
          register_unary_variant_op_decoder_fn_##ctr(op, device, type_name, \
                                                     binary_op_function)

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_VARIANT_OP_REGISTRY_H_
