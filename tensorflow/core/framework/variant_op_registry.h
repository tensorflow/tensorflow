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

enum VariantDeviceCopyDirection {
  INVALID_DEVICE_COPY_DIRECTION = 0,
  HOST_TO_DEVICE = 1,
  DEVICE_TO_HOST = 2,
  DEVICE_TO_DEVICE = 3,
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

  // An AsyncTensorDeviceCopyFn is a function provided to
  // the user-provided DeviceCopyFn callback as the third argument ("copier").
  //
  // Expected inputs:
  //   from: A Tensor on the host (if performing cpu->gpu copy), or
  //         device (if performing gpu->cpu or gpu->gpu copy).
  //   to: An empty/uninitialized tensor.  It will be updated upon
  //       successful return of the function with the correct dtype and shape.
  //       However, the copied data will not be available until the compute
  //       stream has been synchronized.
  //
  // Returns:
  //   The status upon memory allocation / initialization of the
  //   "to" tensor, and enqueue of the copy onto the compute stream.
  //   Any failure of the copy itself will update the underlying
  //   stream status and propagate through the runtime independent
  //   of the caller.
  typedef std::function<Status(const Tensor& from, Tensor* to)>
      AsyncTensorDeviceCopyFn;

  // The AsyncVariantDeviceCopyFn is the signature of the 'device_copy_fn'
  // expected to be passed to the registration macro
  // INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION.
  typedef std::function<Status(const Variant& from, Variant* to,
                               AsyncTensorDeviceCopyFn copy_fn)>
      AsyncVariantDeviceCopyFn;

  // Add a shape lookup function to the registry.
  void RegisterShapeFn(const string& type_name, const VariantShapeFn& shape_fn);

  // Returns nullptr if no shape function was found for the given TypeName.
  VariantShapeFn* GetShapeFn(StringPiece type_name);

  // Add a decode function to the registry.
  void RegisterDecodeFn(const string& type_name,
                        const VariantDecodeFn& decode_fn);

  // Returns nullptr if no decode function was found for the given TypeName.
  VariantDecodeFn* GetDecodeFn(StringPiece type_name);

  // Add a copy-to-GPU function to the registry.
  void RegisterDeviceCopyFn(const VariantDeviceCopyDirection direction,
                            const string& type_name,
                            const AsyncVariantDeviceCopyFn& device_copy_fn);

  // Returns nullptr if no copy function was found for the given
  // TypeName and direction.
  AsyncVariantDeviceCopyFn* GetDeviceCopyFn(
      const VariantDeviceCopyDirection direction, StringPiece type_name);

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
  std::unordered_map<StringPiece, VariantShapeFn, StringPieceHasher> shape_fns;
  std::unordered_map<StringPiece, VariantDecodeFn, StringPieceHasher>
      decode_fns;

  // Map std::pair<Direction, type_name> to function.
  struct PairHash {
    template <typename Direction>
    std::size_t operator()(const std::pair<Direction, StringPiece>& x) const {
      // The hash of an enum is just its value as a std::size_t.
      std::size_t ret = static_cast<std::size_t>(std::get<0>(x));
      ret = Hash64Combine(ret, sp_hasher_(std::get<1>(x)));
      return ret;
    }
    StringPieceHasher sp_hasher_;
  };

  std::unordered_map<std::pair<VariantDeviceCopyDirection, StringPiece>,
                     AsyncVariantDeviceCopyFn, PairHash>
      device_copy_fns;

  // Map std::tuple<Op, device, type_name> to function.
  struct TupleHash {
    template <typename Op>
    std::size_t operator()(
        const std::tuple<Op, StringPiece, StringPiece>& x) const {
      // The hash of an enum is just its value as a std::size_t.
      std::size_t ret = static_cast<std::size_t>(std::get<0>(x));
      ret = Hash64Combine(ret, sp_hasher_(std::get<1>(x)));
      ret = Hash64Combine(ret, sp_hasher_(std::get<2>(x)));
      return ret;
    }
    StringPieceHasher sp_hasher_;
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

// Copies a variant between CPU<->GPU, or between GPU<->GPU.
// The variant 'from' must have a registered DeviceCopyFn for the
// given direction.  The returned variant 'to' will have
// (some subset of its) tensors stored on destination according to the
// registered DeviceCopyFn function for the given direction.  Returns
// an Internal error if the Variant does not have a registered
// DeviceCopyFn function for the given direction, or if initiating the
// copy fails.
//
// REQUIRES:
//   'to' is not null.
//
Status VariantDeviceCopy(
    const VariantDeviceCopyDirection direction, const Variant& from,
    Variant* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy_fn);

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
    UnaryVariantOpRegistry::Global()->RegisterShapeFn(
        type_name,
        [type_name, shape_fn](const Variant& v, TensorShape* s) -> Status {
          const T* t = v.get<T>();
          if (t == nullptr) {
            return errors::Internal(
                "VariantShapeFn: Could not access object, type_name: ",
                type_name);
          }
          return shape_fn(*t, s);
        });
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
    UnaryVariantOpRegistry::Global()->RegisterDecodeFn(
        type_name, [type_name](Variant* v) -> bool {
          DCHECK_NE(v, nullptr);
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
        });
  }
};

template <typename T>
class UnaryVariantDeviceCopyRegistration {
 public:
  typedef std::function<Status(const T& t, T* t_out,
                               UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn)>
      LocalVariantDeviceCopyFn;
  UnaryVariantDeviceCopyRegistration(
      const VariantDeviceCopyDirection direction, const string& type_name,
      const LocalVariantDeviceCopyFn& device_copy_fn) {
    UnaryVariantOpRegistry::Global()->RegisterDeviceCopyFn(
        direction, type_name,
        [type_name, device_copy_fn](
            const Variant& from, Variant* to,
            UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn
                device_copy_tensor_fn) -> Status {
          DCHECK_NE(to, nullptr);
          *to = T();
          if (from.get<T>() == nullptr) {
            return errors::Internal(
                "VariantCopyToGPUFn: Could not access object, type_name: ",
                type_name);
          }
          const T& t = *from.get<T>();
          T* t_out = to->get<T>();
          return device_copy_fn(t, t_out, device_copy_tensor_fn);
        });
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
    UnaryVariantOpRegistry::Global()->RegisterUnaryOpFn(
        op, device, type_name,
        [type_name, unary_op_fn](OpKernelContext* ctx, const Variant& v,
                                 Variant* v_out) -> Status {
          DCHECK_NE(v_out, nullptr);
          *v_out = T();
          if (v.get<T>() == nullptr) {
            return errors::Internal(
                "VariantUnaryOpFn: Could not access object, type_name: ",
                type_name);
          }
          const T& t = *v.get<T>();
          T* t_out = v_out->get<T>();
          return unary_op_fn(ctx, t, t_out);
        });
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
    UnaryVariantOpRegistry::Global()->RegisterBinaryOpFn(
        op, device, type_name,
        [type_name, binary_op_fn](OpKernelContext* ctx, const Variant& a,
                                  const Variant& b, Variant* out) -> Status {
          DCHECK_NE(out, nullptr);
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
        });
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

// ****** NOTE ******
// FOR INTERNAL USE ONLY.  IF YOU USE THIS WE MAY BREAK YOUR CODE.
// ****** NOTE ******
//
// Register a device copy variant function for the given copy
// direction and type; where direction is the enum
// VariantDeviceCopyDirection, and the device_copy_fn has signature:
//
//   Status device_copy_fn(
//     const T& t, T* t_out,
//     const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copier);
//
// And device_copy_fn calls copier 0 or more times.  For details on
// the behavior of the copier function, see the comments at the
// declaration of UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn.
//
// Note, the device_copy_fn may choose to keep some tensors
// on host, e.g. by assigning to->tensor = from.tensor (assuming
// from.tensor is already on host); or by setting
//   to->tensor = Tensor(cpu_allocator(), ...)
// and manually updating its values.
//
// If this is the case, the CopyFns for HOST_TO_DEVICE,
// DEVICE_TO_HOST, and DEVICE_TO_DEVICE must perform host-to-host
// copies in a consistent manner.  For example, one must always
// manually copy any "always on host" tensors in all directions instead of e.g.
//   - performing a host-to-host copy in one direction,
//   - using the provided copier function in the reverse direction.
// Doing the latter will cause program failures.
//
// ****** NOTE ******
// FOR INTERNAL USE ONLY.  IF YOU USE THIS WE MAY BREAK YOUR CODE.
// ****** NOTE ******
#define INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(       \
    T, direction, type_name, device_copy_fn)                        \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION_UNIQ_HELPER( \
      __COUNTER__, T, direction, type_name, device_copy_fn)

#define INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION_UNIQ_HELPER( \
    ctr, T, direction, type_name, device_copy_fn)                         \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION_UNIQ(              \
      ctr, T, direction, type_name, device_copy_fn)

#define INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION_UNIQ(             \
    ctr, T, direction, type_name, device_copy_fn)                              \
  static variant_op_registry_fn_registration::                                 \
      UnaryVariantDeviceCopyRegistration<T>                                    \
          register_unary_variant_op_device_copy_fn_##ctr(direction, type_name, \
                                                         device_copy_fn)

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
