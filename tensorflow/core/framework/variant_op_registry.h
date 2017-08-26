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
#include <vector>

#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"

namespace tensorflow {

// A global UnaryVariantOpRegistry is used to hold callback functions
// for different variant types.  To be used by ShapeOp, RankOp, and
// SizeOp, decoding, etc.

class UnaryVariantOpRegistry {
 public:
  typedef std::function<Status(const Variant& v, TensorShape*)> VariantShapeFn;
  typedef std::function<bool(Variant*)> VariantDecodeFn;

  // Add a shape lookup function to the registry.
  void RegisterShapeFn(const string& type_name, const VariantShapeFn& shape_fn);

  // Returns nullptr if no shape function was found for the given TypeName.
  VariantShapeFn* GetShapeFn(const string& type_name);

  // Add a decode function to the registry.
  void RegisterDecodeFn(const string& type_name,
                        const VariantDecodeFn& decode_fn);

  // Returns nullptr if no decode function was found for the given TypeName.
  VariantDecodeFn* GetDecodeFn(const string& type_name);

  static UnaryVariantOpRegistry* Global();

 private:
  std::unordered_map<string, VariantShapeFn> shape_fns;
  std::unordered_map<string, VariantDecodeFn> decode_fns;
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

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_VARIANT_OP_REGISTRY_H_
