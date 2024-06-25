/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_WRAPPER_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_WRAPPER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"

namespace tflite {
namespace shim {
namespace op_wrapper {

using ::tflite::shim::OpKernelShim;
using ::tflite::shim::Runtime;

// Represents an attribute which can have many types. The first template
// parameter should be an AttrName, and the packed parameter should be the
// list of types of the attribute.
// TODO(b/265879737): When we begin compiling using C++20, the first template
// parameter should be changed to the template type.
template <typename N, typename... T>
struct Attr {
  const char* Name() const { return N::Name(); }
};

// Used to store the name of an attribute.
template <char const* str>
struct AttrName {
  static const char* Name() { return str; }
};

// Object for passing around types.
template <typename T>
struct AttrType {
  using type = T;
};

// The following constexprs are used to create the variant type which contains
// the combinations of our templated op. This variant is what is wrapped and
// ultimately called by the wrapper op.
//
// Example:
//   TmplOp with Attrs {"AType: {bool, float}", "BType: {int32, int64}"};
// Call:
//   const char a_type[]("AType"), b_type[]("BType");
//   VariantOp<Rt,
//             TmplOp,
//              Attr<AttrName<a_type>, bool, float>,
//              Attr<AttrName<b_type>, int32_t, int64_t>> x;
// Result:
//   absl::variant<TmplOp<Rt, bool, int32_t>, TmplOp<Rt, bool, int64_t>,
//                 TmplOp<Rt, float, int32_t>, TmplOp<Rt, float, int64_t>> x;

// Prepends a type onto a tuple.
template <typename T, typename... Us>
static constexpr std::tuple<T, Us...> prependTypeInner(T, std::tuple<Us...>);

// Prepend a type on each inner tuple group. This expression unwraps the inner
// tuples, and the inner expression performs the prepending.
template <typename T, typename... Us>
static constexpr auto prependType(T, std::tuple<Us...>)
    -> std::tuple<decltype(prependTypeInner(std::declval<T>(),
                                            std::declval<Us>()))...>;

// Base case for recursively processing all combinations of remaining
// attributes. The result is a tuple containing each type individually.
template <typename Name, typename... Ts>
static constexpr std::tuple<std::tuple<Ts>...> getCombinations(
    Attr<Name, Ts...>);

// Base case for recursively processing all types of an attribute.
template <typename Name, typename Head, typename... Attrs>
static constexpr auto getCombinations(Attr<Name, Head>, Attrs...)
    -> decltype(prependType(std::declval<Head>(),
                            getCombinations(std::declval<Attrs>()...)));

// Creates a tuple of tuples from a list of Attrute types by recursively
// popping the first type off the first attribute and prepending it to the
// combination of other attribute types. This result is then combined with the
// recursive processing of other types left.
template <typename Name, typename Head, typename... Tail, typename... Attrs>
static constexpr auto getCombinations(Attr<Name, Head, Tail...>, Attrs...)
    -> decltype(std::tuple_cat(
        prependType(std::declval<Head>(),
                    getCombinations(std::declval<Attrs>()...)),
        getCombinations(std::declval<Attr<Name, Tail...>>(),
                        std::declval<Attrs>()...)));

// Converts a tuple of types into the corresponding op type.
template <Runtime Rt, template <Runtime, typename...> typename Op,
          typename... Ts>
static constexpr Op<Rt, Ts...> convertTuplesToOpsInner(std::tuple<Ts...>);

// Convert a tuple of types into our op with those types. We first need to
// unwrap the inner tuples, we can then convert each individually in the
// inner expression and wrap them back up into a tuple.
template <Runtime Rt, template <Runtime, typename...> typename Op,
          typename... Ts>
static constexpr auto convertTuplesToOps(std::tuple<Ts...>) -> std::tuple<
    decltype(convertTuplesToOpsInner<Rt, Op>(std::declval<Ts>()))...>;

// Convert a tuple of types into a variant of types.
template <typename... Ts>
static constexpr std::variant<Ts...> convertTupleToVariant(std::tuple<Ts...>);

// The variant Op type created with TMP. A tuple of tuples containing the
// attribute combinations is first created. Then each inner tuple is converted
// into the op types, and finally the outer tuple is converted into a variant.
// Note, this uses a struct rather than a type alias because of a C++ limitation
// with template parameter packs not being deduced for aliases.
template <Runtime Rt, template <Runtime, typename...> typename Op,
          typename FirstAttr, typename... OtherAttrs>
struct VariantOp {
  using type =
      decltype(convertTupleToVariant(convertTuplesToOps<Rt, Op>(getCombinations(
          std::declval<FirstAttr>(), std::declval<OtherAttrs>()...))));
};

// Intermediate object used by the OpWrapper to properly extend OpKernelShim.
template <Runtime Rt>
class OpWrapperExtension : public OpKernelShim<OpWrapperExtension, Rt> {};

// Wraps a polymorphic op to be used by TF Lite. At this time, TF Lite does not
// support TypeConstraints like TensorFlow. This will wrap the op variants
// and delegate calls to the correctly typed variant when called.
//
// Example usage:
// Given a templated Op `TmplOp` with Attrs:
//     Attrs {"AType: {bool, float}", "BType: {int32_t, int64_t}"};
//
// We can define our type with the following (note that until C++20, these
// strings cannot be defined inline):
//
// const char a_type[]("AType"), b_type[]("BType");
// template <shim::Runtime Rt>
// using OpWrapperType = OpWrapper<Rt, TmplOp,
//     Attr<AttrName<a_type>, bool, float>,
//     Attr<AttrName<b_type>, int32_t, int64_t>>;
template <Runtime Rt, template <Runtime, typename...> typename Op,
          typename... As>
class OpWrapper : public OpWrapperExtension<Rt> {
 public:
  // This variant can be any permutation of the Op and its template params.
  using TmplOpType = typename VariantOp<Rt, Op, As...>::type;
  // For static calls, the exact type shouldn't matter, we just need a type.
  using TmplOpType0 = typename std::variant_alternative<0, TmplOpType>::type;

  using typename OpKernelShim<OpWrapperExtension, Rt>::InitContext;
  using typename OpKernelShim<OpWrapperExtension, Rt>::InvokeContext;
  using typename OpKernelShim<OpWrapperExtension, Rt>::ShapeInferenceContext;
  OpWrapper() = default;

  // For the static methods, they shouldn't change based on the types.
  static const char* OpName() { return TmplOpType0::OpName(); }
  static const char* Doc() { return TmplOpType0::Doc(); }

  static std::vector<std::string> Attrs() { return TmplOpType0::Attrs(); }
  static std::vector<std::string> Inputs() { return TmplOpType0::Inputs(); }
  static std::vector<std::string> Outputs() { return TmplOpType0::Outputs(); }

  static absl::Status ShapeInference(ShapeInferenceContext* context) {
    return TmplOpType0::ShapeInference(context);
  }

  // Creates the correctly typed wrapped object before delegating the Init call
  // to it. Invoke will also use this variant.
  absl::Status Init(InitContext* context) {
    SH_RETURN_IF_ERROR(SetVariantOp<As...>(context));

    return std::visit(
        [context](auto&& op) -> absl::Status { return op.Init(context); },
        *op_);
  }

  // Call Invoke on the created wrapped object.
  absl::Status Invoke(InvokeContext* context) {
    return std::visit(
        [context](auto&& op) -> absl::Status { return op.Invoke(context); },
        *op_);
  }

 private:
  // Sets op_ to the variant type matching the type attributes provided by the
  // InitContext. Similar to creating the variant type, we recursively
  // get all combinations of the attributes.
  template <typename FirstAttr, typename... Attrs>
  absl::Status SetVariantOp(InitContext* c) {
    return CombineAttributeTypes(this, c, FirstAttr{}, Attrs{}...);
  }

  // A simple object to hold Attrutes while we recursively find the
  // combinations. When called, it will unwrap the stored types to call the
  // underlying function.
  // The template parameters are:
  //   F: Object to wrap which will be another Forwarder object or the OpWrapper
  //   Name: AttrName of the attribute.
  //   T: Type of attribute for this combination.
  template <typename F, typename Name, typename T>
  struct Forwarder {
   public:
    explicit Forwarder(F* f) : inner(f) {}

    template <typename... Args>
    absl::Status SetOpCombination(Args... args) {
      return inner->SetOpCombination(Name::Name(), AttrType<T>{}, args...);
    }

   private:
    F* inner;
  };

  // Recursively processes for each combination of attribute types. First,
  // running over the first attibute and sub-combinations, then running over
  // the combinations of the remaining types of the first attribute.
  template <typename F, typename Name, typename Head, typename... Tail,
            typename... Attrs>
  absl::Status CombineAttributeTypes(F* obj, InitContext* c,
                                     Attr<Name, Head, Tail...>, Attrs... rest) {
    SH_RETURN_IF_ERROR(
        ApplyAttrType(obj, c, Name{}, AttrType<Head>{}, rest...));

    return CombineAttributeTypes(obj, c, Attr<Name, Tail...>{}, rest...);
  }

  // Base case for recursively processing types of an attribute.
  template <typename F, typename Name, typename... Attrs>
  absl::Status CombineAttributeTypes(F*, InitContext*, Attr<Name>, Attrs...) {
    return absl::OkStatus();
  }

  // Saves the names and types of each attribute in the current combination
  // in a Forwarder object which will ultimately call a typed function.
  template <typename F, typename Name, typename T, typename Attr,
            typename... Attrs>
  absl::Status ApplyAttrType(F* obj, InitContext* c, Name, AttrType<T>, Attr a,
                             Attrs... rest) {
    Forwarder<F, Name, T> forwarder(obj);

    return CombineAttributeTypes(&forwarder, c, a, rest...);
  }

  // Base case for recursively finding combinations of attributes.
  template <typename F, typename Name, typename T>
  absl::Status ApplyAttrType(F* obj, InitContext* c, Name, AttrType<T> t) {
    return obj->SetOpCombination(Name::Name(), t, c);
  }

  // Checks the attribute types from the context for this particular attribute
  // type combination. If correct, we set the op variant to this op combo.
  //
  // For this, we actually need to overload the functiona nd create a template
  // for each number of attributes.
  template <typename T>
  absl::Status SetOpCombination(std::string Name1, AttrType<T>,
                                InitContext* context) {
    int64_t datatype_1;
    SH_RETURN_IF_ERROR(context->GetAttr(Name1, &datatype_1));
    if (datatype_1 == typeToTfLiteType<T>()) {
      this->op_ = std::make_unique<TmplOpType>(Op<Rt, T>());
    }
    return absl::OkStatus();
  }

  template <typename T, typename U>
  absl::Status SetOpCombination(std::string Name1, AttrType<T>,
                                std::string Name2, AttrType<U>,
                                InitContext* context) {
    int64_t datatype_1, datatype_2;
    SH_RETURN_IF_ERROR(context->GetAttr(Name1, &datatype_1));
    SH_RETURN_IF_ERROR(context->GetAttr(Name2, &datatype_2));
    if (datatype_1 == typeToTfLiteType<T>() &&
        datatype_2 == typeToTfLiteType<U>()) {
      this->op_ = std::make_unique<TmplOpType>(Op<Rt, T, U>());
    }
    return absl::OkStatus();
  }

  template <typename T, typename U, typename V>
  absl::Status SetOpCombination(std::string Name1, AttrType<T>,
                                std::string Name2, AttrType<U>,
                                std::string Name3, AttrType<V>,
                                InitContext* context) {
    int64_t datatype_1, datatype_2, datatype_3;
    SH_RETURN_IF_ERROR(context->GetAttr(Name1, &datatype_1));
    SH_RETURN_IF_ERROR(context->GetAttr(Name2, &datatype_2));
    SH_RETURN_IF_ERROR(context->GetAttr(Name3, &datatype_3));
    if (datatype_1 == typeToTfLiteType<T>() &&
        datatype_2 == typeToTfLiteType<U>() &&
        datatype_3 == typeToTfLiteType<V>()) {
      this->op_ = std::make_unique<TmplOpType>(Op<Rt, T, U, V>());
    }
    return absl::OkStatus();
  }

  template <typename T, typename U, typename V, typename W>
  absl::Status SetOpCombination(std::string Name1, AttrType<T>,
                                std::string Name2, AttrType<U>,
                                std::string Name3, AttrType<V>,
                                std::string Name4, AttrType<W>,
                                InitContext* context) {
    int64_t datatype_1, datatype_2, datatype_3, datatype_4;
    SH_RETURN_IF_ERROR(context->GetAttr(Name1, &datatype_1));
    SH_RETURN_IF_ERROR(context->GetAttr(Name2, &datatype_2));
    SH_RETURN_IF_ERROR(context->GetAttr(Name3, &datatype_3));
    SH_RETURN_IF_ERROR(context->GetAttr(Name4, &datatype_4));
    if (datatype_1 == typeToTfLiteType<T>() &&
        datatype_2 == typeToTfLiteType<U>() &&
        datatype_3 == typeToTfLiteType<V>() &&
        datatype_4 == typeToTfLiteType<W>()) {
      this->op_ = std::make_unique<TmplOpType>(Op<Rt, T, U, V, W>());
    }
    return absl::OkStatus();
  }

  template <typename T, typename U, typename V, typename W, typename X>
  absl::Status SetOpCombination(std::string Name1, AttrType<T>,
                                std::string Name2, AttrType<U>,
                                std::string Name3, AttrType<V>,
                                std::string Name4, AttrType<W>,
                                std::string Name5, AttrType<X>,
                                InitContext* context) {
    int64_t datatype_1, datatype_2, datatype_3, datatype_4, datatype_5;
    SH_RETURN_IF_ERROR(context->GetAttr(Name1, &datatype_1));
    SH_RETURN_IF_ERROR(context->GetAttr(Name2, &datatype_2));
    SH_RETURN_IF_ERROR(context->GetAttr(Name3, &datatype_3));
    SH_RETURN_IF_ERROR(context->GetAttr(Name4, &datatype_4));
    SH_RETURN_IF_ERROR(context->GetAttr(Name5, &datatype_5));
    if (datatype_1 == typeToTfLiteType<T>() &&
        datatype_2 == typeToTfLiteType<U>() &&
        datatype_3 == typeToTfLiteType<V>() &&
        datatype_4 == typeToTfLiteType<W>() &&
        datatype_5 == typeToTfLiteType<X>()) {
      this->op_ = std::make_unique<TmplOpType>(Op<Rt, T, U, V, W, X>());
    }
    return absl::OkStatus();
  }

 protected:
  // The wrapped object variant.
  std::unique_ptr<TmplOpType> op_;
};

}  // namespace op_wrapper
}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_WRAPPER_H_
