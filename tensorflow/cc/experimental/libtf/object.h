/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
/// @file object.h
/// @brief Object hierarchy for the TensorFlow C++ API. All "objects" are
/// derived from the `Handle` class. Instances of `Handle` are referred to as
/// "handles". All handles have a tagged value.
///
/// Example Usage:
/// Object runtime = GetRuntime("tfrt");
/// Object module = runtime.Get("Import")("cool_mobilenet")
/// runtime.Get("Tensor")(Tuple(5,5,5), 3.3);
/// Object test = CreateModule("test");
/// test.Set("cool_function", callable);
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_OBJECT_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_OBJECT_H_

#include <string>
#include <utility>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tf {
namespace libtf {

using TaggedValue = impl::TaggedValue;
class Handle;

// Necessary forward declare.
template <class T>
Handle Convert(T value);

/// @brief Base Handle class that wraps TaggedValue data. All data creation and
/// manipulation should done using Handle instances. Users should not be working
/// with TaggedValues directly.

/// The `Handle` class contains a TaggedValue in the `value_` member, which
/// contains the underlying data. An object belonging to `Foo`, a derived class
/// of `Handle`, can be referred to as a `Foo` handle.
///
/// It is important that all derived classes do not add any new data fields.
/// This ensures that it is always safe to slice down (i.e. assign an object of
/// a derived class to the base class) a handle to the base Handle class.
class Handle {
 public:
  /// Default constructor, which initializes a TaggedValue with type NONE.
  Handle() : value_(TaggedValue::None()) {}

 public:
  /// Constructs a handle from a TaggedValue.
  explicit Handle(TaggedValue value) : value_(std::move(value)) {}
  // explicit Handle(TaggedValue value, Handle* class_input)
  //     : value_(std::move(value)), class_(class_input) {}
  // const Handle& type() { return *class_; }

 protected:
  /// The wrapped TaggedValue.
  TaggedValue value_;
  // effectively a "weak reference" to intern'd class value.
  // types are compared by comparing pointer values here.
  // Handle* class_;  // effectively a "weak reference" to intern'd class value.

  /// The Integer handle.
  friend class Integer;
  /// The Float handle.
  friend class Float;
  /// The String handle.
  friend class String;
  /// The Object handle.
  friend class Object;
  /// The List handle.
  friend class List;
  /// The Dictionary handle.
  friend class Dictionary;
  /// The Tuple handle.
  friend class Tuple;
  /// The Callable handle.
  friend class Callable;
  /// The Tensor handle.
  friend class Tensor;
  /// Converts a Handle instance to an instance of a derived class `T`.
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
  /// Infrastructure for converting a TaggedValue tuple function signature to an
  /// unpacked variable list.
  template <typename Fn, class TRET, class... ArgsOut>
  friend class UneraseCallHelper;
};

// Forward declare.
template <class T>
tensorflow::StatusOr<T> Cast(Handle handle);

/// @brief The None class for holding TaggedValues of type NONE.
class None final : public Handle {
 public:
  /// Creates a handle that wraps a NONE TaggedValue.
  None() : Handle(TaggedValue::None()) {}

 private:
  explicit None(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The String class for holding TaggedValues of type STRING.
class String final : public Handle {
 public:
  /// Creates a handle that wraps a STRING TaggedValue.
  explicit String(const char* s) : Handle(TaggedValue(s)) {}
  /// Returns the underlying TaggedValue string.
  const char* get() const { return value_.s(); }

 private:
  // Private since it is in general unsafe.
  explicit String(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The `Object` class modeled after Python "objects".
///
/// An `Object` uses a TaggedValue dictionary to store its attributes. The
/// "__parent__" attribute is reserved.
class Object : public Handle {
 public:
  /// Constructs a handle that acts as an object.
  Object() : Handle(TaggedValue::Dict()) {}
  /// Retrieves the key of the object's parent.
  static const String& ParentKey();

  /// @brief Gets an object member attribute`key`.
  ///
  /// If the `key` is not found in the object, the object's "__parent__"
  /// attribute is then searched.
  ///
  /// @tparam T The desired return type.
  /// @param key The key to look up.
  /// @return `StatusOr` wrapping the key's value.
  template <class T = Handle>
  tensorflow::StatusOr<T> Get(const String& key) {
    auto& dict = value_.dict();
    auto it = dict.find(key.value_);
    if (it != dict.end()) {
      return Cast<T>(Handle(it->second));
    } else {
      // Lookup in object stored by reference in attribute  "__parent__".
      auto it_class = dict.find(ParentKey().value_);
      if (it_class != dict.end()) {
        auto& class_dict_maybe = it_class->second;
        if (class_dict_maybe.type() == TaggedValue::DICT) {
          auto& dict = class_dict_maybe.dict();
          auto it = dict.find(key.value_);
          if (it != value_.dict().end()) {
            return Cast<T>(Handle(it->second));
          }
        }
      }
    }
    return tensorflow::errors::NotFound("Key not in dictionary.");
  }

  /// Sets `key` attribute with the underlying value of `h`.
  void Set(const String& key, Handle h) {
    value_.dict()[key.value_] = std::move(h.value_);
  }

  /// Removes `key` from the object's attributes.
  void Unset(const String& key) { value_.dict().erase(key.value_); }
  // TODO(b/): Adding dir() is in the future.
 private:
  // Private since it is in general unsafe.
  explicit Object(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The Dictionary class for holding TaggedValues of type DICT.
class Dictionary final : public Handle {
 public:
  /// Constructs a handle that wraps a DICT TaggedValue.
  Dictionary() : Handle(TaggedValue::Dict()) {}
  // TODO(aselle): make this private to preserve invariant.

  /// Retrieves `key` with type `T`.
  template <class T>
  tensorflow::StatusOr<T> Get(const Handle& key) {
    auto it = value_.dict().find(key.value_);
    if (it != value_.dict().end()) return Cast<T>(Handle(it->second));
    return tensorflow::errors::NotFound("Key not in dictionary.");
  }
  /// Sets `key` with value `value`.
  void Set(const String& key, Handle value) {
    value_.dict()[key.value_] = std::move(value.value_);
  }
  /// Sets `key` with value `value`.
  void Set(const Handle& key, Handle value) {
    value_.dict()[key.value_] = std::move(value.value_);
  }
  /// Retrieves size of dictionary.
  size_t size() const { return value_.dict().size(); }

 private:
  // Private since it is in general unsafe.
  explicit Dictionary(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The Integer class for holding TaggedValues of type INT.
class Integer final : public Handle {
 public:
  /// Creates a handle that wraps an INT TaggedValue.
  explicit Integer(Handle h) : Handle(h.value_) {}
  /// Creates a handle that wraps an INT TaggedValue.
  explicit Integer(int64_t i) : Handle(TaggedValue(i)) {}
  /// Retrieves the underlying integer value.
  int64_t get() const { return value_.i64().get(); }

 private:
  // Private since it is in general unsafe.
  explicit Integer(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The Float class for holding TaggedValues of type FLOAT.
class Float final : public Handle {
 public:
  /// Constructs a Float handle that wraps a FLOAT TaggedValue.
  explicit Float(Handle h) : Handle(h.value_) {}
  /// Constructs a Float handle that wraps a FLOAT TaggedValue.
  explicit Float(float i) : Handle(TaggedValue(i)) {}
  /// Retrieves the underlying float value.
  float get() const { return value_.f32().get(); }

 private:
  // Private since it is in general unsafe.
  explicit Float(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The Tensor class for holding TaggedValues of type TENSOR.
class Tensor final : public Handle {
 public:
  /// Constructs a Tensor handle from a Handle that wraps a TENSOR TaggedValue.
  explicit Tensor(Handle h) : Handle(h.value_) {}

  /// @brief Retrieves the value of the Tensor handle.

  /// @param data Buffer in which to copy contents of the handle.
  /// @throws InvalidArgument Raises error if `data` is of invalid size.
  template <class T>
  tensorflow::Status GetValue(absl::Span<T> data) const;

 private:
  // Private since it is in general unsafe.
  explicit Tensor(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

template <class T>
tensorflow::Status Tensor::GetValue(absl::Span<T> data) const {
  tensorflow::AbstractTensorPtr t;
  {
    const auto abstract_t = value_.tensor().get();
    if (!tensorflow::ImmediateExecutionTensorHandle::classof(abstract_t)) {
      return tensorflow::errors::InvalidArgument(
          "Attempting to get value of non eager tensor.");
    }
    auto imm_t =
        static_cast<tensorflow::ImmediateExecutionTensorHandle*>(abstract_t);
    tensorflow::Status status;
    t.reset(imm_t->Resolve(&status));
    if (!status.ok()) {
      return status;
    }
  }
  if (data.size() != t->NumElements()) {
    return tensorflow::errors::InvalidArgument(absl::StrCat(
        "Mismatched number of elements: \n", "Expected: ", data.size(), "\n",
        "Actual: ", t->NumElements(), "\n"));
  }
  memcpy(data.data(), t->Data(), t->ByteSize());
  return ::tensorflow::OkStatus();
}

/// @brief The Tuple class for holding TaggedValues of type TUPLE.
class Tuple : public Handle {
 public:
  /// Constructs a Tuple handle.
  template <class... T>
  explicit Tuple(T... args) : Handle(TaggedValue::Tuple()) {
    add(args...);
  }

  /// Retrieves value at index `i`.
  template <class T>
  tensorflow::StatusOr<T> Get(size_t i) {
    if (i >= value_.tuple().size())
      return tensorflow::errors::InvalidArgument("Out of bounds index.");
    return Cast<T>(Handle(value_.tuple()[i]));
  }

  /// Retrieves number of elements.
  size_t size() const { return value_.tuple().size(); }

 private:
  // Add an item to a tuple. Should only be done by special construction
  // like Callables (which are a friend).
  void add() {}
  template <class T, class... T2>
  void add(T arg, T2... args) {
    value_.tuple().emplace_back(Convert(arg).value_);
    add(args...);
  }

  // Private since it is in general unsafe.
  explicit Tuple(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The List class for holding TaggedValues of type LIST.
class List final : public Handle {
 public:
  /// Constructs a List handle.
  template <class... T>
  explicit List(T... args) : Handle(TaggedValue::List()) {}
  /// Retrieves value at index `i`.
  template <class T>
  tensorflow::StatusOr<T> Get(size_t i) {
    if (i >= size()) {
      return tensorflow::errors::InvalidArgument("Out of bounds index.");
    }
    return Cast<T>(Handle(value_.list()[i]));
  }

  /// Sets value `h` at index `i`.
  tensorflow::Status Set(size_t i, Handle h) {
    if (i >= size()) {
      return tensorflow::errors::InvalidArgument("Out of bounds index.");
    }
    value_.list()[i] = std::move(h.value_);
    return ::tensorflow::OkStatus();
  }

  /// Appends `arg` to list.
  template <class T>
  void append(T arg) {
    value_.list().emplace_back(Convert(arg).value_);
  }
  /// Retrieves size of list.
  size_t size() const { return value_.list().size(); }

 private:
  // Private since it is in general unsafe.
  explicit List(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The `KeywordArg` class for storing keyword arguments as name value
/// pairs.
class KeywordArg {
 public:
  explicit KeywordArg(const char* s) : key_(String(s)), value_() {}

  template <class T>
  KeywordArg& operator=(const T obj) {
    value_ = Convert(obj);
    return *this;
  }

  friend class Callable;

 private:
  String key_;
  Handle value_;
};

/// @brief The Callable class for creating callables.
class Callable final : public Handle {
 private:
  // Collect arguments for call
  void CollectArgs(Tuple& args, Dictionary& kwargs, int idx) {}
  template <typename T, typename... Types>
  void CollectArgs(Tuple& args, Dictionary& kwargs, int idx, T v,
                   Types... vars) {
    const Handle& o = Convert(v);
    args.value_.tuple().emplace_back(o.value_);
    CollectArgs(args, kwargs, idx + 1, vars...);
  }
  template <typename... Types>
  void CollectArgs(Tuple& args, Dictionary& kwargs, int idx, KeywordArg v,
                   Types... vars) {
    kwargs.Set(v.key_, v.value_);
    CollectArgs(args, kwargs, idx + 1, vars...);
  }

 public:
  /// @brief Calls the wrapped TaggedValue function on a variable argument
  /// list.
  template <typename TReturn = Handle, typename... Types>
  tensorflow::StatusOr<TReturn> Call(Types... vars) {
    Dictionary kwargs = Dictionary();
    Tuple args;
    CollectArgs(args, kwargs, 0, vars...);
    auto maybe_value =
        value_.func()(std::move(args.value_), std::move(kwargs.value_));
    if (!maybe_value.ok()) {
      return maybe_value.status();
    }
    return Cast<TReturn>(Handle(maybe_value.value()));
  }

 public:
  // TODO(aselle): need to find a way to write test w/o this being public.
  // Private since it is in general unsafe.
  explicit Callable(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

namespace internal {
/// @brief The Capsule class for holding pointers.
class Capsule final : public Handle {
 public:
  /// Statically cast the TaggedValue capsule to type `T`.
  template <class T>
  T cast() {
    return static_cast<T>(value_.capsule());
  }

 private:
  // Private since it is in general unsafe.
  explicit Capsule(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> tf::libtf::Cast(Handle handle);
};
}  // namespace internal

/// @defgroup Util Functions for type conversion
///
/// @brief Functions for retrieving and converting Handle types.
/// @{

/// Retrieves tagged type of `T` handle.
template <class T>
inline TaggedValue::Type TypeToTaggedType() {}
/// Retrieves tagged type of base class handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Handle>() {
  return TaggedValue::Type::NONE;
}
/// Retrieves tagged type of None handle.
template <>
inline TaggedValue::Type TypeToTaggedType<None>() {
  return TaggedValue::Type::NONE;
}
/// Retrieves tagged type of String handle.
template <>
inline TaggedValue::Type TypeToTaggedType<String>() {
  return TaggedValue::Type::STRING;
}
/// Retrieves tagged type of Callable handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Callable>() {
  return TaggedValue::Type::FUNC;
}
/// Retrieves tagged type of Integer handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Integer>() {
  return TaggedValue::Type::INT64;
}
/// Retrieves tagged type of Float handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Float>() {
  return TaggedValue::Type::FLOAT32;
}
/// Retrieves tagged type of Object handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Object>() {
  return TaggedValue::Type::DICT;
}
/// Retrieves tagged type of Dictionary handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Dictionary>() {
  return TaggedValue::Type::DICT;
}
/// Retrieves tagged type of List handle.
template <>
inline TaggedValue::Type TypeToTaggedType<List>() {
  return TaggedValue::Type::LIST;
}
/// Retrieves tagged type of Tensor handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Tensor>() {
  return TaggedValue::Type::TENSOR;
}
/// Retrieves tagged type of Capsule handle.
template <>
inline TaggedValue::Type TypeToTaggedType<internal::Capsule>() {
  return TaggedValue::Type::CAPSULE;
}
// TODO(unknown): fully populate

/// @brief Casts a handle to type `T`
///
/// @param handle The handle to cast.
/// @tparam T The target handle type.
/// @exception InvalidArgument Raises error if the underlying TaggedValue type
/// of `handle` is not equivalent to `T`.
template <class T>
tensorflow::StatusOr<T> Cast(Handle handle) {
  if (handle.value_.type() == TypeToTaggedType<T>() ||
      std::is_same<T, Handle>::value)
    return T((std::move(handle.value_)));
  return tensorflow::errors::InvalidArgument("Incompatible cast.");
}

// Converters for C++ primitives like float and int to handles. Allows callable
// calls and list appends to be more idiomatic.

/// Converts a C++ const char* to a String handle.
template <>
inline Handle Convert(const char* value) {
  return String(value);
}
/// Converts a C++ int32_t to an Integer handle.
template <>
inline Handle Convert(int32_t value) {
  return Integer(value);
}
/// Converts a C++ int64_t to an Integer handle.
template <>
inline Handle Convert(int64_t value) {
  return Integer(value);
}
/// Converts a C++ float to an Integer handle.
template <>
inline Handle Convert(float value) {
  return Float(value);
}
/// Converts a value with primitive type T to a Handle.
template <class T>
inline Handle Convert(T value) {
  return Handle(std::move(value));
}

/// @}

// in the future it will be possible to make additional hard typed APIs
// by generating code by introspecting objects.

// Here's a code gen'd example
// The dynamic structure can be turned into it.
/*
class Tf : Object {
  Tensor ones(Tensor shape, String dtype);
  // ...
}
*/

// Adapter to allow users to define Callables. Use TFLIB_CALLABLE_ADAPTOR
// instead.
template <typename TF, typename TReturn, typename... TFuncArgs>
class CallableWrapper;

// Template extracts arguments from a lambda function. This base
// class definition inherits from a another specialization in order. We use
// this top level template to extract the function pointer associated with
// the created lambda functor class.
template <typename TLambda>
class CallableWrapperUnpackArgs
    : public CallableWrapperUnpackArgs<decltype(&TLambda::operator())> {
 public:
  CallableWrapperUnpackArgs(TLambda fn, const char* name)
      : CallableWrapperUnpackArgs<decltype(&TLambda::operator())>(fn, name) {}
};

// This specialization unpacks the arguments from a normal function pointer.
template <typename TReturn, typename... TFuncArgs>
class CallableWrapperUnpackArgs<TReturn (*)(TFuncArgs...)>
    : public CallableWrapper<TReturn (*)(TFuncArgs...), TReturn, TFuncArgs...> {
  using Fn = TReturn (*)(TFuncArgs...);

 public:
  CallableWrapperUnpackArgs(Fn fn, const char* name)
      : CallableWrapper<Fn, TReturn, TFuncArgs...>(fn, name) {}
};

// This is the second stage of extracting the arguments from lambda function.
// NOTE: CallableWrapper's first template argument is the type of the
// function or functor (not the member pointer).
template <typename TClass, typename TReturn, typename... TFuncArgs>
class CallableWrapperUnpackArgs<TReturn (TClass::*)(TFuncArgs...) const>
    : public CallableWrapper<TClass, TReturn, TFuncArgs...> {
  using Fn = TClass;

 public:
  CallableWrapperUnpackArgs(Fn fn, const char* name)
      : CallableWrapper<Fn, TReturn, TFuncArgs...>(fn, name) {}
};

template <class Fn, typename TReturn, class... ArgsOut>
class UneraseCallHelper;

// UneraseCallHelper::Call allows transforming all the incoming arguments
// from a TaggedValue tuple to a variadic list of args.  The class template
// starts as a list of argument types and ends empty. The static member
// template starts empty and ends with the unerased types of the signature.

// Base case (all arguments are processed, so call the function TFunc.
template <class Fn, typename TReturn>
class UneraseCallHelper<Fn, TReturn> {
 public:
  template <typename... ArgsOut>
  static tensorflow::StatusOr<TaggedValue> Call(const char* name, Fn functor_,
                                                int argument_index,
                                                const TaggedValue& args_in,
                                                ArgsOut... args) {
    // Call concrete type function
    TReturn ret = functor_(args...);
    return ret.value_;
  }
};

// Unpack a single argument case. Each argument is then cast.
template <class Fn, typename TReturn, class TSignatureArg,
          class... TSignatureRest>
class UneraseCallHelper<Fn, TReturn, TSignatureArg, TSignatureRest...> {
 public:
  template <typename... TArgsOut>
  static tensorflow::StatusOr<TaggedValue> Call(const char* name, Fn fn,
                                                int argument_index,
                                                TaggedValue& args_in,
                                                TArgsOut... args) {
    Handle h(std::move(args_in.tuple()[argument_index]));
    tensorflow::StatusOr<TSignatureArg> x = Cast<TSignatureArg>(std::move(h));
    if (!x.ok())
      return tensorflow::errors::InvalidArgument(
          std::string("Function ") + name + " Arg " +
          std::to_string(argument_index) +
          " cannot be cast to desired signature type ");
    return UneraseCallHelper<Fn, TReturn, TSignatureRest...>::template Call(
        name, fn, argument_index + 1, args_in, args..., *x);
  }
};

// Template specialization that allows extracting arguments from a C function
// pointer.
template <class Fn, typename TReturn, typename... TFuncArgs>
class CallableWrapper {
 private:
  Fn functor_;
  const char* name_;

 public:
  explicit CallableWrapper(Fn fn, const char* name)
      : functor_(fn), name_(name) {}

  // Entry point of the Adaptor functor. Note args, and kwargs are attempted
  // to be moved.
  tensorflow::StatusOr<TaggedValue> operator()(TaggedValue args,
                                               TaggedValue kwargs) {
    constexpr size_t argument_count = sizeof...(TFuncArgs);
    if (argument_count != args.tuple().size())
      return tensorflow::errors::InvalidArgument(
          std::string("Function ") + name_ + " expected " +
          std::to_string(argument_count) + " args.");
    return UneraseCallHelper<Fn, TReturn, TFuncArgs...>::Call(name_, functor_,
                                                              0, args);
  }
};

// Wrap a function that uses object handles as arguments and return types
// with one that takes TaggedValues. For example:
// Tuple Pack(Integer, Float, String);
// TaggedValue callable = TFLIB_CALLABLE_ADAPTOR(Pack);
#define TFLIB_CALLABLE_ADAPTOR(x) ::tf::libtf::CreateCallableAdaptor(x, #x)

template <class TF>
TaggedValue CreateCallableAdaptor(TF x, const char* name) {
  return TaggedValue((CallableWrapperUnpackArgs<TF>(x, name)));
}

}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_OBJECT_H_
