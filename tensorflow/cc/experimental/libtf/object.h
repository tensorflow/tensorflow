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
// Object hierarchy for the tensor flow C++ API.
// Objects have attack value and a class which is another object.
//
// Example Usage:
// Object runtime = GetRuntime("tfrt");
// Object module = runtime.Get("Import")("cool_mobilenet")
// runtime.Get("Tensor")(Tuple(5,5,5), 3.3);
// Object test = CreateModule("test");
// test.Set("cool_function", callable);
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

// Allows converting C++ primitives like float, int to handles automatically.
// Allows callable calls and list appends to be more idiomatic.
template <class T>
Handle Convert(T value);

// Base object class that holds all data. it is important that all derive
// classes do not hold any additional data. that means that it is always safe to
// slice down to this value.
class Handle {
 public:
  Handle() : value_(TaggedValue::None()) {}

 public:
  explicit Handle(TaggedValue value) : value_(std::move(value)) {}
  // explicit Handle(TaggedValue value, Handle* class_input)
  //     : value_(std::move(value)), class_(class_input) {}
  // const Handle& type() { return *class_; }

 protected:
  TaggedValue value_;
  // effectively a "weak reference" to intern'd class value.
  // types are compared by comparing pointer values here.
  // Handle* class_;  // effectively a "weak reference" to intern'd class value.

  friend class Integer;
  friend class Float;
  friend class String;
  friend class Object;
  friend class List;
  friend class Dictionary;
  friend class Tuple;
  friend class Callable;
  friend class Tensor;
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
  template <typename Fn, class TRET, class... ArgsOut>
  friend class UneraseCallHelper;
};
template <class T>
tensorflow::StatusOr<T> Cast(Handle handle);

class None final : public Handle {
 public:
  None() : Handle(TaggedValue::None()) {}

 private:
  explicit None(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

class String final : public Handle {
 public:
  explicit String(const char* s) : Handle(TaggedValue(s)) {}
  const char* get() const { return value_.s(); }

 private:
  // Private since it is in general unsafe.
  explicit String(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

class Object : public Handle {
 public:
  Object() : Handle(TaggedValue::Dict()) {}

  static const String* parent_;

  // Get a object member attribute named `key`. Does lookup in referenced
  // object named "__parent__" if key not found locally.
  template <class T = Handle>
  tensorflow::StatusOr<T> Get(const String& key) {
    // static String* parent_token = new String("__parent__");
    auto& dict = value_.dict();
    auto it = dict.find(key.value_);
    if (it != dict.end()) {
      return Cast<T>(Handle(it->second));
    } else {
      // Lookup in object stored by reference in attribute  "__parent__".
      auto it_class = dict.find(parent_->value_);
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
  void Set(const String& key, Handle h) {
    value_.dict()[key.value_] = std::move(h.value_);
  }
  void Unset(const String& key) { value_.dict().erase(key.value_); }
  // Adding dir() is in the future.
 private:
  // Private since it is in general unsafe.
  explicit Object(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

class Dictionary final : public Handle {
 public:
  Dictionary() : Handle(TaggedValue::Dict()) {}
  // TODO(aselle): make this private to preserve invariant.

  template <class T>
  tensorflow::StatusOr<T> Get(const Handle& key) {
    auto it = value_.dict().find(key.value_);
    if (it != value_.dict().end()) return Cast<T>(Handle(it->second));
    return tensorflow::errors::NotFound("Key not in dictionary.");
  }
  void Set(const String& key, Handle value) {
    value_.dict()[key.value_] = std::move(value.value_);
  }
  void Set(const Handle& key, Handle value) {
    value_.dict()[key.value_] = std::move(value.value_);
  }
  size_t size() const { return value_.dict().size(); }

 private:
  // Private since it is in general unsafe.
  explicit Dictionary(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

class Integer final : public Handle {
 public:
  explicit Integer(Handle h) : Handle(h.value_) {}
  explicit Integer(int64_t i) : Handle(TaggedValue(i)) {}
  int64_t& get() { return value_.i64(); }
  const int64_t& get() const { return value_.i64(); }

 private:
  // Private since it is in general unsafe.
  explicit Integer(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

class Float final : public Handle {
 public:
  explicit Float(Handle h) : Handle(h.value_) {}
  explicit Float(float i) : Handle(TaggedValue(i)) {}
  float& get() { return value_.f32(); }
  const float& get() const { return value_.f32(); }

 private:
  // Private since it is in general unsafe.
  explicit Float(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

class Tensor final : public Handle {
 public:
  explicit Tensor(Handle h) : Handle(h.value_) {}

  // Copies contents of tensor into `data`.
  //
  // Raises error if `data` is of invalid size.
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
  return tensorflow::Status::OK();
}

class Tuple : public Handle {
 public:
  template <class... T>
  explicit Tuple(T... args) : Handle(TaggedValue::Tuple()) {
    add(args...);
  }

  template <class T>
  tensorflow::StatusOr<T> Get(size_t i) {
    if (i >= value_.tuple().size())
      return tensorflow::errors::InvalidArgument("Out of bounds index.");
    return Cast<T>(Handle(value_.tuple()[i]));
  }

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

class List final : public Handle {
 public:
  template <class... T>
  explicit List(T... args) : Handle(TaggedValue::List()) {}
  template <class T>
  tensorflow::StatusOr<T> Get(size_t i) {
    if (i >= size()) {
      return tensorflow::errors::InvalidArgument("Out of bounds index.");
    }
    return Cast<T>(Handle(value_.list()[i]));
  }

  tensorflow::Status Set(size_t i, Handle h) {
    if (i >= size()) {
      return tensorflow::errors::InvalidArgument("Out of bounds index.");
    }
    value_.list()[i] = std::move(h.value_);
    return tensorflow::Status::OK();
  }
  template <class T>
  void append(T arg) {
    value_.list().emplace_back(Convert(arg).value_);
  }
  size_t size() const { return value_.list().size(); }

 private:
  // Private since it is in general unsafe.
  explicit List(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

// Import a module named `module`.
Object Import(const TaggedValue& module);
// Idea of injecting random graph def code into a callable form.
// This return an object with members that are the signature entry points
// to this saved model (i.e. module).
Object ImportGraphDef(const char* runtime, const char* graphdef);
Object CreateModule(const char* runtime, const char* name);

// Store keyword argument name value pairs.
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

// callable function object.
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
    return Cast<TReturn>(Handle(maybe_value.ValueOrDie()));
  }

 public:
  // TODO(aselle): need to find a way to write test w/o this being public.
  // Private since it is in general unsafe.
  explicit Callable(TaggedValue v) : Handle(std::move(v)) {}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

namespace internal {
class Capsule final : public Handle {
 public:
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

template <class T>
inline TaggedValue::Type TypeToTaggedType() {}
template <>
inline TaggedValue::Type TypeToTaggedType<Handle>() {
  return TaggedValue::Type::NONE;
}
template <>
inline TaggedValue::Type TypeToTaggedType<None>() {
  return TaggedValue::Type::NONE;
}
template <>
inline TaggedValue::Type TypeToTaggedType<String>() {
  return TaggedValue::Type::STRING;
}
template <>
inline TaggedValue::Type TypeToTaggedType<Callable>() {
  return TaggedValue::Type::FUNC;
}
template <>
inline TaggedValue::Type TypeToTaggedType<Integer>() {
  return TaggedValue::Type::INT64;
}
template <>
inline TaggedValue::Type TypeToTaggedType<Float>() {
  return TaggedValue::Type::FLOAT32;
}
template <>
inline TaggedValue::Type TypeToTaggedType<Object>() {
  return TaggedValue::Type::DICT;
}
template <>
inline TaggedValue::Type TypeToTaggedType<Dictionary>() {
  return TaggedValue::Type::DICT;
}
template <>
inline TaggedValue::Type TypeToTaggedType<List>() {
  return TaggedValue::Type::LIST;
}
template <>
inline TaggedValue::Type TypeToTaggedType<Tensor>() {
  return TaggedValue::Type::TENSOR;
}
template <>
inline TaggedValue::Type TypeToTaggedType<internal::Capsule>() {
  return TaggedValue::Type::CAPSULE;
}
// TODO(unknown): fully populate

template <class T>
tensorflow::StatusOr<T> Cast(Handle handle) {
  if (handle.value_.type() == TypeToTaggedType<T>() ||
      std::is_same<T, Handle>::value)
    return T((std::move(handle.value_)));
  return tensorflow::errors::InvalidArgument("Incompatible cast.");
}

template <>
inline Handle Convert(const char* value) {
  return String(value);
}
template <>
inline Handle Convert(int32_t value) {
  return Integer(value);
}
template <>
inline Handle Convert(int64_t value) {
  return Integer(value);
}
template <>
inline Handle Convert(float value) {
  return Float(value);
}
template <class T>
inline Handle Convert(T value) {
  return Handle(std::move(value));
}

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
