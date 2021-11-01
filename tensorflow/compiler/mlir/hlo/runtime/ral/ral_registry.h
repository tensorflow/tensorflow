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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_RUNTIME_RAL_RAL_REGISTRY_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_RUNTIME_RAL_RAL_REGISTRY_H_

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "ral/ral_base.h"

namespace mlir {
namespace disc_ral {

/////////////////////////////////////////////////////
//===----------------------------------------------------------------------===//
// MemRefType implementation
//===----------------------------------------------------------------------===//

// A struct that corresponds to how MLIR represents memref.
template <typename T, int N>
struct MemRefType {
  T* basePtr;  // NOLINT
  T* data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

// A specialization for rank-0 memref.
template <typename T>
struct MemRefType<T, 0> {
  T* basePtr;  // NOLINT
  T* data;
  int64_t offset;
};

/////////////////////////////////////////////////////
//===----------------------------------------------------------------------===//
// A helper struct to convert a c++ type to its corresponding encoding string.
//===----------------------------------------------------------------------===//

template <typename T>
struct TypeEncoder;

#define DISC_DEFINE_TYPE_NAME(type, name)               \
  template <>                                           \
  struct TypeEncoder<type> {                            \
    static inline std::string Invoke() { return name; } \
  };

DISC_DEFINE_TYPE_NAME(bool, "i1");
DISC_DEFINE_TYPE_NAME(int16_t, "i16");
DISC_DEFINE_TYPE_NAME(int32_t, "i32");
DISC_DEFINE_TYPE_NAME(int64_t, "i64");
DISC_DEFINE_TYPE_NAME(size_t, "i64");
DISC_DEFINE_TYPE_NAME(float, "f32");
DISC_DEFINE_TYPE_NAME(double, "f64");
DISC_DEFINE_TYPE_NAME(void, "void");
DISC_DEFINE_TYPE_NAME(char, "i8");

template <typename T, int N>
struct TypeEncoder<MemRefType<T, N>> {
  static inline std::string Invoke() {
    std::ostringstream out;
    out << "m" << N << "d" << TypeEncoder<T>::Invoke();
    return out.str();
  }
};

// Erases the concrete type of pointer (e.g. char* -> void*)
template <typename T>
struct TypeEncoder<T*> {
  static inline std::string Invoke() {
    return std::is_pointer<T>::value
               ? "p" + TypeEncoder<typename std::remove_cv<T>::type>::Invoke()
               : "pvoid";
  }
};

/////////////////////////////////////////////////////
//===----------------------------------------------------------------------===//
// A helper struct to encode a disc supported function.
//===----------------------------------------------------------------------===//

template <typename... Remaining>
struct VariadicTypeEncoder;

template <typename T, typename... Remaining>
struct VariadicTypeEncoder<T, Remaining...> {
  static inline std::string Invoke() {
    return TypeEncoder<T>::Invoke() + "_" +
           VariadicTypeEncoder<Remaining...>::Invoke();
  }
};

template <>
struct VariadicTypeEncoder<> {
  static inline std::string Invoke() { return ""; }
};

template <typename F>
struct FunctionEncoder;

template <typename Return, typename... Args>
struct FunctionEncoder<Return (*)(Args...)> {
  static inline std::string Invoke(const std::string& prefix) {
    return prefix + "___" + VariadicTypeEncoder<Args...>::Invoke() + "__" +
           TypeEncoder<Return>::Invoke();
  }
};

template <typename R, typename F, typename... ArgTypes>
struct FunctionWrapperImpl;

template <typename R, typename F, typename T, typename... RemainingArgTypes>
struct FunctionWrapperImpl<R, F, T, RemainingArgTypes...> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    FunctionWrapperImpl<R, F, RemainingArgTypes...>::Invoke(
        f, args + 1, std::forward<ParsedArgs>(parsed_args)...,
        *static_cast<T*>(args[0]));
  }
};

// A specialization for MemRefType (memref rank > 0)
template <typename R, typename F, typename T, int N,
          typename... RemainingArgTypes>
struct FunctionWrapperImpl<R, F, MemRefType<T, N>, RemainingArgTypes...> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    MemRefType<T, N> memref;
    memref.basePtr = *static_cast<T**>(*args++);
    memref.data = *static_cast<T**>(*args++);
    memref.offset = *static_cast<int64_t*>(*args++);
    for (int i = 0; i < N; ++i) {
      memref.sizes[i] = *static_cast<int64_t*>(*args++);
    }
    for (int i = 0; i < N; ++i) {
      memref.strides[i] = *static_cast<int64_t*>(*args++);
    }
    FunctionWrapperImpl<R, F, RemainingArgTypes...>::Invoke(
        f, args, std::forward<ParsedArgs>(parsed_args)..., std::move(memref));
  }
};

// A specialization for rank-0 MemRefType
template <typename R, typename F, typename T, typename... RemainingArgTypes>
struct FunctionWrapperImpl<R, F, MemRefType<T, 0>, RemainingArgTypes...> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    MemRefType<T, 0> memref;
    memref.basePtr = *static_cast<T**>(*args++);
    memref.data = *static_cast<T**>(*args++);
    memref.offset = *static_cast<int64_t*>(*args++);
    FunctionWrapperImpl<R, F, RemainingArgTypes...>::Invoke(
        f, args, std::forward<ParsedArgs>(parsed_args)..., std::move(memref));
  }
};

template <typename R, typename F>
struct FunctionWrapperImpl<R, F> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    *static_cast<R*>(*args) =
        std::move(f(std::forward<ParsedArgs>(parsed_args)...));
  }
};

template <typename F>
struct FunctionWrapperImpl<void, F> {
  template <typename... ParsedArgs>
  static inline void Invoke(F f, void** args, ParsedArgs&&... parsed_args) {
    f(std::forward<ParsedArgs>(parsed_args)...);
  }
};

template <typename F, F f>
struct FunctionWrapper;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct FunctionWrapper<Return (*)(Args...), impl_fn> {
  static void Invoke(void** args) {
    FunctionWrapperImpl<Return, Return (*)(Args...), Args...>::Invoke(impl_fn,
                                                                      args);
  }
};

class FunctionRegistry {
 public:
  ~FunctionRegistry();

  // process-level singleton
  static FunctionRegistry& Global();

  // Inserts a ral function to the function registry using key `name` and
  // returns true if success.
  bool Register(const std::string& name, ral_func_t api_func);

  // Returns the function corresponding to the key `name` or null if not found.
  ral_func_t Find(const std::string& name);

 private:
  FunctionRegistry();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Macros used to define DISC RAL functions.
#define DISC_RAL_FUNCTION(name, device, ...) \
  DISC_RAL_FUNCTION_UNIQ_HELPER(name, device, __COUNTER__, __VA_ARGS__)

#define DISC_RAL_FUNCTION_UNIQ_HELPER(name, device, ctr, ...) \
  DISC_RAL_API_UNIQ(name, device, ctr, __VA_ARGS__)

#define DISC_RAL_API_UNIQ(name, device, ctr, ...)                            \
  static bool unused_ret_val_##ctr =                                         \
      ::mlir::disc_ral::FunctionRegistry::Global().Register(                 \
          ::mlir::disc_ral::FunctionEncoder<decltype(&__VA_ARGS__)>::Invoke( \
              std::string(name) + "___" + std::string(device)),              \
          ::mlir::disc_ral::FunctionWrapper<decltype(&__VA_ARGS__),          \
                                            &__VA_ARGS__>::Invoke);

#define DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE_WITH_RANK(T, N, name, d, ...) \
  DISC_RAL_FUNCTION(name, d, __VA_ARGS__<T, N>)

#define DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE(T, name, d, ...)                \
  DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE_WITH_RANK(T, 0, name, d, __VA_ARGS__) \
  DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE_WITH_RANK(T, 1, name, d, __VA_ARGS__) \
  DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE_WITH_RANK(T, 2, name, d, __VA_ARGS__) \
  DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE_WITH_RANK(T, 3, name, d, __VA_ARGS__) \
  DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE_WITH_RANK(T, 4, name, d, __VA_ARGS__) \
  DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE_WITH_RANK(T, 5, name, d, __VA_ARGS__) \
  DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE_WITH_RANK(T, 6, name, d, __VA_ARGS__) \
  DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE_WITH_RANK(T, 7, name, d, __VA_ARGS__) \
  DISC_RAL_RANK_SPECIALIZATION_FOR_TYPE_WITH_RANK(T, 8, name, d, __VA_ARGS__)

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_RUNTIME_RAL_RAL_REGISTRY_H_
