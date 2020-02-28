/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_CHECK_MACROS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_CHECK_MACROS_H_

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace ruy {
namespace check_macros {

constexpr int kValueBufSize = 32;

template <typename T, typename Enable = void>
struct ToString {
  static void Run(const T& value, char* buf) {
    snprintf(buf, kValueBufSize, "(?)");
  }
};

template <>
struct ToString<float, void> {
  static void Run(float value, char* buf) {
    snprintf(buf, kValueBufSize, "%.9g", static_cast<double>(value));
  }
};

template <>
struct ToString<double, void> {
  static void Run(double value, char* buf) {
    snprintf(buf, kValueBufSize, "%.16g", value);
  }
};

template <typename T>
struct ToString<T, typename std::enable_if<std::is_integral<T>::value>::type> {
  static void Run(const T& value, char* buf) {
    snprintf(buf, kValueBufSize, "%lld", static_cast<long long>(value));
  }
};

template <typename T>
struct ToString<T*, void> {
  static void Run(T* value, char* buf) {
    snprintf(buf, kValueBufSize, "%p", value);
  }
};

template <typename T>
struct ToString<T, typename std::enable_if<std::is_enum<T>::value>::type> {
  static void Run(const T& value, char* buf) {
    snprintf(buf, kValueBufSize, "(enum value %d)", static_cast<int>(value));
  }
};

inline void Failure(const char* file, int line, const char* macro,
                    const char* condition) {
  fprintf(stderr, "%s:%d: %s condition not satisfied: %s\n", file, line, macro,
          condition);
  abort();
}

template <typename LhsType, typename RhsType>
inline void Failure(const char* file, int line, const char* macro,
                    const char* lhs, const LhsType& lhs_value, const char* op,
                    const char* rhs, const RhsType& rhs_value) {
  char lhs_value_buf[kValueBufSize];
  ToString<LhsType>::Run(lhs_value, lhs_value_buf);
  char rhs_value_buf[kValueBufSize];
  ToString<RhsType>::Run(rhs_value, rhs_value_buf);
  fprintf(stderr,
          "%s:%d: %s condition not satisfied:   [ %s %s %s ]   with values   [ "
          "%s %s %s ].\n",
          file, line, macro, lhs, op, rhs, lhs_value_buf, op, rhs_value_buf);
  abort();
}

#define RUY_CHECK_IMPL(macro, condition)                                  \
  do {                                                                    \
    if (!(condition)) {                                                   \
      ruy::check_macros::Failure(__FILE__, __LINE__, #macro, #condition); \
    }                                                                     \
  } while (false)

#define RUY_CHECK_OP_IMPL(macro, lhs, op, rhs)                                \
  do {                                                                        \
    const auto& lhs_value = (lhs);                                            \
    const auto& rhs_value = (rhs);                                            \
    if (!(lhs_value op rhs_value)) {                                          \
      ruy::check_macros::Failure(__FILE__, __LINE__, #macro, #lhs, lhs_value, \
                                 #op, #rhs, rhs_value);                       \
    }                                                                         \
  } while (false)

#define RUY_CHECK(condition) RUY_CHECK_IMPL(RUY_CHECK, condition)
#define RUY_CHECK_EQ(x, y) RUY_CHECK_OP_IMPL(RUY_CHECK_EQ, x, ==, y)
#define RUY_CHECK_NE(x, y) RUY_CHECK_OP_IMPL(RUY_CHECK_NE, x, !=, y)
#define RUY_CHECK_GE(x, y) RUY_CHECK_OP_IMPL(RUY_CHECK_GE, x, >=, y)
#define RUY_CHECK_GT(x, y) RUY_CHECK_OP_IMPL(RUY_CHECK_GT, x, >, y)
#define RUY_CHECK_LE(x, y) RUY_CHECK_OP_IMPL(RUY_CHECK_LE, x, <=, y)
#define RUY_CHECK_LT(x, y) RUY_CHECK_OP_IMPL(RUY_CHECK_LT, x, <, y)

#ifdef NDEBUG
#define RUY_DCHECK(condition)
#define RUY_DCHECK_EQ(x, y)
#define RUY_DCHECK_NE(x, y)
#define RUY_DCHECK_GE(x, y)
#define RUY_DCHECK_GT(x, y)
#define RUY_DCHECK_LE(x, y)
#define RUY_DCHECK_LT(x, y)
#else
#define RUY_DCHECK(condition) RUY_CHECK(condition)
#define RUY_DCHECK_EQ(x, y) RUY_CHECK_EQ(x, y)
#define RUY_DCHECK_NE(x, y) RUY_CHECK_NE(x, y)
#define RUY_DCHECK_GE(x, y) RUY_CHECK_GE(x, y)
#define RUY_DCHECK_GT(x, y) RUY_CHECK_GT(x, y)
#define RUY_DCHECK_LE(x, y) RUY_CHECK_LE(x, y)
#define RUY_DCHECK_LT(x, y) RUY_CHECK_LT(x, y)
#endif

}  // end namespace check_macros
}  // end namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_CHECK_MACROS_H_
