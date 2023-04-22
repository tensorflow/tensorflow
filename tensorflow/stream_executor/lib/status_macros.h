/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Helper macros for dealing with the port::Status datatype.

#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STATUS_MACROS_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STATUS_MACROS_H_

// Early-returns the status if it is in error; otherwise, proceeds.
//
// The argument expression is guaranteed to be evaluated exactly once.
#define SE_RETURN_IF_ERROR(__status) \
  do {                               \
    auto status = __status;          \
    if (!status.ok()) {              \
      return status;                 \
    }                                \
  } while (false)

// Identifier concatenation helper macros.
#define SE_MACRO_CONCAT_INNER(__x, __y) __x##__y
#define SE_MACRO_CONCAT(__x, __y) SE_MACRO_CONCAT_INNER(__x, __y)

// Implementation of SE_ASSIGN_OR_RETURN that uses a unique temporary identifier
// for avoiding collision in the enclosing scope.
#define SE_ASSIGN_OR_RETURN_IMPL(__lhs, __rhs, __name) \
  auto __name = (__rhs);                               \
  if (!__name.ok()) {                                  \
    return __name.status();                            \
  }                                                    \
  __lhs = std::move(__name.ValueOrDie());

// Early-returns the status if it is in error; otherwise, assigns the
// right-hand-side expression to the left-hand-side expression.
//
// The right-hand-side expression is guaranteed to be evaluated exactly once.
#define SE_ASSIGN_OR_RETURN(__lhs, __rhs) \
  SE_ASSIGN_OR_RETURN_IMPL(__lhs, __rhs,  \
                           SE_MACRO_CONCAT(__status_or_value, __COUNTER__))

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_STATUS_MACROS_H_
