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
  __lhs = __name.ConsumeValueOrDie();

// Early-returns the status if it is in error; otherwise, assigns the
// right-hand-side expression to the left-hand-side expression.
//
// The right-hand-side expression is guaranteed to be evaluated exactly once.
#define SE_ASSIGN_OR_RETURN(__lhs, __rhs) \
  SE_ASSIGN_OR_RETURN_IMPL(__lhs, __rhs,  \
                           SE_MACRO_CONCAT(__status_or_value, __COUNTER__))

// Logs the status and returns false if it is in error; otherwise, returns true.
//
// The argument expression is guaranteed to be evaluated exactly once.
//
// TODO(leary) remove as many of these as possible with port::Status
// proliferation.
#define SE_RETURN_STATUS_AS_BOOL(__status) \
  do {                                     \
    auto status = __status;                \
    if (__status.ok()) {                   \
      return true;                         \
    }                                      \
    LOG(ERROR) << status;                  \
    return false;                          \
  } while (false)

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_STATUS_MACROS_H_
