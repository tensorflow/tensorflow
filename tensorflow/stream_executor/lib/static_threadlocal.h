// Copyright 2006 Google Inc.
// All rights reserved.
// Author: Yaz Saito (saito@google.com)
#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STATIC_THREADLOCAL_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STATIC_THREADLOCAL_H_

// For POD types in TLS mode, s_obj_VAR is the thread-local variable.
#define SE_STATIC_THREAD_LOCAL_POD(_Type_, _var_)               \
  static thread_local _Type_ s_obj_##_var_;                     \
  namespace {                                                   \
  class ThreadLocal_##_var_ {                                   \
  public:                                                       \
    ThreadLocal_##_var_() {}                                    \
    void Init() {}                                              \
    inline _Type_ *pointer() const {                            \
      return &s_obj_##_var_;                                    \
    }                                                           \
    inline _Type_ *safe_pointer() const {                       \
      return &s_obj_##_var_;                                    \
    }                                                           \
    _Type_ &get() const {                                       \
      return s_obj_##_var_;                                     \
    }                                                           \
    bool is_native_tls() const { return true; }                 \
  private:                                                      \
    SE_DISALLOW_COPY_AND_ASSIGN(ThreadLocal_##_var_);           \
  } _var_;                                                      \
  }

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_STATIC_THREADLOCAL_H_
