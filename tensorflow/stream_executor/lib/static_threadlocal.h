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

#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STATIC_THREADLOCAL_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STATIC_THREADLOCAL_H_

#ifdef _MSC_VER
#define __thread __declspec(thread) 
#endif

// For POD types in TLS mode, s_obj_VAR is the thread-local variable.
#define SE_STATIC_THREAD_LOCAL_POD(_Type_, _var_)               \
  static __thread _Type_ s_obj_##_var_;                         \
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
