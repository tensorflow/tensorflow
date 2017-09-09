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

#ifndef TENSORFLOW_KERNELS_JVM_CALLBACK_OP_H_
#define TENSORFLOW_KERNELS_JVM_CALLBACK_OP_H_

#include <jni.h>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

namespace {
template <class T>
inline T* require_handle(JNIEnv* env, jlong handle, const char* object_name) {
  static_assert(sizeof(jlong) >= sizeof(T*), "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    std::stringstream msg;
    msg << "Object '" << object_name << "' has been disposed already.";
    throw_exception(env, jvm_null_pointer_exception, msg.str().c_str());
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}
}

namespace tensorflow {

// A call to the registered JVM function.
struct JVMCall {
  JNIEnv* env;
  jclass registry;
  jmethodID call_method_id;

  // Passed to the JVM to call the function registered with this ID.
  int id;

  // Inputs and outputs of this function invocation.
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
};

struct JVMCallbackOp : public OpKernel {
  explicit JVMCallbackOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_JVM_CALLBACK_OP_H_
