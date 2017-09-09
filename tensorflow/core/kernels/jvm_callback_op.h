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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace {
// Copy of the C Eager API struct due to the circular dependency issue.
  struct TF_Status {
  tensorflow::Status status;
};

// Copy of the C Eager API struct due to the circular dependency issue.
struct TFE_TensorHandle {
  TFE_TensorHandle(const tensorflow::Tensor& t, tensorflow::Device* d)
      : t(t), d(d) {}

  tensorflow::Tensor t;
  tensorflow::Device* d;
};

template <typename T>
T pointerFromString(const std::string &text, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr) {
  std::stringstream ss(text);
  void* pointer;
  ss >> pointer;
  return (T) pointer;
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

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_JVM_CALLBACK_OP_H_
