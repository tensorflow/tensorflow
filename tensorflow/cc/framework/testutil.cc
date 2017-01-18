/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/framework/testutil.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/default_device.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

namespace test {

void GetTensors(const Scope& scope, OutputList tensors,
                std::vector<Tensor>* out) {
  ClientSession session(scope);
  TF_CHECK_OK(session.Run(tensors, out));
}

void GetTensor(const Scope& scope, Output tensor, Tensor* out) {
  std::vector<Tensor> outputs;
  GetTensors(scope, {tensor}, &outputs);
  *out = outputs[0];
}

}  // end namespace test
}  // end namespace tensorflow
