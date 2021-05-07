// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

using tensorflow::Pyo;
using tensorflow::TF_StatusPtr;
using tensorflow::TFE_TensorHandleToNumpy;

PYBIND11_MODULE(pywrap_tensor_test_util, m) {
  m.def("get_scalar_one", []() {
    // Builds a TFE_TensorHandle and then converts to NumPy ndarray
    // using TFE_TensorHandleToNumpy.
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TF_StatusPtr status(TF_NewStatus());
    TFE_Context* ctx = TFE_NewContext(opts, status.get());
    TFE_TensorHandle* handle = TestScalarTensorHandle(ctx, 1.0f);
    auto result = Pyo(TFE_TensorHandleToNumpy(handle, status.get()));
    TFE_DeleteTensorHandle(handle);
    TFE_DeleteContext(ctx);
    TFE_DeleteContextOptions(opts);
    return result;
  });
}
