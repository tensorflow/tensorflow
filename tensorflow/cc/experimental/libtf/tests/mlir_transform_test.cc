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
#include "tensorflow/cc/experimental/libtf/mlir/mlir_transform.h"

#include <iostream>
#include <string>

#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {

TEST(TransformTest, LoadSavedModel) {
  Object mlir = MLIR();
  TF_ASSERT_OK_AND_ASSIGN(Callable load,
                          mlir.Get<Callable>(String("LoadSavedModel")));

  TF_ASSERT_OK_AND_ASSIGN(
      Handle model_bad,
      load.Call<Handle>(mlir, String("/error/doesnotexist___31284382")));
  TF_ASSERT_OK(Cast<None>(model_bad).status());

  const std::string model_good_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/cc/experimental/libtf/tests/testdata/simple-model");

  TF_ASSERT_OK_AND_ASSIGN(
      Object model_good,
      load.Call<Object>(mlir, String(model_good_path.c_str())));

  TF_ASSERT_OK_AND_ASSIGN(Callable to_string,
                          model_good.Get<Callable>(String("ToString")));

  TF_ASSERT_OK_AND_ASSIGN(String s, to_string.Call<String>(model_good));

  ASSERT_GT(strlen(s.get()), 0);
}

}  // namespace libtf
}  // namespace tf
