/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/compiler/mlir/python/mlir.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/safe_ptr.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {

namespace {

class MlirTest : public ::testing::Test {};

TEST_F(MlirTest, ImportGraphDef) {
  tensorflow::Safe_TF_StatusPtr status = tensorflow::make_safe(TF_NewStatus());
  std::string input_graphdef = R"pb(
    node {
      name: "Const"
      op: "Const"
      attr {
        key: "dtype"
        value { type: DT_INT32 }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape { dim { size: 1 } }
            int_val: 1
          }
        }
      }
    }
    node {
      name: "Const_1"
      op: "Const"
      attr {
        key: "dtype"
        value { type: DT_INT32 }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape { dim { size: 1 } }
            int_val: 2
          }
        }
      }
    }
    node {
      name: "Add"
      op: "Add"
      input: "Const"
      input: "Const_1"
      attr {
        key: "T"
        value { type: DT_INT32 }
      }
    }
  )pb";

  std::string result = ImportGraphDef(input_graphdef, /*pass_pipeline=*/"",
                                      /*show_debug_info=*/false, status.get());

  EXPECT_EQ(TF_GetCode(status.get()), TF_OK);
  EXPECT_TRUE(absl::StrContains(result, "tf.Const"));
}

}  // namespace
}  // namespace tensorflow
