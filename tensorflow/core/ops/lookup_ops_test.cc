/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(LookupOpsTest, LookupTableFindV2_ShapeFn) {
  ShapeInferenceTestOp op("LookupTableFindV2");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[?];?;?");
  TF_ASSERT_OK(NodeDefBuilder("test", "LookupTableFindV2")
                   .Input({"table_handle", 0, DT_RESOURCE})
                   .Input({"keys", 0, DT_INT64})
                   .Input({"default_value", 0, DT_FLOAT})
                   .Attr("Tin", DT_INT64)
                   .Attr("Tout", DT_FLOAT)
                   .Finalize(&op.node_def));
  std::vector<std::vector<ShapeInferenceTestOp::ShapeAndType>> types;
  auto set_types = [&op, &types](DataType key_type, DataType value_type) {
    types.emplace_back();
    auto& table = types.back();
    table.emplace_back("[3]", key_type);
    table.emplace_back("[4]", value_type);
    op.input_resource_handle_shapes_and_types = {&table, nullptr, nullptr};
  };
  // If there's no input handle shapes and types, output shape is unknown.
  INFER_OK(op, "[];[?,3];[4]", "?");
  // Set input handle with mismatched key type.
  set_types(DT_INT32, DT_FLOAT);
  INFER_ERROR("read value with wrong dtype", op, "[];[?,3];[4]");
  // Set input handle with mismatched value type.
  set_types(DT_INT64, DT_INT64);
  INFER_ERROR("read value with wrong dtype", op, "[];[?,3];[4]");
  // Set input handle with matched types.
  set_types(DT_INT64, DT_FLOAT);
  INFER_OK(op, "[];[?,3];[4]", "[d1_0,4]");
  INFER_OK(op, "[];[1,3];[4]", "[d1_0,4]");
  INFER_OK(op, "[];[1,?];[4]", "[d1_0,4]");
}

TEST(LookupOpsTest, LookupTableExportV2_ShapeFn) {
  ShapeInferenceTestOp op("LookupTableExportV2");
  TF_ASSERT_OK(NodeDefBuilder("test", "LookupTableExportV2")
                   .Input({"table_handle", 0, DT_RESOURCE})
                   .Attr("Tkeys", DT_INT64)
                   .Attr("Tvalues", DT_FLOAT)
                   .Finalize(&op.node_def));
  std::vector<std::vector<ShapeInferenceTestOp::ShapeAndType>> types;
  auto set_types = [&op, &types](DataType key_type, DataType value_type) {
    types.emplace_back();
    auto& table = types.back();
    table.emplace_back("[3]", key_type);
    table.emplace_back("[4]", value_type);
    op.input_resource_handle_shapes_and_types = {&table};
  };
  // Set input handle with mismatched key type.
  set_types(DT_INT32, DT_FLOAT);
  INFER_ERROR("read value with wrong dtype", op, "[]");
  // Set input handle with mismatched value type.
  set_types(DT_INT64, DT_INT64);
  INFER_ERROR("read value with wrong dtype", op, "[]");
  // Set input handle with matched types.
  set_types(DT_INT64, DT_FLOAT);
  INFER_OK(op, "[]", "?;?");
}

// TODO(b/169969017): add shape fn tests for rest of the ops.

}  // namespace
}  // namespace tensorflow
