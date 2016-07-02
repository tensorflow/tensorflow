/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ArrayOpsTest, Pack_ShapeFn) {
  std::unique_ptr<NodeDef> def_storage(new NodeDef);
  NodeDef* def = def_storage.get();
  auto set_axis = [def](int axis) {
    TF_CHECK_OK(NodeDefBuilder("test", "Pack")
                    .Input({{"a", 0, DT_FLOAT}})
                    .Attr("axis", axis)
                    .Finalize(def));
  };
  const char op[] = "Pack";

  set_axis(0);
  INFER_OK_WITH_DEF(op, def, "?;?;?", "?");

  for (int axis : {0, -3}) {
    set_axis(axis);
    INFER_OK_WITH_DEF(op, def, "?;?", "?");
    INFER_OK_WITH_DEF(op, def, "[1,3];[1,3];?", "[3,d0_0|d1_0,d0_1|d1_1]");
    INFER_OK_WITH_DEF(op, def, "[?,3];[1,3];?", "[3,d1_0,d0_1|d1_1]");
    INFER_OK_WITH_DEF(op, def, "[?,?];[1,3];?", "[3,d1_0,d1_1]");
  }
  for (int axis : {1, -2}) {
    set_axis(axis);
    INFER_OK_WITH_DEF(op, def, "?;?", "?");
    INFER_OK_WITH_DEF(op, def, "[1,3];[1,3];?", "[d0_0|d1_0,3,d0_1|d1_1]");
    INFER_OK_WITH_DEF(op, def, "[?,3];[1,3];?", "[d1_0,3,d0_1|d1_1]");
    INFER_OK_WITH_DEF(op, def, "[?,?];[1,3];?", "[d1_0,3,d1_1]");
  }
  for (int axis : {2, -1}) {
    set_axis(axis);
    INFER_OK_WITH_DEF(op, def, "?;?", "?");
    INFER_OK_WITH_DEF(op, def, "[1,3];[1,3];?", "[d0_0|d1_0,d0_1|d1_1,3]");
    INFER_OK_WITH_DEF(op, def, "[?,3];[1,3];?", "[d1_0,d0_1|d1_1,3]");
    INFER_OK_WITH_DEF(op, def, "[?,?];[1,3];?", "[d1_0,d1_1,3]");
  }

  set_axis(-4);
  INFER_ERROR_WITH_DEF("Invalid axis: -4; must be in [-3,3)", op, def,
                       "[1,3];[1,3];?");
  set_axis(3);
  INFER_ERROR_WITH_DEF("Invalid axis: 3; must be in [-3,3)", op, def,
                       "[1,3];[1,3];?");

  set_axis(0);
  INFER_ERROR_WITH_DEF(("Shapes must be equal rank, but are 3 and 2"
                        "\n\tFrom merging shape 0 with other shapes."),
                       op, def, "[1,2,3];?;[1,4]");
}

TEST(ArrayOpsTest, UnPack_ShapeFn) {
  std::unique_ptr<NodeDef> def_storage(new NodeDef);
  NodeDef* def = def_storage.get();
  auto set_axis = [def](int axis) {
    TF_CHECK_OK(NodeDefBuilder("test", "Unpack")
                    .Input("a", 0, DT_FLOAT)
                    .Attr("axis", axis)
                    .Finalize(def));
  };
  const char op[] = "Unpack";

  set_axis(0);
  INFER_OK_WITH_DEF(op, def, "?;?;?", "?");

  for (int axis : {0, -3}) {
    set_axis(axis);
    INFER_OK_WITH_DEF(op, def, "?", "?");
    INFER_OK_WITH_DEF(op, def, "[1,2,3]", "[d0_1,d0_2]");
    INFER_OK_WITH_DEF(op, def, "[?,?,?]", "[d0_1,d0_2]");
  }
  for (int axis : {1, -2}) {
    set_axis(axis);
    INFER_OK_WITH_DEF(op, def, "[1,2,3]", "[d0_0,d0_2]");
    INFER_OK_WITH_DEF(op, def, "[?,?,?]", "[d0_0,d0_2]");
  }
  for (int axis : {2, -1}) {
    set_axis(axis);
    INFER_OK_WITH_DEF(op, def, "[1,2,3]", "[d0_0,d0_1]");
    INFER_OK_WITH_DEF(op, def, "[?,?,?]", "[d0_0,d0_1]");
  }

  set_axis(-4);
  INFER_ERROR_WITH_DEF("Invalid axis: -4; must be in [-3,3)", op, def,
                       "[1,2,3]");
  set_axis(3);
  INFER_ERROR_WITH_DEF("Invalid axis: 3; must be in [-3,3)", op, def,
                       "[1,2,3]");
}

TEST(ArrayOpsTest, Const_ShapeFn) {
  std::unique_ptr<NodeDef> def_storage(new NodeDef);
  NodeDef* def = def_storage.get();
  TensorProto tensor_proto;
  auto* shape_proto = tensor_proto.mutable_tensor_shape();
  auto rebuild_node_def = [def, &tensor_proto]() {
    TF_CHECK_OK(NodeDefBuilder("test", "Const")
                    .Attr("value", tensor_proto)
                    .Finalize(def));
  };
  const char op[] = "Const";

  TensorShape{}.AsProto(shape_proto);
  rebuild_node_def();
  INFER_OK_WITH_DEF(op, def, "", "[]");
  TensorShape{1, 2, 3, 4}.AsProto(shape_proto);
  rebuild_node_def();
  INFER_OK_WITH_DEF(op, def, "", "[1,2,3,4]");

  shape_proto->add_dim()->set_size(-1);
  rebuild_node_def();
  INFER_ERROR_WITH_DEF("Shape [1,2,3,4,-1] has negative dimensions", op, def,
                       "");
}

}  // end namespace tensorflow
