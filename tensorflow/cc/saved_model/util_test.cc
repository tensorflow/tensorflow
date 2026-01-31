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
#include "tensorflow/cc/saved_model/util.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/cc/saved_model/test_utils.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tsl/platform/status_matchers.h"

namespace tensorflow {
namespace saved_model {
namespace {

using tsl::testing::StatusIs;

TEST(UtilTest, TestGetWriteVersionV2) {
  SavedModel saved_model_proto;
  MetaGraphDef* meta_graphdef = saved_model_proto.add_meta_graphs();
  auto* object_graph_def = meta_graphdef->mutable_object_graph_def();
  object_graph_def->add_nodes();
  EXPECT_EQ(GetWriteVersion(saved_model_proto), "2");
}

TEST(UtilTest, TestGetWriteVersionV1) {
  SavedModel saved_model_proto;
  saved_model_proto.add_meta_graphs();
  EXPECT_EQ(GetWriteVersion(saved_model_proto), "1");
  saved_model_proto.add_meta_graphs();
  EXPECT_EQ(GetWriteVersion(saved_model_proto), "1");
}

class GetInputValuesTest : public ::testing::Test {
 public:
  GetInputValuesTest() {
    (*sig_.mutable_inputs())["x"].set_name("feed_x");
    (*sig_.mutable_inputs())["y"].set_name("feed_y");

    (*sig_.mutable_defaults())["x"] = CreateTensorProto(1);
    (*sig_.mutable_defaults())["y"] = CreateTensorProto("A");

    request_["x"] = CreateTensorProto(2);
    request_["y"] = CreateTensorProto("B");

    unaliased_request_["feed_x"] = CreateTensorProto(2);
    unaliased_request_["feed_y"] = CreateTensorProto("B");

    input_x_ = CreateTensorProto(2);
    input_y_ = CreateTensorProto("B");
    default_x_ = CreateTensorProto(1);
    default_y_ = CreateTensorProto("A");
  }

  template <class T>
  TensorProto CreateTensorProto(const T& val) {
    Tensor tensor(val);
    TensorProto tensor_proto;
    tensor.AsProtoTensorContent(&tensor_proto);
    return tensor_proto;
  }

  void ConvertOutputTensorToProto(
      std::vector<std::pair<string, Tensor>>& inputs,
      std::vector<std::pair<string, TensorProto>>& protos) {
    for (const auto& input : inputs) {
      TensorProto tensor_proto;
      input.second.AsProtoTensorContent(&tensor_proto);
      protos.push_back({input.first, std::move(tensor_proto)});
    }
  }

  SignatureDef sig_;
  google::protobuf::Map<std::string, TensorProto> request_;
  std::map<std::string, TensorProto> unaliased_request_;
  TensorProto input_x_, input_y_, default_x_, default_y_;
};

TEST_F(GetInputValuesTest, RequestContainsInvalidInputs) {
  google::protobuf::Map<std::string, TensorProto> local_request = request_;
  local_request["xx"] = CreateTensorProto(2);

  std::vector<std::pair<string, Tensor>> inputs;
  EXPECT_THAT(GetInputValues(sig_, local_request, inputs),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(GetInputValuesTest, RequestContainsAllTheInputs) {
  std::vector<std::pair<string, Tensor>> inputs;
  TF_EXPECT_OK(GetInputValues(sig_, request_, inputs));

  std::vector<std::pair<string, TensorProto>> exp_inputs;
  ConvertOutputTensorToProto(inputs, exp_inputs);

  EXPECT_THAT(exp_inputs,
              UnorderedElementsAre(Pair("feed_x", EqualsProto(input_x_)),
                                   Pair("feed_y", EqualsProto(input_y_))));
}

TEST_F(GetInputValuesTest, RequestContainsNoInputs) {
  google::protobuf::Map<std::string, TensorProto> local_request = request_;
  local_request.erase("x");
  local_request.erase("y");

  std::vector<std::pair<string, Tensor>> inputs;
  TF_EXPECT_OK(GetInputValues(sig_, local_request, inputs));

  std::vector<std::pair<string, TensorProto>> exp_inputs;
  ConvertOutputTensorToProto(inputs, exp_inputs);

  EXPECT_THAT(exp_inputs,
              UnorderedElementsAre(Pair("feed_x", EqualsProto(default_x_)),
                                   Pair("feed_y", EqualsProto(default_y_))));
}

TEST_F(GetInputValuesTest, RequestContainsPartialInputs) {
  google::protobuf::Map<std::string, TensorProto> local_request = request_;
  local_request.erase("y");

  std::vector<std::pair<string, Tensor>> inputs;
  TF_EXPECT_OK(GetInputValues(sig_, local_request, inputs));

  std::vector<std::pair<string, TensorProto>> exp_inputs;
  ConvertOutputTensorToProto(inputs, exp_inputs);

  EXPECT_THAT(exp_inputs,
              UnorderedElementsAre(Pair("feed_x", EqualsProto(input_x_)),
                                   Pair("feed_y", EqualsProto(default_y_))));
}

}  // namespace
}  // namespace saved_model
}  // namespace tensorflow
