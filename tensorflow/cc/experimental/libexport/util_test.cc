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
#include "tensorflow/cc/experimental/libexport/util.h"

#include <string>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {
namespace libexport {
namespace {

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

}  // namespace
}  // namespace libexport
}  // namespace tensorflow
