/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"
#include "tensorflow/c/experimental/saved_model/core/test_utils.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace {

MetaGraphDef ParseMetaGraphDef(absl::string_view text_proto) {
  MetaGraphDef value;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(text_proto, &value));
  return value;
}

class PartiallyReviveSavedModelObjectsTest : public ::testing::Test {
 public:
  PartiallyReviveSavedModelObjectsTest()
      : device_mgr_(testing::CreateTestingDeviceMgr()),
        ctx_(testing::CreateTestingEagerContext(device_mgr_.get())) {}

  ImmediateExecutionContext* context() { return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

// A minimal, valid object graph: root (node 0) with a "signatures" child
// (node 1, the required empty signature_map), a resource node (node 2,
// where `%RESOURCE_CHILD%` is substituted by each test with a malformed or
// dangling ObjectReference), and a function node (node 3) that has no
// concrete functions, so it is skipped by the main revival loop and never
// added to `concrete_functions` -- used to exercise a resource child that
// is in range and really is kind()==kFunction, but was never revived.
constexpr absl::string_view kMetaGraphTemplate = R"(
object_graph_def {
  nodes {
    children {
      node_id: 1
      local_name: "signatures"
    }
    children {
      node_id: 2
      local_name: "my_resource"
    }
    user_object {
      identifier: "_generic_user_object"
      version {
        producer: 1
        min_consumer: 1
      }
    }
  }
  nodes {
    user_object {
      identifier: "signature_map"
      version {
        producer: 1
        min_consumer: 1
      }
    }
  }
  nodes {
    %RESOURCE_CHILD%
    resource {
      device: ""
    }
  }
  nodes {
    function {
    }
  }
}
)";

std::string MetaGraphWithResourceChild(absl::string_view resource_child) {
  std::string text(kMetaGraphTemplate);
  constexpr absl::string_view kPlaceholder = "%RESOURCE_CHILD%";
  size_t pos = text.find(kPlaceholder);
  CHECK(pos != std::string::npos);
  text.replace(pos, kPlaceholder.size(), resource_child);
  return text;
}

TEST_F(PartiallyReviveSavedModelObjectsTest,
       NegativeCreateResourceChildNodeIdIsRejected) {
  MetaGraphDef metagraph = ParseMetaGraphDef(MetaGraphWithResourceChild(R"(
    children {
      node_id: -1
      local_name: "_create_resource"
    }
  )"));

  PartiallyRevivedObjects objects;
  absl::Status status = internal::PartiallyReviveSavedModelObjects(
      metagraph, context(), /*directory=*/"", &objects);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
  EXPECT_THAT(status.message(), ::testing::HasSubstr("out of range"));
}

TEST_F(PartiallyReviveSavedModelObjectsTest,
       OutOfRangeInitializeChildNodeIdIsRejected) {
  MetaGraphDef metagraph = ParseMetaGraphDef(MetaGraphWithResourceChild(R"(
    children {
      node_id: 100
      local_name: "_initialize"
    }
  )"));

  PartiallyRevivedObjects objects;
  absl::Status status = internal::PartiallyReviveSavedModelObjects(
      metagraph, context(), /*directory=*/"", &objects);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
  EXPECT_THAT(status.message(), ::testing::HasSubstr("out of range"));
}

TEST_F(PartiallyReviveSavedModelObjectsTest,
       NegativeDestroyResourceChildNodeIdIsRejected) {
  MetaGraphDef metagraph = ParseMetaGraphDef(MetaGraphWithResourceChild(R"(
    children {
      node_id: -1
      local_name: "_destroy_resource"
    }
  )"));

  PartiallyRevivedObjects objects;
  absl::Status status = internal::PartiallyReviveSavedModelObjects(
      metagraph, context(), /*directory=*/"", &objects);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
  EXPECT_THAT(status.message(), ::testing::HasSubstr("out of range"));
}

// A resource's "_create_resource" child can be in range and genuinely point
// at a SavedObject::kFunction node, yet that node was never added to
// `concrete_functions` (e.g. because it declared no concrete functions and
// was skipped). Using operator[] here would previously default-construct a
// bogus TFConcreteFunctionRevivalState with null fdef/saved_concrete_func;
// this should instead be rejected.
TEST_F(PartiallyReviveSavedModelObjectsTest,
       CreateResourceChildPointingAtUnrevivedFunctionIsRejected) {
  MetaGraphDef metagraph = ParseMetaGraphDef(MetaGraphWithResourceChild(R"(
    children {
      node_id: 3
      local_name: "_create_resource"
    }
  )"));

  PartiallyRevivedObjects objects;
  absl::Status status = internal::PartiallyReviveSavedModelObjects(
      metagraph, context(), /*directory=*/"", &objects);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("not found in revived concrete functions"));
}

TEST_F(PartiallyReviveSavedModelObjectsTest,
       FunctionWithMissingFunctionDefIsRejected) {
  MetaGraphDef metagraph = ParseMetaGraphDef(R"(
object_graph_def {
  nodes {
    children { node_id: 1 local_name: "signatures" }
    user_object { identifier: "_generic_user_object" }
  }
  nodes {
    user_object { identifier: "signature_map" }
  }
  nodes {
    function { concrete_functions: "missing_func" }
  }
}
)");

  PartiallyRevivedObjects objects;
  absl::Status status = internal::PartiallyReviveSavedModelObjects(
      metagraph, context(), /*directory=*/"", &objects);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr(
                  "Unable to find FunctionDef with name 'missing_func'"));
}

TEST_F(PartiallyReviveSavedModelObjectsTest,
       FunctionWithMissingSavedConcreteFunctionIsRejected) {
  MetaGraphDef metagraph = ParseMetaGraphDef(R"(
graph_def {
  library {
    function {
      signature { name: "dangling_func" }
    }
  }
}
object_graph_def {
  nodes {
    children { node_id: 1 local_name: "signatures" }
    user_object { identifier: "_generic_user_object" }
  }
  nodes {
    user_object { identifier: "signature_map" }
  }
  nodes {
    function { concrete_functions: "dangling_func" }
  }
}
)");

  PartiallyRevivedObjects objects;
  absl::Status status = internal::PartiallyReviveSavedModelObjects(
      metagraph, context(), /*directory=*/"", &objects);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
  EXPECT_THAT(
      status.message(),
      ::testing::HasSubstr("Unable to find SavedConcreteFunction with name "
                           "'dangling_func'"));
}

}  // namespace
}  // namespace tensorflow
