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

#include "tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/revived_objects.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/test_utils.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {
namespace {

class PartiallyRevivedObjectsTest : public ::testing::Test {
 public:
  PartiallyRevivedObjectsTest()
      : device_mgr_(testing::CreateTestingDeviceMgr()),
        ctx_(testing::CreateTestingEagerContext(device_mgr_.get())) {}

  ImmediateExecutionContext* context() { return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

// A concrete function whose SavedConcreteFunction declares a negative
// bound_inputs node id should be rejected, not read out of bounds.
TEST_F(PartiallyRevivedObjectsTest, NegativeBoundInputNodeIdIsRejected) {
  SavedObjectGraph obj_graph;
  obj_graph.add_nodes();

  SavedConcreteFunction saved_concrete_func;
  saved_concrete_func.add_bound_inputs(-1);

  FunctionDef fdef;

  TFConcreteFunctionRevivalState state;
  state.node_id = 0;
  state.fdef = &fdef;
  state.saved_concrete_func = &saved_concrete_func;

  PartiallyRevivedObjects objects;
  objects.concrete_functions[0] = std::move(state);

  RevivedObjects revived;
  absl::Status status = objects.Build(context(), obj_graph, &revived);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
  EXPECT_THAT(status.message(), ::testing::HasSubstr("out of range"));
}

// A concrete function whose SavedConcreteFunction declares a bound_inputs
// node id past the end of the object graph should be rejected, not read out
// of bounds.
TEST_F(PartiallyRevivedObjectsTest, OutOfRangeBoundInputNodeIdIsRejected) {
  SavedObjectGraph obj_graph;
  obj_graph.add_nodes();

  SavedConcreteFunction saved_concrete_func;
  saved_concrete_func.add_bound_inputs(5);

  FunctionDef fdef;

  TFConcreteFunctionRevivalState state;
  state.node_id = 0;
  state.fdef = &fdef;
  state.saved_concrete_func = &saved_concrete_func;

  PartiallyRevivedObjects objects;
  objects.concrete_functions[0] = std::move(state);

  RevivedObjects revived;
  absl::Status status = objects.Build(context(), obj_graph, &revived);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION) << status.message();
  EXPECT_THAT(status.message(), ::testing::HasSubstr("out of range"));
}

}  // namespace
}  // namespace tensorflow
