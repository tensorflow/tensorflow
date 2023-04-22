/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <memory>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

static void EXPECT_SummaryMatches(const Summary& actual,
                                  const string& expected_str) {
  Summary expected;
  CHECK(protobuf::TextFormat::ParseFromString(expected_str, &expected));
  EXPECT_EQ(expected.DebugString(), actual.DebugString());
}

// --------------------------------------------------------------------------
// SummaryTensorOpV2
// --------------------------------------------------------------------------
class SummaryTensorOpV2Test : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("myop", "TensorSummaryV2")
                     .Input(FakeInput(DT_STRING))
                     .Input(FakeInput(DT_STRING))
                     .Input(FakeInput(DT_STRING))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SummaryTensorOpV2Test, BasicPluginData) {
  MakeOp();

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({}), {"tag_foo"});
  AddInputFromArray<tstring>(TensorShape({}), {"some string tensor content"});

  // Create a SummaryMetadata that stores data for 2 plugins.
  SummaryMetadata summary_metadata;
  SummaryMetadata::PluginData* plugin_data =
      summary_metadata.mutable_plugin_data();
  plugin_data->set_plugin_name("foo");
  plugin_data->set_content("content_for_plugin_foo");
  AddInputFromArray<tstring>(TensorShape({}),
                             {summary_metadata.SerializeAsString()});

  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());
  ASSERT_EQ(1, summary.value_size());

  // Check the content of the tensor stored in the summary.
  Tensor string_content_tensor;
  CHECK(string_content_tensor.FromProto(summary.value(0).tensor()));
  ASSERT_EQ("some string tensor content",
            string_content_tensor.scalar<tstring>()());

  // Check plugin-related data.
  ASSERT_EQ("tag_foo", summary.value(0).tag());
  ASSERT_EQ("foo", summary.value(0).metadata().plugin_data().plugin_name());
  ASSERT_EQ("content_for_plugin_foo",
            summary.value(0).metadata().plugin_data().content());
}

}  // namespace
}  // namespace tensorflow
