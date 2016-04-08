/* Copyright 2015 Google Inc. All Rights Reserved.

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
#include "tensorflow/core/framework/graph.pb.h"
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
// SummaryImageOp
// --------------------------------------------------------------------------
class SummaryImageOpTest : public OpsTestBase {
 protected:
  void MakeOp(int max_images) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ImageSummary")
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Attr("max_images", max_images)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void CheckAndRemoveEncodedImages(Summary* summary) {
    for (int i = 0; i < summary->value_size(); ++i) {
      Summary::Value* value = summary->mutable_value(i);
      ASSERT_TRUE(value->has_image()) << "No image for value: " << value->tag();
      ASSERT_FALSE(value->image().encoded_image_string().empty())
          << "No encoded_image_string for value: " << value->tag();
      if (VLOG_IS_ON(2)) {
        // When LOGGING, output the images to disk for manual inspection.
        TF_CHECK_OK(WriteStringToFile(
            Env::Default(), strings::StrCat("/tmp/", value->tag(), ".png"),
            value->image().encoded_image_string()));
      }
      value->mutable_image()->clear_encoded_image_string();
    }
  }
};

TEST_F(SummaryImageOpTest, ThreeGrayImagesOutOfFive4dInput) {
  MakeOp(3 /* max images */);

  // Feed and run
  AddInputFromArray<string>(TensorShape({}), {"tag"});
  AddInputFromArray<float>(TensorShape({5, 2, 1, 1}),
                           {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<string>()());

  CheckAndRemoveEncodedImages(&summary);
  EXPECT_SummaryMatches(summary, R"(
    value { tag: 'tag/image/0' image { width: 1 height: 2 colorspace: 1} }
    value { tag: 'tag/image/1' image { width: 1 height: 2 colorspace: 1} }
    value { tag: 'tag/image/2' image { width: 1 height: 2 colorspace: 1} }
  )");
}

TEST_F(SummaryImageOpTest, OneGrayImage4dInput) {
  MakeOp(1 /* max images */);

  // Feed and run
  AddInputFromArray<string>(TensorShape({}), {"tag"});
  AddInputFromArray<float>(TensorShape({5 /*batch*/, 2, 1, 1 /*depth*/}),
                           {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<string>()());

  CheckAndRemoveEncodedImages(&summary);
  EXPECT_SummaryMatches(summary, R"(
    value { tag: 'tag/image' image { width: 1 height: 2 colorspace: 1} })");
}

TEST_F(SummaryImageOpTest, OneColorImage4dInput) {
  MakeOp(1 /* max images */);

  // Feed and run
  AddInputFromArray<string>(TensorShape({}), {"tag"});
  AddInputFromArray<float>(
      TensorShape({1 /*batch*/, 5 /*rows*/, 2 /*columns*/, 3 /*depth*/}),
      {
          /* r0, c0, RGB */ 1.0, 0.1, 0.2,
          /* r0, c1, RGB */ 1.0, 0.3, 0.4,
          /* r1, c0, RGB */ 0.0, 1.0, 0.0,
          /* r1, c1, RGB */ 0.0, 1.0, 0.0,
          /* r2, c0, RGB */ 0.0, 0.0, 1.0,
          /* r2, c1, RGB */ 0.0, 0.0, 1.0,
          /* r3, c0, RGB */ 1.0, 1.0, 0.0,
          /* r3, c1, RGB */ 1.0, 0.0, 1.0,
          /* r4, c0, RGB */ 1.0, 1.0, 0.0,
          /* r4, c1, RGB */ 1.0, 0.0, 1.0,
      });
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<string>()());

  CheckAndRemoveEncodedImages(&summary);
  EXPECT_SummaryMatches(summary, R"(
    value { tag: 'tag/image' image { width: 2 height: 5 colorspace: 3} })");
}

}  // namespace
}  // namespace tensorflow
