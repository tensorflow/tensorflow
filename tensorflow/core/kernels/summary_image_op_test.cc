/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

  void MakeOpV2(int max_images, float vmin, float vmax, bool clip) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ImageSummary")
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Attr("max_images", max_images)
                     .Attr("vmin", vmin)
                     .Attr("vmax", vmax)
                     .Attr("clip", clip)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void CheckAndRemoveEncodedImages(Summary* summary) {
    for (int i = 0; i < summary->value_size(); ++i) {
      Summary::Value* value = summary->mutable_value(i);
      TF_CHECK_OK(WriteStringToFile(
          Env::Default(), strings::StrCat("/tmp/", value->tag(), ".png"),
          value->image().encoded_image_string()));

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
          /* r0, c0, RGB */ 1.0f, 0.1f, 0.2f,
          /* r0, c1, RGB */ 1.0f, 0.3f, 0.4f,
          /* r1, c0, RGB */ 0.0f, 1.0f, 0.0f,
          /* r1, c1, RGB */ 0.0f, 1.0f, 0.0f,
          /* r2, c0, RGB */ 0.0f, 0.0f, 1.0f,
          /* r2, c1, RGB */ 0.0f, 0.0f, 1.0f,
          /* r3, c0, RGB */ 1.0f, 1.0f, 0.0f,
          /* r3, c1, RGB */ 1.0f, 0.0f, 1.0f,
          /* r4, c0, RGB */ 1.0f, 1.0f, 0.0f,
          /* r4, c1, RGB */ 1.0f, 0.0f, 1.0f,
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

TEST_F(SummaryImageOpTest, OneColorImage4dInputWithClip) {
  MakeOpV2(1,     /* max images */
           -2.0f, /*vmin*/
           8.0f,  /*vmax*/
           true /*clip*/);

  // Feed and run
  AddInputFromArray<string>(TensorShape({}), {"tag"});
  AddInputFromArray<float>(
      TensorShape({1 /*batch*/, 5 /*rows*/, 2 /*columns*/, 3 /*depth*/}),
      {
          /* r0, c0, RGB */ -1.1f, -2.5f, -3.9f,
          /* r0, c1, RGB */ -4.1f, -5.5f, -6.9f,
          /* r1, c0, RGB */ 3.1f,  4.5f,  5.9f,
          /* r1, c1, RGB */ 6.1f,  7.5f,  8.9f,
          /* r2, c0, RGB */ 9.1f,  10.5f, 11.9f,
          /* r2, c1, RGB */ 12.1f, 16.5f, 26.9f,
          /* r3, c0, RGB */ 17.1f, 7.5f,  27.9f,
          /* r3, c1, RGB */ 12.1f, 8.5f,  28.9f,
          /* r4, c0, RGB */ 19.1f, 29.5f, 9.9f,
          /* r4, c1, RGB */ 21.1f, 12.5f, 10.9f,
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

TEST_F(SummaryImageOpTest, OneColorImage4dInputWithoutClip) {
  MakeOpV2(1,     /* max images */
           -2.0f, /*vmin*/
           8.0f,  /*vmax*/
           false /*clip*/);

  // Feed and run
  AddInputFromArray<string>(TensorShape({}), {"tag"});
  AddInputFromArray<float>(
      TensorShape({1 /*batch*/, 5 /*rows*/, 2 /*columns*/, 3 /*depth*/}),
      {
          /* r0, c0, RGB */ -1.1f, -2.5f, -3.9f,
          /* r0, c1, RGB */ -4.1f, -5.5f, -6.9f,
          /* r1, c0, RGB */ 3.1f,  4.5f,  5.9f,
          /* r1, c1, RGB */ 6.1f,  7.5f,  8.9f,
          /* r2, c0, RGB */ 9.1f,  10.5f, 11.9f,
          /* r2, c1, RGB */ 12.1f, 16.5f, 26.9f,
          /* r3, c0, RGB */ 17.1f, 7.5f,  27.9f,
          /* r3, c1, RGB */ 12.1f, 8.5f,  28.9f,
          /* r4, c0, RGB */ 19.1f, 29.5f, 9.9f,
          /* r4, c1, RGB */ 21.1f, 12.5f, 10.9f,
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
