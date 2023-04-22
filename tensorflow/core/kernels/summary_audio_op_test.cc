/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
// SummaryAudioOp
// --------------------------------------------------------------------------
class SummaryAudioOpTest : public OpsTestBase {
 protected:
  void MakeOp(const int max_outputs) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "AudioSummaryV2")
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Attr("max_outputs", max_outputs)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void CheckAndRemoveEncodedAudio(Summary* summary) {
    for (int i = 0; i < summary->value_size(); ++i) {
      Summary::Value* value = summary->mutable_value(i);
      ASSERT_TRUE(value->has_audio()) << "No audio for value: " << value->tag();
      ASSERT_FALSE(value->audio().encoded_audio_string().empty())
          << "No encoded_audio_string for value: " << value->tag();
      if (VLOG_IS_ON(2)) {
        // When LOGGING, output the audio to disk for manual inspection.
        TF_CHECK_OK(WriteStringToFile(
            Env::Default(), strings::StrCat("/tmp/", value->tag(), ".wav"),
            value->audio().encoded_audio_string()));
      }
      value->mutable_audio()->clear_encoded_audio_string();
    }
  }
};

TEST_F(SummaryAudioOpTest, Basic3D) {
  const float kSampleRate = 44100.0f;
  const int kMaxOutputs = 3;
  MakeOp(kMaxOutputs);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({}), {"tag"});
  AddInputFromArray<float>(TensorShape({4, 2, 2}),
                           {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  AddInputFromArray<float>(TensorShape({}), {kSampleRate});

  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());

  CheckAndRemoveEncodedAudio(&summary);
  EXPECT_SummaryMatches(summary, R"(
    value { tag: 'tag/audio/0'
            audio { content_type: "audio/wav" sample_rate: 44100 num_channels: 2
                    length_frames: 2 } }
    value { tag: 'tag/audio/1'
            audio { content_type: "audio/wav" sample_rate: 44100 num_channels: 2
                    length_frames: 2 } }
    value { tag: 'tag/audio/2'
            audio { content_type: "audio/wav" sample_rate: 44100 num_channels: 2
                    length_frames: 2 } }
  )");
}

TEST_F(SummaryAudioOpTest, Basic2D) {
  const float kSampleRate = 44100.0f;
  const int kMaxOutputs = 3;
  MakeOp(kMaxOutputs);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({}), {"tag"});
  AddInputFromArray<float>(TensorShape({4, 4}),
                           {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  AddInputFromArray<float>(TensorShape({}), {kSampleRate});

  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());

  CheckAndRemoveEncodedAudio(&summary);
  EXPECT_SummaryMatches(summary, R"(
    value { tag: 'tag/audio/0'
            audio { content_type: "audio/wav" sample_rate: 44100 num_channels: 1
                    length_frames: 4 } }
    value { tag: 'tag/audio/1'
            audio { content_type: "audio/wav" sample_rate: 44100 num_channels: 1
                    length_frames: 4 } }
    value { tag: 'tag/audio/2'
            audio { content_type: "audio/wav" sample_rate: 44100 num_channels: 1
                    length_frames: 4 } }
  )");
}

}  // namespace
}  // namespace tensorflow
