/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <ostream>
#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/odml_sdpa_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

struct SDPATestParams {
  std::string model_name;
  std::string custom_test_name;
  int batch;
  int input_seq_len;
  int max_seq_len;
  int q_heads;
  int kv_heads;
  int head_dim;  // embedding_dim//q_heads
};

void PrintTo(const SDPATestParams& p, std::ostream* os) {
  if (p.model_name != kOdmlSdpaCustom) {
    *os << "{ TFLite file: " << p.model_name << ".tflite.bin }";
  } else {
    *os << "{ Custom test: " << p.custom_test_name << ", b:" << p.batch
        << ", isl:" << p.input_seq_len << ", msl:" << p.max_seq_len
        << ", q:" << p.q_heads << ", k:" << p.kv_heads << "h:" << p.head_dim
        << " }";
  }
}

std::string TestName(const testing::TestParamInfo<SDPATestParams>& info) {
  if (info.param.model_name != kOdmlSdpaCustom) {
    return info.param.model_name;
  }
  return "CustomOp" + info.param.custom_test_name;
}

class SDPATest : public testing::TestWithParam<SDPATestParams> {};

TEST_P(SDPATest, CompareWithTFLiteReference) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);
  const SDPATestParams& p = GetParam();

  ODMLSDPATester tester(p.model_name);
  if (p.model_name == kOdmlSdpaCustom) {
    tester
        .QueryShape({p.batch, p.input_seq_len, p.q_heads, p.head_dim})  // q
        .KeyShape({p.batch, p.max_seq_len, p.kv_heads, p.head_dim})     // k
        .ValueShape({p.batch, p.max_seq_len, p.kv_heads, p.head_dim})   // v
        .MaskShape({p.batch, 1, p.input_seq_len, p.max_seq_len});       // mask
  }
  tester.Test(xnnpack_delegate.get());
}

INSTANTIATE_TEST_SUITE_P(SDPA, SDPATest,
                         testing::Values(SDPATestParams{kOdmlSdpaCompositeMqa},
                                         SDPATestParams{kOdmlSdpaCompositeMha},
                                         SDPATestParams{kOdmlSdpaCompositeGqa},
                                         SDPATestParams{
                                             kOdmlSdpaCustom,
                                             /*.custom_test_name=*/"MQA",
                                             /*.batch=*/1,
                                             /*.input_seq_len=*/1,
                                             /*.max_seq_len=*/64,
                                             /*.q_heads=*/32,
                                             /*.kv_heads=*/1,
                                             /*.head_dim=*/4,
                                         },
                                         SDPATestParams{
                                             kOdmlSdpaCustom,
                                             /*.custom_test_name=*/"MHA",
                                             /*.batch=*/1,
                                             /*.input_seq_len=*/1,
                                             /*.max_seq_len=*/64,
                                             /*.q_heads=*/32,
                                             /*.kv_heads=*/32,
                                             /*.head_dim=*/4,
                                         },
                                         SDPATestParams{
                                             kOdmlSdpaCustom,
                                             /*.custom_test_name=*/"GQA",
                                             /*.batch=*/1,
                                             /*.input_seq_len=*/1,
                                             /*.max_seq_len=*/64,
                                             /*.q_heads=*/32,
                                             /*.kv_heads=*/4,
                                             /*.head_dim=*/4,
                                         }),
                         TestName);

}  // namespace xnnpack
}  // namespace tflite
