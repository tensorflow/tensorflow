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

// For Google-internal use only.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/util/autotune_maps/autotune_serialize.h"

#include "absl/types/variant.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/autotune_maps/conv_autotune_maps.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {
using stream_executor::dnn::AlgorithmConfig;
using stream_executor::dnn::AlgorithmDesc;
using stream_executor::gpu::GpuDriver;
using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;

// Tests when there is no entry in the autotune maps.
TEST(AutotuneSerializeTest, Empty) {
  TF_CHECK_OK(GpuDriver::Init());
  ResetAutotuneMaps();
  std::string output;
  TF_CHECK_OK(SerializeAutotuneMaps(&output));
  TF_CHECK_OK(LoadSerializedAutotuneMaps(output));
  EXPECT_EQ(ConvAutotuneMap::GetInstance()->GetMap().size(), 0);
}

// Tests the consistency of SerializeAutotuneMaps and LoadSerializedAutotuneMaps
// by:
// 1. Insert predefined entries into the autotune maps.
// 2. Serialize it to string using SerializeAutotuneMaps.
// 3. Reset autotune maps.
// 4. Use MergeFromstring to load the entries from string to autotune maps.
// 5. Check if entries in autotune maps are equal to the predefined ones.
TEST(AutotuneSerializeTest, Consistency) {
  TF_CHECK_OK(GpuDriver::Init());
  ResetAutotuneMaps();
  ConvParameters conv_params_example_a = {
      /*batch=*/1,
      /*in_depths=*/1,
      /*in=*/{{1, 1}},
      /*data_format=*/TensorFormat::FORMAT_NCHW,
      /*out_depths=*/1,
      /*filter=*/{{1, 1}},
      /*dilation=*/{{1, 1}},
      /*stride=*/{{1, 1}},
      /*padding=*/{{1, 1}},
      /*dtype=*/DataType::DT_INT8,
      /*device_id=*/0,
      /*group_count=*/1};
  ConvParameters fused_params_example_a = {
      /*batch=*/1,
      /*in_depths=*/1,
      /*in=*/{{1, 1}},
      /*data_format=*/TensorFormat::FORMAT_NCHW,
      /*out_depths=*/1,
      /*filter=*/{{1, 1}},
      /*dilation=*/{{1, 1}},
      /*stride=*/{{1, 1}},
      /*padding=*/{{1, 1}},
      /*dtype=*/DataType::DT_INT8,
      /*device_id=*/0,
      /*group_count=*/1,
      ConvParameters::FusionInfo{1.0, 0., 0.,
                                 /*activation_mode=*/
                                 se::dnn::ActivationMode::kNone,
                                 /*is_contrib=*/false},
  };
  ConvParameters contrib_fused_params_example_a = {
      /*batch=*/1,
      /*in_depths=*/1,
      /*in=*/{{1, 1}},
      /*data_format=*/TensorFormat::FORMAT_NCHW,
      /*out_depths=*/1,
      /*filter=*/{{1, 1}},
      /*dilation=*/{{1, 1}},
      /*stride=*/{{1, 1}},
      /*padding=*/{{1, 1}},
      /*dtype=*/DataType::DT_INT8,
      /*device_id=*/0,
      /*group_count=*/1,
      ConvParameters::FusionInfo{1.0, 0., 0.,
                                 /*activation_mode=*/
                                 se::dnn::ActivationMode::kRelu,
                                 /*is_contrib=*/true}};

  AlgorithmDesc algorithm(/*algo_id=*/1, /*use_tensor_ops=*/true);
  AlgorithmDesc algorithm_no_scratch(/*algo_id=*/1, /*use_tensor_ops=*/true);
  AutotuneEntry<se::dnn::ConvOp> example_a(algorithm, algorithm_no_scratch);
  ConvAutotuneMap::GetInstance()->Insert(conv_params_example_a, example_a);
  ConvAutotuneMap::GetInstance()->Insert(fused_params_example_a, example_a);
  ConvAutotuneMap::GetInstance()->Insert(contrib_fused_params_example_a,
                                         example_a);
  std::string serialized_string;
  TF_CHECK_OK(SerializeAutotuneMaps(&serialized_string));
  ResetAutotuneMaps();
  TF_CHECK_OK(LoadSerializedAutotuneMaps(serialized_string));
  EXPECT_EQ(ConvAutotuneMap::GetInstance()->GetMap().size(), 3);

  AutotuneEntry<se::dnn::ConvOp> entry;
  EXPECT_TRUE(
      ConvAutotuneMap::GetInstance()->Find(conv_params_example_a, &entry));
  EXPECT_EQ(entry, example_a);
  EXPECT_TRUE(
      ConvAutotuneMap::GetInstance()->Find(fused_params_example_a, &entry));
  EXPECT_EQ(entry, example_a);
  EXPECT_TRUE(ConvAutotuneMap::GetInstance()->Find(
      contrib_fused_params_example_a, &entry));
  EXPECT_EQ(entry, example_a);
}

// Test that LoadSerializedAutotuneMaps will reject entries with incompatible
// version.
TEST(AutotuneSerializeTest, VersionControl) {
  TF_CHECK_OK(GpuDriver::Init());
  ResetAutotuneMaps();

  ConvParameters fused_params_example_a = {
      /*batch=*/1,
      /*in_depths=*/1,
      /*in=*/{{1, 1}},
      /*data_format=*/TensorFormat::FORMAT_NCHW,
      /*out_depths=*/1,
      /*filter=*/{{1, 1}},
      /*dilation=*/{{1, 1}},
      /*stride=*/{{1, 1}},
      /*padding=*/{{1, 1}},
      /*dtype=*/DataType::DT_INT8,
      /*device_id=*/0,
      /*group_count=*/1,
      ConvParameters::FusionInfo{1.0, 0., 0.,
                                 /*activation_mode=*/
                                 se::dnn::ActivationMode::kNone,
                                 /*is_contrib=*/false},
      /*version=*/ConvParameters::kVersion - 1};

  AlgorithmDesc algorithm(/*algo_id=*/1, /*use_tensor_ops=*/true);
  AlgorithmDesc algorithm_no_scratch(/*algo_id=*/1, /*use_tensor_ops=*/true);
  AlgorithmConfig algorithm_config_example_a(algorithm, /*scratch_size=*/1,
                                             algorithm_no_scratch);

  ConvAutotuneMap::GetInstance()->Insert(
      fused_params_example_a,
      AutotuneEntry<se::dnn::ConvOp>(algorithm_config_example_a));

  std::string serialized_string;
  TF_CHECK_OK(SerializeAutotuneMaps(&serialized_string));

  ResetAutotuneMaps();
  EXPECT_THAT(
      LoadSerializedAutotuneMaps(serialized_string),
      StatusIs(error::ABORTED,
               HasSubstr("Aborted because the loaded autotune results")));
  EXPECT_EQ(ConvAutotuneMap::GetInstance()->GetMap().size(), 0);
}
}  // namespace
}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
