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

#include "tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/test.h"

namespace tensorflow {
namespace {

Status ParseTextProto(absl::string_view text_proto,
                      tpu::TPUEmbeddingConfiguration* parsed_proto) {
  tsl::protobuf::TextFormat::Parser parser;
  // Attempt to parse as text.
  tsl::protobuf::io::ArrayInputStream input_stream(text_proto.data(),
                                                   text_proto.size());
  if (parser.Parse(&input_stream, parsed_proto)) {
    return absl::OkStatus();
  }
  parsed_proto->Clear();
  return errors::InvalidArgument("Could not parse text proto: ", text_proto);
}

TEST(TPUEmbeddingConfigurationProtoRewriteTest, FillFeatureDescriptor) {
  const std::string config_str = R"pb(
    table_descriptor {
      name: "T0"
      vocabulary_size: 35324928
      dimension: 128
      num_features: 3
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    table_descriptor {
      name: "T1"
      vocabulary_size: 3122176
      dimension: 128
      num_features: 2
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    mode: TRAINING
    batch_size_per_tensor_core: 256
    num_hosts: 16
    num_tensor_cores: 128
    pipeline_execution_with_tensor_core: true
  )pb";
  tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  TF_ASSERT_OK(ParseTextProto(config_str, &tpu_embedding_config));
  TF_ASSERT_OK(
      PopulateMissingFieldsInTPUEmbeddingConfig(&tpu_embedding_config));

  EXPECT_EQ(tpu_embedding_config.feature_descriptor_size(), 2);
  const auto& feature_0 = tpu_embedding_config.feature_descriptor(0);
  EXPECT_EQ(feature_0.table_id(), 0);
  EXPECT_THAT(feature_0.input_shape(), ::testing::ElementsAre(256 * 3));
  const auto& feature_1 = tpu_embedding_config.feature_descriptor(1);
  EXPECT_EQ(feature_1.table_id(), 1);
  EXPECT_THAT(feature_1.input_shape(), ::testing::ElementsAre(256 * 2));
}

TEST(TPUEmbeddingConfigurationProtoRewriteTest, FillBatchSizeAndNumFeatures) {
  const std::string config_str = R"pb(
    table_descriptor {
      name: "T0"
      vocabulary_size: 35324928
      dimension: 128
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    table_descriptor {
      name: "T1"
      vocabulary_size: 3122176
      dimension: 128
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    feature_descriptor {
      name: "F0"
      table_id: 0
      input_shape: [ 100, 5 ]
    }
    feature_descriptor {
      name: "F1"
      table_id: 1
      input_shape: [ 200, 5, 20 ]
    }
    feature_descriptor {
      name: "F2"
      table_id: 0
      input_shape: [ 50 ]
    }
    feature_descriptor {
      name: "F3"
      table_id: 0
      input_shape: [ 100, 2, 3 ]
    }
    mode: TRAINING
    num_hosts: 16
    num_tensor_cores: 128
    pipeline_execution_with_tensor_core: true
  )pb";
  tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  TF_ASSERT_OK(ParseTextProto(config_str, &tpu_embedding_config));
  TF_ASSERT_OK(
      PopulateMissingFieldsInTPUEmbeddingConfig(&tpu_embedding_config));

  EXPECT_EQ(tpu_embedding_config.batch_size_per_tensor_core(), 50);
  const auto& table_0 = tpu_embedding_config.table_descriptor(0);
  EXPECT_EQ(table_0.num_features(), 23);
  const auto& table_1 = tpu_embedding_config.table_descriptor(1);
  EXPECT_EQ(table_1.num_features(), 400);
}

TEST(TPUEmbeddingConfigurationProtoRewriteTest, InvalidBatchSizeOrNumFeatures) {
  const std::string config_str = R"pb(
    table_descriptor {
      name: "T0"
      vocabulary_size: 35324928
      dimension: 128
      num_features: 3
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    feature_descriptor {
      table_id: 0
      input_shape: [ 768 ]
    }
    mode: TRAINING
    batch_size_per_tensor_core: 256
    num_hosts: 16
    num_tensor_cores: 128
    pipeline_execution_with_tensor_core: true
  )pb";
  tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  TF_ASSERT_OK(ParseTextProto(config_str, &tpu_embedding_config));
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.clear_feature_descriptor();
    invalid_config.clear_batch_size_per_tensor_core();
    EXPECT_THAT(
        PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
        tensorflow::testing::StatusIs(
            absl::StatusCode::kInvalidArgument,
            ::testing::HasSubstr("Invalid batch_size_per_tensor_core")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.clear_feature_descriptor();
    invalid_config.mutable_table_descriptor(0)->clear_num_features();
    EXPECT_THAT(PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
                tensorflow::testing::StatusIs(
                    absl::StatusCode::kInvalidArgument,
                    ::testing::HasSubstr("Invalid num_features")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    EXPECT_THAT(
        PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
        tensorflow::testing::StatusIs(
            absl::StatusCode::kInvalidArgument,
            ::testing::HasSubstr(
                "The batch_size_per_tensor_core field must NOT be populated")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.clear_batch_size_per_tensor_core();
    EXPECT_THAT(PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
                tensorflow::testing::StatusIs(
                    absl::StatusCode::kInvalidArgument,
                    ::testing::HasSubstr("The TableDescriptor.num_features "
                                         "field must NOT be populated")));
  }
}

TEST(TPUEmbeddingConfigurationProtoRewriteTest, InvalidFeatureDescriptor) {
  const std::string config_str = R"pb(
    table_descriptor {
      name: "T0"
      vocabulary_size: 35324928
      dimension: 128
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    table_descriptor {
      name: "T1"
      vocabulary_size: 3122176
      dimension: 128
      optimization_parameters {
        adagrad {}
        learning_rate { constant: 0.1 }
      }
    }
    feature_descriptor {
      name: "F1"
      table_id: 0
      input_shape: [ 768 ]
    }
    feature_descriptor {
      name: "F2"
      table_id: 1
      input_shape: [ 512 ]
    }
    mode: TRAINING
    num_hosts: 16
    num_tensor_cores: 128
    pipeline_execution_with_tensor_core: true
  )pb";
  tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  TF_ASSERT_OK(ParseTextProto(config_str, &tpu_embedding_config));
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.mutable_feature_descriptor(0)->set_table_id(2);
    EXPECT_THAT(PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
                tensorflow::testing::StatusIs(
                    absl::StatusCode::kInvalidArgument,
                    ::testing::HasSubstr("Invalid table_id")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.mutable_feature_descriptor(0)->clear_input_shape();
    EXPECT_THAT(
        PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
        tensorflow::testing::StatusIs(
            absl::StatusCode::kInvalidArgument,
            ::testing::HasSubstr("The input_shape field cannot be empty")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.mutable_feature_descriptor(0)->set_input_shape(0, -5);
    EXPECT_THAT(
        PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
        tensorflow::testing::StatusIs(
            absl::StatusCode::kInvalidArgument,
            ::testing::HasSubstr("The input_shape dimension sizes must all")));
  }
  {
    tpu::TPUEmbeddingConfiguration invalid_config = tpu_embedding_config;
    invalid_config.mutable_feature_descriptor(1)->set_table_id(0);
    EXPECT_THAT(PopulateMissingFieldsInTPUEmbeddingConfig(&invalid_config),
                tensorflow::testing::StatusIs(
                    absl::StatusCode::kInvalidArgument,
                    ::testing::HasSubstr(
                        "No feature_descriptor fields found for table: T1")));
  }
}

}  // namespace
}  // namespace tensorflow
