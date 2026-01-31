/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "tensorflow/cc/saved_model/fingerprinting.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tsl/platform/statusor.h"

namespace tensorflow::saved_model::fingerprinting {

namespace {

TEST(FingerprintingTest, TestChunkedProto) {
  const std::string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "chunked_saved_model/chunked_model");
  TF_ASSERT_OK_AND_ASSIGN(FingerprintDef fingerprint_pb,
                          CreateFingerprintDef(export_dir));
  EXPECT_GT(fingerprint_pb.saved_model_checksum(), 0);
  // We test for multiple fingerprints due to non-determinism when building with
  // different compilation_mode flag options.
  EXPECT_THAT(absl::flat_hash_set<uint64_t>(
                  {906548630859202535U, 9562420523583756263U}),
              ::testing::Contains(fingerprint_pb.graph_def_program_hash()));
  EXPECT_EQ(fingerprint_pb.signature_def_hash(), 1043582354059066488U);
  EXPECT_THAT(absl::flat_hash_set<uint64_t>(
                  {2766043449526180728U, 11894619660760763927U}),
              ::testing::Contains(fingerprint_pb.saved_object_graph_hash()));
  EXPECT_EQ(fingerprint_pb.checkpoint_hash(), 0);
}

}  // namespace
}  // namespace tensorflow::saved_model::fingerprinting
