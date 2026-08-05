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

#include "tensorflow/cc/saved_model/singleprint.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/core/protobuf/fingerprint.pb.h"

namespace tensorflow::saved_model::fingerprinting {
namespace {

TEST(SingleprintTest, TestSingleprintFromHashes) {
  EXPECT_EQ(Singleprint(1, 2, 3, 4), "1/2/3/4");
}

TEST(SingleprintTest, TestSingleprintFromProto) {
  FingerprintDef fingerprint_pb;
  fingerprint_pb.set_graph_def_program_hash(10);
  fingerprint_pb.set_signature_def_hash(20);
  fingerprint_pb.set_saved_object_graph_hash(30);
  fingerprint_pb.set_checkpoint_hash(40);
  EXPECT_EQ(Singleprint(fingerprint_pb), "10/20/30/40");
}

}  // namespace
}  // namespace tensorflow::saved_model::fingerprinting
