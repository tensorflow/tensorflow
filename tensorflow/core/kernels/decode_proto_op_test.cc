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

#include <string>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class DecodeProtoOpTest : public OpsTestBase {
 protected:
  void Init(const std::string& message_type, const std::string& field,
            DataType dtype) {
    TF_ASSERT_OK(NodeDefBuilder("op", "DecodeProtoV2")
                     .Input(FakeInput(DT_STRING))
                     .Attr("message_type", message_type)
                     .Attr("field_names", {field})
                     .Attr("output_types", {dtype})
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

using std::string_literals::operator""s;

TEST_F(DecodeProtoOpTest, NonRepeatedPackedFieldLastValueWins) {
  // tensorflow.TensorShapeProto has `bool unknown_rank = 3`. Tag 0x18 is that
  // field as a varint, and tag 0x1a is the same field packed.
  const std::string packed_true = "\x1a\x01\x01"s;
  const std::string packed_false = "\x1a\x01\x00"s;
  const std::string packed_empty = "\x1a\x00"s;
  const std::string unpacked_true = "\x18\x01"s;
  const std::string unpacked_false = "\x18\x00"s;
  Init("tensorflow.TensorShapeProto", "unknown_rank", DT_BOOL);
  AddInputFromArray<tstring>(
      TensorShape({5}),
      {packed_true, packed_true + packed_false, unpacked_false + packed_true,
       packed_true + unpacked_false, unpacked_true + packed_empty});
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int32_t>(
      *GetOutput(0), test::AsTensor<int32_t>({1, 1, 1, 1, 1}, {5, 1}));
  test::ExpectTensorEqual<bool>(
      *GetOutput(1),
      test::AsTensor<bool>({true, false, true, false, true}, {5, 1}));
}

TEST_F(DecodeProtoOpTest, RepeatedPackedFieldAppends) {
  // tensorflow.Int64List has `repeated int64 value = 1 [packed = true]`.
  Init("tensorflow.Int64List", "value", DT_INT64);
  // Two packed runs, then one unpacked value.
  const std::string serialized = "\x0a\x02\x01\x02\x0a\x01\x03\x08\x04"s;
  AddInputFromArray<tstring>(TensorShape({1}), {serialized});
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int32_t>(*GetOutput(0),
                                   test::AsTensor<int32_t>({4}, {1, 1}));
  test::ExpectTensorEqual<int64_t>(
      *GetOutput(1), test::AsTensor<int64_t>({1, 2, 3, 4}, {1, 4}));
}

TEST_F(DecodeProtoOpTest, RepeatedPackedFieldEmptyAtEnd) {
  Init("tensorflow.Int64List", "value", DT_INT64);
  // An empty packed run after the output is already full.
  const std::string serialized = "\x0a\x02\x01\x02\x0a\x00"s;
  AddInputFromArray<tstring>(TensorShape({1}), {serialized});
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int32_t>(*GetOutput(0),
                                   test::AsTensor<int32_t>({2}, {1, 1}));
  test::ExpectTensorEqual<int64_t>(
      *GetOutput(1), test::AsTensor<int64_t>({1, 2}, {1, 2}));
}

}  // namespace
}  // namespace tensorflow
