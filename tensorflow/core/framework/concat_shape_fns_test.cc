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
#include <gtest/gtest.h>

#include <cstdint>

#include "absl/status/status.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace shape_inference {
namespace {

OpDef MakeConcatOpDef() {
  OpDef op_def;
  for (int i = 0; i < 3; ++i) {
    OpDef::ArgDef* input = op_def.add_input_arg();
    input->set_name("input");
    input->set_type(DT_FLOAT);
  }
  OpDef::ArgDef* output = op_def.add_output_arg();
  output->set_name("output");
  output->set_type(DT_FLOAT);
  return op_def;
}

TEST(ConcatShapeFnsTest, ValidAxisDtypesAreAccepted) {
  NodeDef def;
  const OpDef op_def = MakeConcatOpDef();
  {
    const Tensor axis = test::AsScalar<int32_t>(1);
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {PartialTensorShape({2, 3}), PartialTensorShape({2, 4}),
                        PartialTensorShape({})},
                       {nullptr, nullptr, &axis}, {}, {});

    const absl::Status status = ConcatV2Shape(&c);
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_EQ("[2,7]", c.DebugString(c.output(0)));
  }
  {
    const Tensor axis = test::AsScalar<int64_t>(1);
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {PartialTensorShape({2, 3}), PartialTensorShape({2, 4}),
                        PartialTensorShape({})},
                       {nullptr, nullptr, &axis}, {}, {});

    const absl::Status status = ConcatV2Shape(&c);
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_EQ("[2,7]", c.DebugString(c.output(0)));
  }
}

template <typename T>
void ExpectInvalidAxisDtype() {
  NodeDef def;
  const Tensor axis = test::AsScalar<T>(static_cast<T>(0));
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeConcatOpDef(),
                     {PartialTensorShape({2, 3}), PartialTensorShape({2, 4}),
                      PartialTensorShape({})},
                     {nullptr, nullptr, &axis}, {}, {});

  EXPECT_THAT(
      ConcatV2Shape(&c),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             "concat_dim must be an int32 or int64 scalar"))
      << DataTypeString(DataTypeToEnum<T>::v());
}

TEST(ConcatShapeFnsTest, InvalidAxisDtypeReturnsError) {
  ExpectInvalidAxisDtype<bool>();
  ExpectInvalidAxisDtype<float>();
  ExpectInvalidAxisDtype<double>();
  ExpectInvalidAxisDtype<int8_t>();
  ExpectInvalidAxisDtype<uint8_t>();
  ExpectInvalidAxisDtype<int16_t>();
  ExpectInvalidAxisDtype<uint16_t>();
  ExpectInvalidAxisDtype<uint32_t>();
  ExpectInvalidAxisDtype<uint64_t>();
}

TEST(ConcatShapeFnsTest, InvalidAxisShapeReturnsError) {
  NodeDef def;
  const OpDef op_def = MakeConcatOpDef();
  {
    const Tensor axis(DT_INT32, TensorShape({0}));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {PartialTensorShape({2, 3}), PartialTensorShape({2, 4}),
                        PartialTensorShape({})},
                       {nullptr, nullptr, &axis}, {}, {});

    EXPECT_THAT(ConcatV2Shape(&c),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       "concat_dim must be a scalar"));
  }
  {
    const Tensor axis(DT_INT32, TensorShape({2}));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {PartialTensorShape({2, 3}), PartialTensorShape({2, 4}),
                        PartialTensorShape({})},
                       {nullptr, nullptr, &axis}, {}, {});

    EXPECT_THAT(ConcatV2Shape(&c),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       "concat_dim must be a scalar"));
  }
  {
    const Tensor axis(DT_INT64, TensorShape({0}));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {PartialTensorShape({2, 3}), PartialTensorShape({2, 4}),
                        PartialTensorShape({})},
                       {nullptr, nullptr, &axis}, {}, {});

    EXPECT_THAT(ConcatV2Shape(&c),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       "concat_dim must be a scalar"));
  }
  {
    const Tensor axis(DT_INT64, TensorShape({2}));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {PartialTensorShape({2, 3}), PartialTensorShape({2, 4}),
                        PartialTensorShape({})},
                       {nullptr, nullptr, &axis}, {}, {});

    EXPECT_THAT(ConcatV2Shape(&c),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       "concat_dim must be a scalar"));
  }
}

}  // namespace
}  // namespace shape_inference
}  // namespace tensorflow
