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

#include "tensorflow/core/framework/op_segment.h"

#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class OpSegmentTest : public ::testing::Test {
 protected:
  DeviceBase device_;
  std::vector<NodeDef> int32_nodedefs_;
  std::vector<NodeDef> float_nodedefs_;

  OpSegmentTest() : device_(Env::Default()) {
    for (int i = 0; i < 10; ++i) {
      NodeDef def;
      TF_CHECK_OK(NodeDefBuilder(strings::StrCat("op", i), "Mul")
                      .Input("x", 0, DT_INT32)
                      .Input("y", 0, DT_INT32)
                      .Finalize(&def));
      int32_nodedefs_.push_back(def);
      TF_CHECK_OK(NodeDefBuilder(strings::StrCat("op", i), "Mul")
                      .Input("x", 0, DT_FLOAT)
                      .Input("y", 0, DT_FLOAT)
                      .Finalize(&def));
      float_nodedefs_.push_back(def);
    }
  }

  void ValidateOpAndTypes(OpKernel* op, const NodeDef& expected, DataType dt) {
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(expected.DebugString(), op->def().DebugString());
    EXPECT_EQ(2, op->num_inputs());
    EXPECT_EQ(dt, op->input_type(0));
    EXPECT_EQ(dt, op->input_type(1));
    EXPECT_EQ(1, op->num_outputs());
    EXPECT_EQ(dt, op->output_type(0));
  }

  OpSegment::CreateKernelFn GetFn(const NodeDef* ndef) {
    return [this, ndef](OpKernel** kernel) {
      Status s;
      auto created = CreateOpKernel(DEVICE_CPU, &device_, cpu_allocator(),
                                    *ndef, TF_GRAPH_DEF_VERSION, &s);
      if (s.ok()) {
        *kernel = created.release();
      }
      return s;
    };
  }
};

TEST_F(OpSegmentTest, Basic) {
  OpSegment opseg;
  OpKernel* op;

  opseg.AddHold("A");
  opseg.AddHold("B");
  for (int i = 0; i < 10; ++i) {
    // Register in session A.
    auto* ndef = &float_nodedefs_[i];
    TF_EXPECT_OK(opseg.FindOrCreate("A", ndef->name(), &op, GetFn(ndef)));
    ValidateOpAndTypes(op, *ndef, DT_FLOAT);

    // Register in session B.
    ndef = &int32_nodedefs_[i];
    TF_EXPECT_OK(opseg.FindOrCreate("B", ndef->name(), &op, GetFn(ndef)));
    ValidateOpAndTypes(op, *ndef, DT_INT32);
  }

  auto reterr = [](OpKernel** kernel) {
    return errors::Internal("Should not be called");
  };
  for (int i = 0; i < 10; ++i) {
    // Lookup op in session A.
    TF_EXPECT_OK(
        opseg.FindOrCreate("A", strings::StrCat("op", i), &op, reterr));
    ValidateOpAndTypes(op, float_nodedefs_[i], DT_FLOAT);

    // Lookup op in session B.
    TF_EXPECT_OK(
        opseg.FindOrCreate("B", strings::StrCat("op", i), &op, reterr));
    ValidateOpAndTypes(op, int32_nodedefs_[i], DT_INT32);
  }

  opseg.RemoveHold("A");
  opseg.RemoveHold("B");
}

TEST_F(OpSegmentTest, SessionNotFound) {
  OpSegment opseg;
  OpKernel* op;
  NodeDef def = float_nodedefs_[0];
  Status s = opseg.FindOrCreate("A", def.name(), &op, GetFn(&def));
  EXPECT_TRUE(errors::IsNotFound(s)) << s;
}

TEST_F(OpSegmentTest, CreateFailure) {
  OpSegment opseg;
  OpKernel* op;
  NodeDef def = float_nodedefs_[0];
  def.set_op("nonexistop");
  opseg.AddHold("A");
  Status s = opseg.FindOrCreate("A", def.name(), &op, GetFn(&def));
  EXPECT_TRUE(errors::IsNotFound(s)) << s;
  opseg.RemoveHold("A");
}

TEST_F(OpSegmentTest, AddRemoveHolds) {
  OpSegment opseg;
  OpKernel* op;
  const auto& ndef = int32_nodedefs_[0];

  // No op.
  opseg.RemoveHold("null");

  // Thread1 register the op and wants to ensure it alive.
  opseg.AddHold("foo");
  TF_EXPECT_OK(opseg.FindOrCreate("foo", ndef.name(), &op, GetFn(&ndef)));

  // Thread2 starts some execution needs "op" to be alive.
  opseg.AddHold("foo");

  // Thread1 clears session "foo".  E.g., a master sends CleanupGraph
  // before an execution finishes.
  opseg.RemoveHold("foo");

  // Thread2 should still be able to access "op".
  ValidateOpAndTypes(op, ndef, DT_INT32);

  // Thread2 then remove its hold on "foo".
  opseg.RemoveHold("foo");
}

}  // namespace tensorflow
