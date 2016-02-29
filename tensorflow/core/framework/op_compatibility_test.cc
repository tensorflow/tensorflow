/* Copyright 2015 Google Inc. All Rights Reserved.

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

// Test that verifies that various changes to an OpDef are
// backwards-compatible.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class TestKernel : public OpKernel {
 public:
  explicit TestKernel(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("ndef", TensorShape({}),
                                                     &out_tensor));
    out_tensor->scalar<string>()() = SummarizeNodeDef(def());
  }
};

class OpCompatibilityTest : public OpsTestBase {
 protected:
  const OpDef* RegisteredOpDef() {
    Status status;
    const OpDef* new_op_def =
        OpRegistry::Global()->LookUp(node_def()->op(), &status);
    TF_CHECK_OK(status);
    return new_op_def;
  }

  void ExpectSuccess(const OpDef& old_op_def) {
    // Record the original signature before we change *node_def().
    DataTypeVector old_in_types, old_out_types;
    TF_ASSERT_OK(InOutTypesForNode(*node_def(), old_op_def, &old_in_types,
                                   &old_out_types));

    // This should be all that is needed to get compatibility.
    const OpDef* new_op_def = RegisteredOpDef();
    AddDefaultsToNodeDef(*new_op_def, node_def());

    // Validate that it is indeed compatible.
    TF_ASSERT_OK(ValidateNodeDef(*node_def(), *new_op_def));
    DataTypeVector new_in_types, new_out_types;
    TF_ASSERT_OK(InOutTypesForNode(*node_def(), *new_op_def, &new_in_types,
                                   &new_out_types));
    ASSERT_EQ(new_in_types, old_in_types);
    ASSERT_EQ(new_out_types, old_out_types);
    TF_ASSERT_OK(OpDefCompatible(old_op_def, *new_op_def));

    // Verify the Op actually runs.  Result() will return the output.
    TF_ASSERT_OK(InitOp());
    TF_ASSERT_OK(RunOpKernel());
  }

  string Result() { return GetOutput(0)->scalar<string>()(); }

  void ExpectIncompatible(const OpDef& old_op_def, const OpDef& new_op_def,
                          const string& error) {
    // Test OpDefCompatible gives the same answer without the node_def.
    Status status = OpDefCompatible(old_op_def, new_op_def);
    if (status.ok()) {
      ADD_FAILURE() << SummarizeOpDef(old_op_def) << " vs. "
                    << SummarizeOpDef(new_op_def);
    } else {
      EXPECT_TRUE(StringPiece(status.error_message()).contains(error))
          << status << " does not contain " << error;
    }
  }

  void ExpectInvalid(const OpDef& old_op_def, const string& validation_error,
                     const string& compatibility_error) {
    // Record the original signature before we change *node_def().
    DataTypeVector old_in_types, old_out_types;
    TF_ASSERT_OK(InOutTypesForNode(*node_def(), old_op_def, &old_in_types,
                                   &old_out_types));

    // This should be all that is needed to get compatibility.
    const OpDef* new_op_def = RegisteredOpDef();
    AddDefaultsToNodeDef(*new_op_def, node_def());

    // Validate that it does not pass validation.
    Status status = ValidateNodeDef(*node_def(), *new_op_def);
    if (status.ok()) {
      ADD_FAILURE() << SummarizeNodeDef(*node_def());
    } else {
      EXPECT_TRUE(
          StringPiece(status.error_message()).contains(validation_error))
          << status << " does not contain " << validation_error;
    }

    ExpectIncompatible(old_op_def, *new_op_def, compatibility_error);
  }

  void ExpectTypeMismatch(const OpDef& old_op_def,
                          const string& compatibility_error) {
    // Record the original signature before we change *node_def().
    DataTypeVector old_in_types, old_out_types;
    TF_ASSERT_OK(InOutTypesForNode(*node_def(), old_op_def, &old_in_types,
                                   &old_out_types));

    // This should be all that is needed to get compatibility.
    const OpDef* new_op_def = RegisteredOpDef();
    AddDefaultsToNodeDef(*new_op_def, node_def());

    // Validate that it is valid, but with incompatible types.
    TF_ASSERT_OK(ValidateNodeDef(*node_def(), *new_op_def));

    DataTypeVector new_in_types, new_out_types;
    TF_ASSERT_OK(InOutTypesForNode(*node_def(), *new_op_def, &new_in_types,
                                   &new_out_types));
    if (new_in_types == old_in_types && new_out_types == old_out_types) {
      ADD_FAILURE() << SummarizeNodeDef(*node_def()) << "\n"
                    << DataTypeVectorString(new_in_types) << " -> "
                    << DataTypeVectorString(new_out_types);
    }

    ExpectIncompatible(old_op_def, *new_op_def, compatibility_error);
  }
};

// Should be compatible if the Op hasn't changed (sanity check).
REGISTER_OP("Same")
    .Input("a: int32")
    .Input("b: T")
    .Input("c: N * int32")
    .Input("d: N * T")
    .Input("e: TList")
    .Output("ndef: string")
    .Attr("T: type")
    .Attr("N: int")
    .Attr("TList: list(type)");
REGISTER_KERNEL_BUILDER(Name("Same").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, Same) {
  TF_ASSERT_OK(NodeDefBuilder("same", "Same")
                   .Input(FakeInput())
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(3))
                   .Input(FakeInput(3, DT_FLOAT))
                   .Input(FakeInput(2, DT_BOOL))
                   .Finalize(node_def()));
  ExpectSuccess(*RegisteredOpDef());
  EXPECT_EQ(
      "same = Same[N=3, T=DT_FLOAT, TList=[DT_BOOL, DT_BOOL]](a, b, c, c:1, "
      "c:2, d, d:1, d:2, e, e:1)",
      Result());
}

// Should be able to add an attr with a default.
REGISTER_OP("AddAttr").Output("ndef: string").Attr("a: int = 42");
REGISTER_KERNEL_BUILDER(Name("AddAttr").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddAttr) {
  OpDef old_op_def;
  TF_ASSERT_OK(
      OpDefBuilder("AddAttr").Output("ndef: string").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("add_attr", &old_op_def).Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("add_attr = AddAttr[a=42]()", Result());
}

// Should be able to make an attr restriction less strict.
REGISTER_OP("LessStrict").Output("ndef: string").Attr("a: {'A', 'B', 'C'}");
REGISTER_KERNEL_BUILDER(Name("LessStrict").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, LessStrict) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("LessStrict")
                   .Output("ndef: string")
                   .Attr("a: {'A', 'B'}")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("less_strict", &old_op_def)
                   .Attr("a", "B")
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("less_strict = LessStrict[a=\"B\"]()", Result());
}

// Should be able to remove an attr restriction.
REGISTER_OP("RemoveRestriction").Output("ndef: string").Attr("a: type");
REGISTER_KERNEL_BUILDER(Name("RemoveRestriction").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, RemoveRestriction) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("RemoveRestriction")
                   .Output("ndef: string")
                   .Attr("a: {int32, bool}")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("remove_restriction", &old_op_def)
                   .Attr("a", DT_INT32)
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("remove_restriction = RemoveRestriction[a=DT_INT32]()", Result());
}

// Should be able to change the order of attrs.
REGISTER_OP("AttrOrder").Output("ndef: string").Attr("a: int").Attr("b: bool");
REGISTER_KERNEL_BUILDER(Name("AttrOrder").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AttrOrder) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("AttrOrder")
                   .Output("ndef: string")
                   .Attr("b: bool")
                   .Attr("a: int")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("attr_order", &old_op_def)
                   .Attr("b", true)
                   .Attr("a", 7)
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("attr_order = AttrOrder[a=7, b=true]()", Result());
}

// Should be able to add a default to an attr.
REGISTER_OP("AddDefault").Output("ndef: string").Attr("a: int = 1234");
REGISTER_KERNEL_BUILDER(Name("AddDefault").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddDefault) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("AddDefault")
                   .Output("ndef: string")
                   .Attr("a: int")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("add_default", &old_op_def)
                   .Attr("a", 765)
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("add_default = AddDefault[a=765]()", Result());
}

// Should be able to remove a default from an attr, *as long as that
// attr has always existed*.
REGISTER_OP("RemoveDefault").Output("ndef: string").Attr("a: int");
REGISTER_KERNEL_BUILDER(Name("RemoveDefault").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, RemoveDefault) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("RemoveDefault")
                   .Output("ndef: string")
                   .Attr("a: int = 91")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(
      NodeDefBuilder("remove_default", &old_op_def).Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("remove_default = RemoveDefault[a=91]()", Result());
}

// Should be able to make an input/output polymorphic.
// Changing from int32 -> T (where T: type = DT_INT32 by default).
REGISTER_OP("TypePolymorphic")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: type = DT_INT32");
REGISTER_KERNEL_BUILDER(Name("TypePolymorphic").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, TypePolymorphic) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("TypePolymorphic")
                   .Input("a: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("type_polymorphic", &old_op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("type_polymorphic = TypePolymorphic[T=DT_INT32](a)", Result());
}

// Should be able to make a single input/output into a list.
// Changing from int32 -> N * int32 (where N: int = 1 by default).
REGISTER_OP("MakeList")
    .Input("a: N * int32")
    .Output("ndef: string")
    .Attr("N: int = 1");
REGISTER_KERNEL_BUILDER(Name("MakeList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, MakeList) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("MakeList")
                   .Input("a: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("make_list", &old_op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("make_list = MakeList[N=1](a)", Result());
}

// Should be able to make a single input/output into a polymorphic list.
// Changing from int32 -> N * T (where N: int = 1 by default and
//                                     T: type = DT_INT32 by default).
REGISTER_OP("MakePolyList")
    .Input("a: N * T")
    .Output("ndef: string")
    .Attr("N: int = 1")
    .Attr("T: type = DT_INT32");
REGISTER_KERNEL_BUILDER(Name("MakePolyList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, MakePolyList) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("MakePolyList")
                   .Input("a: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("make_poly_list", &old_op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("make_poly_list = MakePolyList[N=1, T=DT_INT32](a)", Result());
}

// Should be able to make a single input/output into an arbitrary list.
// Changing from int32 -> T (where T: list(type) = [DT_INT32] by default).
REGISTER_OP("MakeAnyList")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: list(type) = [DT_INT32]");
REGISTER_KERNEL_BUILDER(Name("MakeAnyList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, MakeAnyList) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("MakeAnyList")
                   .Input("a: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("make_any_list", &old_op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("make_any_list = MakeAnyList[T=[DT_INT32]](a)", Result());
}

// Should be able to make a single polymorphic input/output into a list of
// the same type.  Changing from T -> N * T (where N: int = 1 by default).
REGISTER_OP("PolyIntoList")
    .Input("a: N * T")
    .Output("ndef: string")
    .Attr("N: int = 1")
    .Attr("T: type");
REGISTER_KERNEL_BUILDER(Name("PolyIntoList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, PolyIntoList) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("PolyIntoList")
                   .Input("a: T")
                   .Output("ndef: string")
                   .Attr("T: type")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("poly_into_list", &old_op_def)
                   .Input(FakeInput(DT_INT32))
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("poly_into_list = PolyIntoList[N=1, T=DT_INT32](a)", Result());
}

// Should be able to make a multiple inputs/outputs into a list with
// the default types matching the inputs/outputs being replaced.

// Changing from int32, int32 -> N * int32 (where N: int = 2 by default).
REGISTER_OP("MakeMultipleSameList")
    .Input("a: N * int32")
    .Output("ndef: string")
    .Attr("N: int = 2");
REGISTER_KERNEL_BUILDER(Name("MakeMultipleSameList").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, MakeMultipleSameList) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("MakeMultipleSameList")
                   .Input("a: int32")
                   .Input("b: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("make_list", &old_op_def)
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("make_list = MakeMultipleSameList[N=2](a, b)", Result());
}

// Changing from int32, float -> T
// (where T: list(type) = [int32, float] by default).
REGISTER_OP("MakeMultipleAnyList")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: list(type) = [DT_INT32, DT_FLOAT]");
REGISTER_KERNEL_BUILDER(Name("MakeMultipleAnyList").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, MakeMultipleAnyList) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("MakeMultipleAnyList")
                   .Input("a: int32")
                   .Input("b: float")
                   .Output("ndef: string")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("make_list", &old_op_def)
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("make_list = MakeMultipleAnyList[T=[DT_INT32, DT_FLOAT]](a, b)",
            Result());
}

// Should be able to change the name of an input/output.
REGISTER_OP("ChangeName").Input("y: int32").Output("ndef: string");
REGISTER_KERNEL_BUILDER(Name("ChangeName").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, ChangeName) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("ChangeName")
                   .Input("x: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("change_name", &old_op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("change_name = ChangeName[](a)", Result());
}

// Should be able to add an input/output of type
// N * int32 (where N: int = 0 by default).
REGISTER_OP("AddNInts")
    .Input("a: N * int32")
    .Output("ndef: string")
    .Attr("N: int >= 0 = 0");
REGISTER_KERNEL_BUILDER(Name("AddNInts").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddNInts) {
  OpDef old_op_def;
  TF_ASSERT_OK(
      OpDefBuilder("AddNInts").Output("ndef: string").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("add_n_ints", &old_op_def).Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("add_n_ints = AddNInts[N=0]()", Result());
}

// Should be able to add an input/output of type N * T
// (where N: int = 0 by default, and T: type = any valid default).
REGISTER_OP("AddNSame")
    .Input("a: N * T")
    .Output("ndef: string")
    .Attr("N: int >= 0 = 0")
    .Attr("T: type = DT_BOOL");
REGISTER_KERNEL_BUILDER(Name("AddNSame").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddNSame) {
  OpDef old_op_def;
  TF_ASSERT_OK(
      OpDefBuilder("AddNSame").Output("ndef: string").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("add_n_same", &old_op_def).Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("add_n_same = AddNSame[N=0, T=DT_BOOL]()", Result());
}

// Should be able to add an input/output of type N * T
// (where N: int >= 0 = 0 by default, and T an existing type attr).
REGISTER_OP("AddNSameAsExisting")
    .Input("a: T")
    .Input("b: N * T")
    .Output("ndef: string")
    .Attr("N: int >= 0 = 0")
    .Attr("T: type");
REGISTER_KERNEL_BUILDER(Name("AddNSameAsExisting").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, AddNSameAsExisting) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("AddNSameAsExisting")
                   .Input("a: T")
                   .Output("ndef: string")
                   .Attr("T: type")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("add_n_same_as_existing", &old_op_def)
                   .Input(FakeInput(DT_STRING))
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("add_n_same_as_existing = AddNSameAsExisting[N=0, T=DT_STRING](a)",
            Result());
}

// Should be able to add an input/output of type T
// (where T: list(type) >= 0 = [] by default).
REGISTER_OP("AddAnyList")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: list(type) >= 0 = []");
REGISTER_KERNEL_BUILDER(Name("AddAnyList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddAnyList) {
  OpDef old_op_def;
  TF_ASSERT_OK(
      OpDefBuilder("AddAnyList").Output("ndef: string").Finalize(&old_op_def));
  TF_ASSERT_OK(
      NodeDefBuilder("add_any_list", &old_op_def).Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("add_any_list = AddAnyList[T=[]]()", Result());
}

// Should be able to allow shorter lists.
REGISTER_OP("ShorterAnyList")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: list(type) >= 1");
REGISTER_KERNEL_BUILDER(Name("ShorterAnyList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, ShorterAnyList) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("ShorterAnyList")
                   .Input("a: T")
                   .Output("ndef: string")
                   .Attr("T: list(type) >= 2")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("shorter_any_list", &old_op_def)
                   .Input(FakeInput(2, DT_BOOL))
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("shorter_any_list = ShorterAnyList[T=[DT_BOOL, DT_BOOL]](a, a:1)",
            Result());
}

REGISTER_OP("ShorterSameList")
    .Input("a: N * int32")
    .Output("ndef: string")
    .Attr("N: int >= 1");
REGISTER_KERNEL_BUILDER(Name("ShorterSameList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, ShorterSameList) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("ShorterSameList")
                   .Input("a: N * int32")
                   .Output("ndef: string")
                   .Attr("N: int >= 2")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("shorter_same_list", &old_op_def)
                   .Input(FakeInput(2))
                   .Finalize(node_def()));
  ExpectSuccess(old_op_def);
  EXPECT_EQ("shorter_same_list = ShorterSameList[N=2](a, a:1)", Result());
}

// Negative tests -------------------------------------------------------------

// Can't remove an attr.
REGISTER_OP("RemoveAttr");

TEST_F(OpCompatibilityTest, RemoveAttrFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("RemoveAttr").Attr("a: int").Finalize(&old_op_def));
  TF_ASSERT_OK(
      NodeDefBuilder("fails", &old_op_def).Attr("a", 3).Finalize(node_def()));
  ExpectInvalid(old_op_def, "NodeDef mentions attr 'a' not in",
                "Attr 'a' removed");
}

// Can't add an attr without a default.
REGISTER_OP("AddAttrNoDefault").Attr("a: int");

TEST_F(OpCompatibilityTest, AddAttrNoDefaultFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("AddAttrNoDefault").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def).Finalize(node_def()));
  ExpectInvalid(old_op_def, "NodeDef missing attr 'a'",
                "Attr 'a' added without default");
}

// Can't add a non-list input/output.
REGISTER_OP("AddSingleInput").Input("a: int32");

TEST_F(OpCompatibilityTest, AddSingleInputFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("AddSingleInput").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def).Finalize(node_def()));
  ExpectInvalid(old_op_def,
                "expected inputs 'int32' do not match 0 inputs specified",
                "Input signature mismatch '' vs. 'int32'");
}

// Can't add a list input/output without an empty default.

REGISTER_OP("AddNIntsBigDefault").Input("a: N * int32").Attr("N: int = 1");
REGISTER_OP("AddNSameBigDefault")
    .Input("a: N * T")
    .Attr("N: int = 1")
    .Attr("T: type = DT_INT32");
REGISTER_OP("AddListBigDefault")
    .Input("a: T")
    .Attr("T: list(type) = [DT_INT32]");

TEST_F(OpCompatibilityTest, AddNIntsBigDefaultFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("AddNIntsBigDefault").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def).Finalize(node_def()));
  ExpectInvalid(old_op_def,
                "expected inputs 'int32' do not match 0 inputs specified",
                "Input signature mismatch '' vs. 'int32'");
}

TEST_F(OpCompatibilityTest, AddNSameBigDefaultFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("AddNSameBigDefault").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def).Finalize(node_def()));
  ExpectInvalid(old_op_def,
                "expected inputs 'int32' do not match 0 inputs specified",
                "Input signature mismatch '' vs. 'int32'");
}

TEST_F(OpCompatibilityTest, AddListBigDefaultFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("AddListBigDefault").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def).Finalize(node_def()));
  ExpectInvalid(old_op_def,
                "expected inputs 'int32' do not match 0 inputs specified",
                "Input signature mismatch '' vs. 'int32'");
}

// Can't change the type of an input/output.

REGISTER_OP("ChangeType").Input("a: float");

TEST_F(OpCompatibilityTest, ChangeTypeFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(
      OpDefBuilder("ChangeType").Input("a: int32").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectTypeMismatch(old_op_def,
                     "Input signature mismatch 'int32' vs. 'float'");
}

// Can't change the order of inputs/outputs.

REGISTER_OP("ChangeOrder").Input("a: float").Input("b: int32");

TEST_F(OpCompatibilityTest, ChangeOrderFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("ChangeOrder")
                   .Input("b: int32")
                   .Input("a: float")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def)
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectTypeMismatch(
      old_op_def, "Input signature mismatch 'int32, float' vs. 'float, int32'");
}

// Can't remove inputs/outputs.

REGISTER_OP("RemoveInput");

TEST_F(OpCompatibilityTest, RemoveInputFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(
      OpDefBuilder("RemoveInput").Input("a: float").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectInvalid(old_op_def,
                "expected inputs '' do not match 1 inputs specified",
                "Input signature mismatch 'float' vs. ''");
}

// Can't change the type of an attr.

REGISTER_OP("ChangeAttrType").Attr("a: int");

TEST_F(OpCompatibilityTest, ChangeAttrTypeFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(
      OpDefBuilder("ChangeAttrType").Attr("a: bool").Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def)
                   .Attr("a", true)
                   .Finalize(node_def()));
  ExpectInvalid(old_op_def, "value with type 'bool' when 'int' expected",
                "Attr 'a' changed type 'bool' -> 'int'");
}

// Can't change an attr from a list.

REGISTER_OP("AttrFromList").Attr("a: int");

TEST_F(OpCompatibilityTest, AttrFromListFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(
      OpDefBuilder("AttrFromList").Attr("a: list(int)").Finalize(&old_op_def));
  TF_ASSERT_OK(
      NodeDefBuilder("fails", &old_op_def).Attr("a", {5}).Finalize(node_def()));
  ExpectInvalid(old_op_def, "value with type 'list(int)' when 'int' expected",
                "Attr 'a' changed type 'list(int)' -> 'int'");
}

// Can't change an attr to a list.

REGISTER_OP("AttrToList").Attr("a: list(int)");

TEST_F(OpCompatibilityTest, AttrToListFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("AttrToList").Attr("a: int").Finalize(&old_op_def));
  TF_ASSERT_OK(
      NodeDefBuilder("fails", &old_op_def).Attr("a", 5).Finalize(node_def()));
  ExpectInvalid(old_op_def, "value with type 'int' when 'list(int)' expected",
                "Attr 'a' changed type 'int' -> 'list(int)'");
}

// Can't change an input from polymorphic to a list of any type.

REGISTER_OP("PolymorphicToAnyList").Input("a: T").Attr("T: list(type)");

TEST_F(OpCompatibilityTest, PolymorphicToAnyListFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("PolymorphicToAnyList")
                   .Input("a: T")
                   .Attr("T: type")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def)
                   .Input(FakeInput(DT_INT32))
                   .Finalize(node_def()));
  ExpectInvalid(old_op_def, "value with type 'type' when 'list(type)' expected",
                "Attr 'T' changed type 'type' -> 'list(type)'");
}

// Can't change an input from a list of the same type to a list of any type.

REGISTER_OP("SameToAnyList")
    .Input("a: T")
    .Attr("T: list(type)")
    .Attr("N: int = 1");

TEST_F(OpCompatibilityTest, SameToAnyListFails) {
  OpDef old_op_def;
  TF_ASSERT_OK(OpDefBuilder("SameToAnyList")
                   .Input("a: N * T")
                   .Attr("T: type")
                   .Attr("N: int")
                   .Finalize(&old_op_def));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op_def)
                   .Input(FakeInput(1, DT_INT32))
                   .Finalize(node_def()));
  ExpectInvalid(old_op_def, "value with type 'type' when 'list(type)' expected",
                "Attr 'T' changed type 'type' -> 'list(type)'");
}

// Changing an attr's default is not technically illegal, but should
// be forbidden if it the attr ever didn't exist since it likely
// affects semantics.

}  // namespace
}  // namespace tensorflow
