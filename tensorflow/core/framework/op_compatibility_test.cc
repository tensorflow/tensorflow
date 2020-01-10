// Test that verifies that various changes to an OpDef are
// backwards-compatible.

#include "tensorflow/core/framework/fake_input.h"
#include <gtest/gtest.h>
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/kernels/ops_testutil.h"

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

  void Run(const OpDef& old_op_def) {
    // Record the original signature before we change *node_def().
    DataTypeVector old_in_types, old_out_types;
    ASSERT_OK(InOutTypesForNode(*node_def(), old_op_def, &old_in_types,
                                &old_out_types));

    // This should be all that is needed to get compatiblity.
    const OpDef* new_op_def = RegisteredOpDef();
    AddDefaultsToNodeDef(*new_op_def, node_def());

    // Validate that it is indeed compatible.
    ASSERT_OK(ValidateNodeDef(*node_def(), *new_op_def));
    DataTypeVector new_in_types, new_out_types;
    ASSERT_OK(InOutTypesForNode(*node_def(), *new_op_def, &new_in_types,
                                &new_out_types));
    ASSERT_EQ(new_in_types, old_in_types);
    ASSERT_EQ(new_out_types, old_out_types);

    // Verify the Op actually runs.  Result() will return the output.
    ASSERT_OK(InitOp());
    ASSERT_OK(RunOpKernel());
  }

  string Result() { return GetOutput(0)->scalar<string>()(); }
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
  ASSERT_OK(NodeDefBuilder("same", "Same")
                .Input(FakeInput())
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(3))
                .Input(FakeInput(3, DT_FLOAT))
                .Input(FakeInput(2, DT_BOOL))
                .Finalize(node_def()));
  Run(*RegisteredOpDef());
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
  ASSERT_OK(
      OpDefBuilder("AddAttr").Output("ndef: string").Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("add_attr", &old_op_def).Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("add_attr = AddAttr[a=42]()", Result());
}

// Should be able to make an attr restriction less strict.
REGISTER_OP("LessStrict").Output("ndef: string").Attr("a: {'A', 'B', 'C'}");
REGISTER_KERNEL_BUILDER(Name("LessStrict").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, LessStrict) {
  OpDef old_op_def;
  ASSERT_OK(OpDefBuilder("LessStrict")
                .Output("ndef: string")
                .Attr("a: {'A', 'B'}")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("less_strict", &old_op_def)
                .Attr("a", "B")
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("less_strict = LessStrict[a=\"B\"]()", Result());
}

// Should be able to remove an attr restriction.
REGISTER_OP("RemoveRestriction").Output("ndef: string").Attr("a: type");
REGISTER_KERNEL_BUILDER(Name("RemoveRestriction").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, RemoveRestriction) {
  OpDef old_op_def;
  ASSERT_OK(OpDefBuilder("RemoveRestriction")
                .Output("ndef: string")
                .Attr("a: {int32, bool}")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("remove_restriction", &old_op_def)
                .Attr("a", DT_INT32)
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("remove_restriction = RemoveRestriction[a=DT_INT32]()", Result());
}

// Should be able to change the order of attrs.
REGISTER_OP("AttrOrder").Output("ndef: string").Attr("a: int").Attr("b: bool");
REGISTER_KERNEL_BUILDER(Name("AttrOrder").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AttrOrder) {
  OpDef old_op_def;
  ASSERT_OK(OpDefBuilder("AttrOrder")
                .Output("ndef: string")
                .Attr("b: bool")
                .Attr("a: int")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("attr_order", &old_op_def)
                .Attr("b", true)
                .Attr("a", 7)
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("attr_order = AttrOrder[a=7, b=true]()", Result());
}

// Should be able to add a default to an attr.
REGISTER_OP("AddDefault").Output("ndef: string").Attr("a: int = 1234");
REGISTER_KERNEL_BUILDER(Name("AddDefault").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddDefault) {
  OpDef old_op_def;
  ASSERT_OK(OpDefBuilder("AddDefault")
                .Output("ndef: string")
                .Attr("a: int")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("add_default", &old_op_def)
                .Attr("a", 765)
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("add_default = AddDefault[a=765]()", Result());
}

// Should be able to remove a default from an attr, *as long as that
// attr has always existed*.
REGISTER_OP("RemoveDefault").Output("ndef: string").Attr("a: int");
REGISTER_KERNEL_BUILDER(Name("RemoveDefault").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, RemoveDefault) {
  OpDef old_op_def;
  ASSERT_OK(OpDefBuilder("RemoveDefault")
                .Output("ndef: string")
                .Attr("a: int = 91")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("remove_default", &old_op_def).Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("remove_default = RemoveDefault[a=91]()", Result());
}

// Should be able to make an input polymorphic.
// Changing from int32 -> T (where T: type = DT_INT32 by default).
REGISTER_OP("TypePolymorphic")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: type = DT_INT32");
REGISTER_KERNEL_BUILDER(Name("TypePolymorphic").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, TypePolymorphic) {
  OpDef old_op_def;
  ASSERT_OK(OpDefBuilder("TypePolymorphic")
                .Input("a: int32")
                .Output("ndef: string")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("type_polymorphic", &old_op_def)
                .Input(FakeInput())
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("type_polymorphic = TypePolymorphic[T=DT_INT32](a)", Result());
}

// Should be able to make a single input into a list.
// Changing from int32 -> N * int32 (where N: int = 1 by default).
REGISTER_OP("MakeList")
    .Input("a: N * int32")
    .Output("ndef: string")
    .Attr("N: int = 1");
REGISTER_KERNEL_BUILDER(Name("MakeList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, MakeList) {
  OpDef old_op_def;
  ASSERT_OK(OpDefBuilder("MakeList")
                .Input("a: int32")
                .Output("ndef: string")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("make_list", &old_op_def)
                .Input(FakeInput())
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("make_list = MakeList[N=1](a)", Result());
}

// Should be able to make a single input into a polymorphic list.
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
  ASSERT_OK(OpDefBuilder("MakePolyList")
                .Input("a: int32")
                .Output("ndef: string")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("make_poly_list", &old_op_def)
                .Input(FakeInput())
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("make_poly_list = MakePolyList[N=1, T=DT_INT32](a)", Result());
}

// Should be able to make a single input into an arbitrary list.
// Changing from int32 -> T (where T: list(type) = [DT_INT32] by default).
REGISTER_OP("MakeAnyList")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: list(type) = [DT_INT32]");
REGISTER_KERNEL_BUILDER(Name("MakeAnyList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, MakeAnyList) {
  OpDef old_op_def;
  ASSERT_OK(OpDefBuilder("MakeAnyList")
                .Input("a: int32")
                .Output("ndef: string")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("make_any_list", &old_op_def)
                .Input(FakeInput())
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("make_any_list = MakeAnyList[T=[DT_INT32]](a)", Result());
}

// Should be able to make a single polymorphic input into a list of
// the same type.  Changing from T -> N * T (where N: int = 1 by default).
REGISTER_OP("PolyIntoList")
    .Input("a: N * T")
    .Output("ndef: string")
    .Attr("N: int = 1")
    .Attr("T: type");
REGISTER_KERNEL_BUILDER(Name("PolyIntoList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, PolyIntoList) {
  OpDef old_op_def;
  ASSERT_OK(OpDefBuilder("PolyIntoList")
                .Input("a: T")
                .Output("ndef: string")
                .Attr("T: type")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("poly_into_list", &old_op_def)
                .Input(FakeInput(DT_INT32))
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("poly_into_list = PolyIntoList[N=1, T=DT_INT32](a)", Result());
}

// Should be able to change the name of an input.
REGISTER_OP("ChangeName").Input("y: int32").Output("ndef: string");
REGISTER_KERNEL_BUILDER(Name("ChangeName").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, ChangeName) {
  OpDef old_op_def;
  ASSERT_OK(OpDefBuilder("ChangeName")
                .Input("x: int32")
                .Output("ndef: string")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("change_name", &old_op_def)
                .Input(FakeInput())
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("change_name = ChangeName[](a)", Result());
}

// Should be able to add an input of type
// N * int32 (where N: int = 0 by default).
REGISTER_OP("AddNInts")
    .Input("a: N * int32")
    .Output("ndef: string")
    .Attr("N: int >= 0 = 0");
REGISTER_KERNEL_BUILDER(Name("AddNInts").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddNInts) {
  OpDef old_op_def;
  ASSERT_OK(
      OpDefBuilder("AddNInts").Output("ndef: string").Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("add_n_ints", &old_op_def).Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("add_n_ints = AddNInts[N=0]()", Result());
}

// Should be able to add an input of type N * T
// (where N: int = 0 by default, and T: type = any valid default).
REGISTER_OP("AddNSame")
    .Input("a: N * T")
    .Output("ndef: string")
    .Attr("N: int >= 0 = 0")
    .Attr("T: type = DT_BOOL");
REGISTER_KERNEL_BUILDER(Name("AddNSame").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddNSame) {
  OpDef old_op_def;
  ASSERT_OK(
      OpDefBuilder("AddNSame").Output("ndef: string").Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("add_n_same", &old_op_def).Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("add_n_same = AddNSame[N=0, T=DT_BOOL]()", Result());
}

// Should be able to add an input of type T
// (where T: list(type) = [] by default).
REGISTER_OP("AddAnyList")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: list(type) >= 0 = []");
REGISTER_KERNEL_BUILDER(Name("AddAnyList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddAnyList) {
  OpDef old_op_def;
  ASSERT_OK(
      OpDefBuilder("AddAnyList").Output("ndef: string").Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("add_any_list", &old_op_def).Finalize(node_def()));
  Run(old_op_def);
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
  ASSERT_OK(OpDefBuilder("ShorterAnyList")
                .Input("a: T")
                .Output("ndef: string")
                .Attr("T: list(type) >= 2")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("shorter_any_list", &old_op_def)
                .Input(FakeInput(2, DT_BOOL))
                .Finalize(node_def()));
  Run(old_op_def);
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
  ASSERT_OK(OpDefBuilder("ShorterSameList")
                .Input("a: N * int32")
                .Output("ndef: string")
                .Attr("N: int >= 2")
                .Finalize(&old_op_def));
  ASSERT_OK(NodeDefBuilder("shorter_same_list", &old_op_def)
                .Input(FakeInput(2))
                .Finalize(node_def()));
  Run(old_op_def);
  EXPECT_EQ("shorter_same_list = ShorterSameList[N=2](a, a:1)", Result());
}

// TODO(josh11b): Negative tests?
// * add attr w/out default
// * add non-list input/output
// * add list input/output with non-empty default
// * change type of input/output
// * change order of input/output
// * change type of attr
// * Input("foo: T").Attr("T: type") -> Input("foo: T").Attr("T: list(type)")

// What about changing an attr's default? Not technically illegal, but
// likely should be forbidden since it likely affects semantics.

}  // namespace
}  // namespace tensorflow
