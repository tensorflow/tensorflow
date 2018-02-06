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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

REGISTER_OP("KernelLabel")
    .Output("result: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("KernelLabelRequired")
    .Input("input: int32")
    .Output("result: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &out));
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("GraphDefVersion")
    .Output("version: int32")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("RequiresOlderGraphVersion")
    .Output("version: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (c->graph_def_version() != TF_GRAPH_DEF_VERSION - 1) {
        return errors::InvalidArgument("Wrong graph version for shape");
      }
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("Old")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(8, "For reasons");

REGISTER_RESOURCE_HANDLE_OP(StubResource);

REGISTER_OP("ResourceInitializedOp")
    .Input("resource: resource")
    .Output("initialized: bool")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ResourceCreateOp")
    .Input("resource: resource")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ResourceUsingOp")
    .Input("resource: resource")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TestStringOutput")
    .Input("input: float")
    .Output("output1: float")
    .Output("output2: string")
    .SetShapeFn(shape_inference::UnknownShape);

namespace {
enum KernelLabel { DEFAULT_LABEL, OVERLOAD_1_LABEL, OVERLOAD_2_LABEL };
}  // namespace

template <KernelLabel KL>
class KernelLabelOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("result", TensorShape({}), &output));
    switch (KL) {
      case DEFAULT_LABEL:
        output->scalar<string>()() = "My label is: default";
        break;
      case OVERLOAD_1_LABEL:
        output->scalar<string>()() = "My label is: overload_1";
        break;
      case OVERLOAD_2_LABEL:
        output->scalar<string>()() = "My label is: overload_2";
        break;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KernelLabel").Device(DEVICE_CPU),
                        KernelLabelOp<DEFAULT_LABEL>);
REGISTER_KERNEL_BUILDER(Name("KernelLabel")
                            .Device(DEVICE_CPU)
                            .Label("overload_1"),
                        KernelLabelOp<OVERLOAD_1_LABEL>);
REGISTER_KERNEL_BUILDER(Name("KernelLabel")
                            .Device(DEVICE_CPU)
                            .Label("overload_2"),
                        KernelLabelOp<OVERLOAD_2_LABEL>);

// All "KernelLabelRequired" kernels have labels
REGISTER_KERNEL_BUILDER(
    Name("KernelLabelRequired").Device(DEVICE_CPU).Label("overload_1"),
    KernelLabelOp<OVERLOAD_1_LABEL>);
REGISTER_KERNEL_BUILDER(
    Name("KernelLabelRequired").Device(DEVICE_CPU).Label("overload_2"),
    KernelLabelOp<OVERLOAD_2_LABEL>);

class GraphDefVersionOp : public OpKernel {
 public:
  explicit GraphDefVersionOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    output->scalar<int>()() = graph_def_version_;
  }

 private:
  const int graph_def_version_;
};

REGISTER_KERNEL_BUILDER(Name("GraphDefVersion").Device(DEVICE_CPU),
                        GraphDefVersionOp);

class OldOp : public OpKernel {
 public:
  explicit OldOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {}
};

REGISTER_KERNEL_BUILDER(Name("Old").Device(DEVICE_CPU), OldOp);

// Stubbed-out resource to test resource handle ops.
class StubResource : public ResourceBase {
 public:
  string DebugString() override { return ""; }
};

REGISTER_RESOURCE_HANDLE_KERNEL(StubResource);

REGISTER_KERNEL_BUILDER(Name("ResourceInitializedOp").Device(DEVICE_CPU),
                        IsResourceInitialized<StubResource>);

class ResourceCreateOp : public OpKernel {
 public:
  explicit ResourceCreateOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    OP_REQUIRES_OK(c,
                   CreateResource(c, HandleFromInput(c, 0), new StubResource));
  }
};

REGISTER_KERNEL_BUILDER(Name("ResourceCreateOp").Device(DEVICE_CPU),
                        ResourceCreateOp);

// Uses a ResourceHandle to check its validity.
class ResourceUsingOp : public OpKernel {
 public:
  explicit ResourceUsingOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    StubResource* unused;
    OP_REQUIRES_OK(ctx, LookupResource<StubResource>(
                            ctx, HandleFromInput(ctx, 0), &unused));
  }
};

REGISTER_KERNEL_BUILDER(Name("ResourceUsingOp").Device(DEVICE_CPU),
                        ResourceUsingOp);

// Various test ops without kernels. These are used to test graph construction.

REGISTER_OP("A")
    .Output("out: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("B")
    .Output("out: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Foo1")
    .Input("a: float32")
    .Input("b: int32")
    .Input("c: int32")
    .Output("d: float32")
    .Output("e: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Foo2")
    .Input("a: float32")
    .Input("b: string")
    .Input("c: string")
    .Output("d: float32")
    .Output("e: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Foo3")
    .Input("a: float32")
    .Input("b: string")
    .Input("c: float32")
    .Output("d: float32")
    .Output("e: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("CopyOp").Input("a: T").Output("b: T").Attr("T: type").SetShapeFn(
    shape_inference::UnknownShape);

REGISTER_OP("None").SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("IntOutput")
    .Output("a: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Int64Output")
    .Output("out: int64")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RefOutput")
    .Output("a: Ref(int32)")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("FloatOutput")
    .Output("a: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TwoFloatOutputs")
    .Output("a: float32")
    .Output("b: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("FiveFloatOutputs")
    .Output("a: float32")
    .Output("b: float32")
    .Output("c: float32")
    .Output("d: float32")
    .Output("e: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RefOutputFloatOutput")
    .Output("a: Ref(float32)")
    .Output("b: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RefInputFloatInput")
    .Input("a: Ref(float)")
    .Input("b: float")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("IntInput")
    .Input("a: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("IntInputIntOutput")
    .Input("a: int32")
    .Output("b: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("FloatInput")
    .Input("a: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TwoIntOutputs")
    .Output("a: int32")
    .Output("b: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("IntOutputFloatOutput")
    .Output("a: int32")
    .Output("b: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("FloatOutputStringOutput")
    .Output("a: float32")
    .Output("b: string")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TwoIntInputs")
    .Input("a: int32")
    .Input("b: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TwoFloatInputs")
    .Input("a: float32")
    .Input("b: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("IntInputFloatInput")
    .Input("a: int32")
    .Input("b: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RefInputIntInput")
    .Input("a: Ref(int32)")
    .Input("b: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TwoFloatInputsFloatOutput")
    .Input("a: float32")
    .Input("b: float32")
    .Output("c: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TwoFloatInputsIntOutput")
    .Input("a: float32")
    .Input("b: float32")
    .Output("c: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RefInputFloatInputIntOutput")
    .Input("a: Ref(float32)")
    .Input("b: float32")
    .Output("c: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ListInput")
    .Input("a: N * T")
    .Attr("N: int >= 1")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ListOutput")
    .Output("a: T")
    .Attr("T: list(type) >= 1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Unary").Input("a: T").Output("b: T").Attr("T: type").SetShapeFn(
    shape_inference::UnknownShape);

REGISTER_OP("OpWithDefaultAttr")
    .Output("a: int32")
    .Attr("default_float: float = 123.0")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("OpWithFutureDefaultAttr")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("IntAttr")
    .Output("out: int64")
    .Attr("foo: int = 1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("StringListAttr")
    .Attr("a: list(string)")
    .Attr("b: string")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("DefaultAttrs")
    .Attr("string_val: string = 'abc'")
    .Attr("string_list_val: list(string) = ['abc', '']")
    .Attr("int_val: int = 123")
    .Attr("int_list_val: list(int) = [1, 2, 3]")
    .Attr("float_val: float = 10.0")
    .Attr("float_list_val: list(float) = [10.0]")
    .Attr("bool_val: bool = true")
    .Attr("bool_list_val: list(bool) = [true, false]")
    .Attr("type_val: type = DT_INT32")
    .Attr("type_list_val: list(type) = [DT_INT32, DT_FLOAT]")
    .Attr("shape_val: shape = { dim { size: 2 } dim { size: 1 } }")
    .Attr("shape_list_val: list(shape) = [{}, { dim { size: 1} }]")
    .Attr("tensor_val: tensor = { dtype: DT_INT32 tensor_shape: {} int_val: 1}")
    .Attr(
        "tensor_list_val: list(tensor) = "
        "[{ dtype: DT_INT32 tensor_shape: {} int_val: 1}]")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("FuncAttr")
    .Attr("f: func")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Simple")
    .Input("a: int32")
    .Output("out: float")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("OutT").Output("a: T").Attr("T: type").SetShapeFn(
    shape_inference::UnknownShape);

REGISTER_OP("ReservedInput")
    .Input("input: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Polymorphic")
    .Input("a: T")
    .Output("out: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("PolymorphicOut")
    .Output("out: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("PolymorphicDefaultOut")
    .Output("out: T")
    .Attr("T: type = DT_STRING")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Binary")
    .Input("a: T")
    .Input("b: T")
    .Output("out: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Restrict")
    .Input("a: T")
    .Output("out: T")
    .Attr("T: {string, bool}")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TypeList")
    .Input("a: T")
    .Attr("T: list(type) >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TypeListTwice")
    .Input("a: T")
    .Input("b: T")
    .Attr("T: list(type) >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("OutTypeList")
    .Output("out: T")
    .Attr("T: list(type) >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TypeListRestrict")
    .Input("a: T")
    .Attr("T: list({string, bool})")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("OutTypeListRestrict")
    .Output("out: t")
    .Attr("t: list({string, bool})")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Attr").Attr("a: int").SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrFloat")
    .Attr("a: float")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrBool")
    .Attr("a: bool")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrBoolList")
    .Attr("a: list(bool)")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrMin")
    .Attr("a: int >= 5")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrListMin")
    .Attr("a: list(int) >= 2")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrEnum")
    .Attr("a: {'apples', 'oranges'}")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrEnumList")
    .Attr("a: list({'apples', 'oranges'})")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrShape")
    .Attr("a: shape")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrShapeList")
    .Attr("a: list(shape)")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrPartialShape")
    .Attr("a: shape")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrPartialShapeList")
    .Attr("a: list(shape)")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrDefault")
    .Attr("a: string = 'banana'")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrListDefault")
    .Attr("a: list(int) = [5, 15]")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrEmptyListDefault")
    .Attr("a: list(float) = []")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ReservedAttr")
    .Attr("range: int")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrTypeDefault")
    .Input("a: T")
    .Attr("T: type = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AttrListTypeDefault")
    .Input("a: N * T")
    .Input("b: N * T")
    .Attr("T: type = DT_INT32")
    .Attr("N: int")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NIntsIn")
    .Input("a: N * int32")
    .Attr("N: int >= 2")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NPolymorphicIn")
    .Input("a: N * T")
    .Attr("T: type")
    .Attr("N: int >= 2")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NPolymorphicRestrictIn")
    .Input("a: N * T")
    .Attr("T: {string, bool}")
    .Attr("N: int >= 2")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NInTwice")
    .Input("a: N * int32")
    .Input("b: N * string")
    .Attr("N: int >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NInPolymorphicTwice")
    .Input("a: N * T")
    .Input("b: N * T")
    .Attr("T: type")
    .Attr("N: int >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NInTwoTypeVariables")
    .Input("a: N * S")
    .Input("b: N * T")
    .Attr("S: type")
    .Attr("T: type")
    .Attr("N: int >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("InPolymorphicTwice")
    .Input("a: N * T")
    .Input("b: M * T")
    .Attr("T: type")
    .Attr("N: int >= 0")
    .Attr("M: int >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NIntsOut")
    .Output("a: N * int32")
    .Attr("N: int >= 2")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NIntsOutDefault")
    .Output("a: N * int32")
    .Attr("N: int >= 2 = 3")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NPolymorphicOut")
    .Output("a: N * T")
    .Attr("T: type")
    .Attr("N: int >= 2")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NPolymorphicOutDefault")
    .Output("a: N * T")
    .Attr("T: type = DT_BOOL")
    .Attr("N: int >= 2 = 2")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NPolymorphicRestrictOut")
    .Output("a: N * T")
    .Attr("T: {string, bool}")
    .Attr("N: int >= 2")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RefIn")
    .Input("a: Ref(T)")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TwoRefsIn")
    .Input("a: Ref(T)")
    .Input("b: Ref(T)")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RefOut")
    .Output("a: Ref(T)")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("SimpleStruct")
    .Output("a: n_a * int32")
    .Attr("n_a: int >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("MixedStruct")
    .Output("a: n_a * int32")
    .Output("b: float")
    .Attr("n_a: int >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ComplexStruct")
    .Output("a: n_a * int32")
    .Output("b: n_b * int64")
    .Output("c: t_c")
    .Attr("n_a: int >= 0")
    .Attr("n_b: int >= 0")
    .Attr("t_c: list(type) >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

}  // end namespace tensorflow
