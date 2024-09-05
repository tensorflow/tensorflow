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
#include "tensorflow/cc/experimental/libtf/function.h"

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/graph_function.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {
using tensorflow::AbstractContext;
using tensorflow::AbstractContextPtr;
using tensorflow::AbstractFunctionPtr;
using tensorflow::AbstractTensorHandle;
using tensorflow::DT_FLOAT;
using tensorflow::FunctionDef;
using tensorflow::FunctionDefHelper;
using tensorflow::PartialTensorShape;
using tensorflow::Status;
using tensorflow::StatusOr;
using tensorflow::TF_StatusPtr;
using tensorflow::tracing::graph::GraphFunction;

class FunctionTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool>> {
 public:
  template <class T, TF_DataType datatype>
  impl::TaggedValueTensor CreateScalarTensor(T val) {
    AbstractTensorHandle* raw = nullptr;
    Status s = TestScalarTensorHandle<T, datatype>(ctx_.get(), val, &raw);
    CHECK_EQ(tensorflow::errors::OK, s.code()) << s.message();
    return impl::TaggedValueTensor(raw, /*add_ref=*/false);
  }

  bool UseTfrt() { return std::get<1>(GetParam()); }

  AbstractContextPtr ctx_;

 protected:
  void SetUp() override {
    // Set the tracing impl, GraphDef vs MLIR.
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = tensorflow::StatusFromTF_Status(status.get());
    CHECK_EQ(tensorflow::errors::OK, s.code()) << s.message();

    // Set the runtime impl, Core RT vs TFRT.
    AbstractContext* ctx_raw = nullptr;
    s = BuildImmediateExecutionContext(UseTfrt(), &ctx_raw);
    CHECK_EQ(tensorflow::errors::OK, s.code()) << s.message();
    ctx_.reset(ctx_raw);
  }
};

// TODO(b/191361582): Use Abstract* APIs for building functions so that we can
// test with MLIR.
FunctionDef SquareFunc() {
  return FunctionDefHelper::Define(
      // Function Name
      "SquareFunc",
      // Args
      {"x: float"},
      // Returns
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {{/*ret=*/{"y"},
        /*op=*/"Square",
        /*arg=*/{"x"},
        /*attr=*/{{"T", DT_FLOAT}},
        /*dep=*/{},
        /*device=*/"",
        /*name=*/"square"}});
}

FunctionDef AddFunc() {
  return FunctionDefHelper::Define(
      // Function Name
      "AddFunc",
      // Args
      {"x: float", "y: float"},
      // Returns
      {"z: float"},
      // Attr def
      {},
      // Nodes
      {{/*ret=*/{"z"},
        /*op=*/"Add",
        /*arg=*/{"x", "y"},
        /*attr=*/{{"T", DT_FLOAT}},
        /*dep=*/{},
        /*device=*/"",
        /*name=*/"add"}});
}

FunctionDef IdentityNFunc() {
  return FunctionDefHelper::Define(
      // Function Name
      "IdentityNFunc",
      // Args
      {"x: float", "y: float"},
      // Returns
      {"u: float", "v: float"},
      // Attr def
      {},
      // Nodes
      {{/*ret=*/{"u", "v"},
        /*op=*/"IdentityN",
        /*arg=*/{"x", "y"},
        /*attr=*/{{"T", tensorflow::DataTypeSlice({DT_FLOAT, DT_FLOAT})}},
        /*dep=*/{},
        /*device=*/""}});
}

template <typename T>
void ExpectEquals(AbstractTensorHandle* t, T expected) {
  TF_Tensor* result_t;
  Status s = tensorflow::GetValue(t, &result_t);
  ASSERT_TRUE(s.ok()) << s.message();
  auto value = static_cast<T*>(TF_TensorData(result_t));
  EXPECT_EQ(*value, expected);
  TF_DeleteTensor(result_t);
}

// TODO(srbs): Add tests for captures.
// TODO(srbs): Add tests for polymorphism (different shapes and dtypes).
TEST_P(FunctionTest, Square) {
  // Construct a scalar.
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  FunctionDef fdef = SquareFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue signature(unknown_shape, DT_FLOAT);
  Status s = tf_function.RegisterTrace(std::move(trace), signature, signature);
  ASSERT_TRUE(s.ok()) << s.message();
  TaggedValue args(std::move(x));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(v.ok()) << v.status().message();
  const TaggedValue& result = v.value();
  AbstractTensorHandle* t = result.tensor().get();
  ExpectEquals(t, 4.0f);
}

TEST_P(FunctionTest, Add) {
  // Construct a scalar.
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  FunctionDef fdef = AddFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue tensor_spec(unknown_shape, DT_FLOAT);
  TaggedValue input_signature = TaggedValue::Tuple();
  input_signature.tuple().emplace_back(tensor_spec);
  input_signature.tuple().emplace_back(tensor_spec);
  Status s =
      tf_function.RegisterTrace(std::move(trace), input_signature, tensor_spec);
  ASSERT_TRUE(s.ok()) << s.message();
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(x));
  args.tuple().emplace_back(TaggedValue(x));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(v.ok()) << v.status().message();
  const TaggedValue& result = v.value();
  ExpectEquals(result.tensor().get(), 4.0f);
}

TEST_P(FunctionTest, IdentityN) {
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  impl::TaggedValueTensor y = CreateScalarTensor<float, TF_FLOAT>(4.0f);
  FunctionDef fdef = IdentityNFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue tensor_spec(unknown_shape, DT_FLOAT);
  TaggedValue signature = TaggedValue::Tuple();
  signature.tuple().emplace_back(tensor_spec);
  signature.tuple().emplace_back(tensor_spec);
  Status s = tf_function.RegisterTrace(std::move(trace), signature, signature);
  ASSERT_TRUE(s.ok()) << s.message();
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(x));
  args.tuple().emplace_back(TaggedValue(y));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(v.ok()) << v.status().message();
  const TaggedValue& result = v.value();
  ExpectEquals(result.tuple()[0].tensor().get(), 2.0f);
  ExpectEquals(result.tuple()[1].tensor().get(), 4.0f);
}

TEST_P(FunctionTest, UnaryFuncCalledWithMultipleArgsFails) {
  // Construct a scalar.
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  FunctionDef fdef = SquareFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue signature(unknown_shape, DT_FLOAT);
  Status s = tf_function.RegisterTrace(std::move(trace), signature, signature);
  ASSERT_TRUE(s.ok()) << s.message();
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(x));
  args.tuple().emplace_back(TaggedValue(x));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(tensorflow::errors::IsInvalidArgument(v.status()));
  ASSERT_TRUE(absl::StrContains(v.status().message(), "No match"));
}

TEST_P(FunctionTest, IncorrectArityOfOutputSignatureFails) {
  if (UseTfrt()) {
    GTEST_SKIP() << "TFRT crashes if expected number of output tensors does not"
                    " match actual.";
  }
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  impl::TaggedValueTensor y = CreateScalarTensor<float, TF_FLOAT>(4.0f);
  FunctionDef fdef = IdentityNFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue tensor_spec(unknown_shape, DT_FLOAT);
  TaggedValue input_signature = TaggedValue::Tuple();
  input_signature.tuple().emplace_back(tensor_spec);
  input_signature.tuple().emplace_back(tensor_spec);
  // This is wrong!
  TaggedValue output_signature(unknown_shape, DT_FLOAT);
  Status s = tf_function.RegisterTrace(std::move(trace), input_signature,
                                       output_signature);
  ASSERT_TRUE(s.ok()) << s.message();
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(x));
  args.tuple().emplace_back(TaggedValue(y));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(tensorflow::errors::IsInvalidArgument(v.status())) << v.status();
  ASSERT_TRUE(absl::StrContains(v.status().message(),
                                "Expecting 2 outputs, but *num_retvals is 1"));
}

TEST_P(FunctionTest, IncorrectDtypeInOutputSignatureFails) {
  // Construct a scalar.
  impl::TaggedValueTensor x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  FunctionDef fdef = AddFunc();
  AbstractFunctionPtr trace(new GraphFunction(fdef), /*add_ref=*/false);
  Function tf_function;
  PartialTensorShape unknown_shape;
  TaggedValue input_tensor_spec(unknown_shape, tensorflow::DT_FLOAT);
  TaggedValue input_signature = TaggedValue::Tuple();
  input_signature.tuple().emplace_back(input_tensor_spec);
  input_signature.tuple().emplace_back(input_tensor_spec);
  // Incorrect type.
  TaggedValue output_tensor_spec(unknown_shape, tensorflow::DT_INT64);
  Status s = tf_function.RegisterTrace(std::move(trace), input_signature,
                                       output_tensor_spec);
  ASSERT_TRUE(s.ok()) << s.message();
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(x));
  args.tuple().emplace_back(TaggedValue(x));
  StatusOr<TaggedValue> v = tf_function.Execute(ctx_.get(), args);
  ASSERT_TRUE(tensorflow::errors::IsInternal(v.status())) << v.status();
  ASSERT_TRUE(
      absl::StrContains(v.status().message(), "Shape and dtype of tensor"));
  ASSERT_TRUE(absl::StrContains(v.status().message(),
                                "does not match that in signature"));
}

INSTANTIATE_TEST_SUITE_P(TF2CAPI, FunctionTest,
                         ::testing::Combine(::testing::Values("graphdef",
                                                              "mlir"),
                                            ::testing::Values(false)));

}  // namespace libtf
}  // namespace tf
