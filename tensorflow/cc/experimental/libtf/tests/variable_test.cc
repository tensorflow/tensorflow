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
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/graph_function.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/ops/resource_variable_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/cc/experimental/libtf/function.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
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
using tensorflow::PartialTensorShape;
using tensorflow::Status;
using tensorflow::TF_StatusPtr;

class VariableTest
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

template <typename T>
void ExpectEquals(AbstractTensorHandle* t, T expected) {
  TF_Tensor* result_t;
  Status s = tensorflow::GetValue(t, &result_t);
  ASSERT_TRUE(s.ok()) << s.message();
  auto value = static_cast<T*>(TF_TensorData(result_t));
  EXPECT_EQ(*value, expected);
  TF_DeleteTensor(result_t);
}

TEST_P(VariableTest, CreateAssignReadDestroy) {
  // Create uninitialized variable.
  tensorflow::AbstractTensorHandlePtr var;
  {
    AbstractTensorHandle* var_ptr = nullptr;
    PartialTensorShape scalar_shape;
    TF_EXPECT_OK(
        PartialTensorShape::MakePartialShape<int32_t>({}, 0, &scalar_shape));
    TF_EXPECT_OK(tensorflow::ops::VarHandleOp(ctx_.get(), &var_ptr, DT_FLOAT,
                                              scalar_shape));
    var.reset(var_ptr);
  }
  // Assign a value.
  auto x = CreateScalarTensor<float, TF_FLOAT>(2.0f);
  TF_EXPECT_OK(
      tensorflow::ops::AssignVariableOp(ctx_.get(), var.get(), x.get()));
  // Read variable.
  tensorflow::AbstractTensorHandlePtr value;
  {
    AbstractTensorHandle* value_ptr = nullptr;
    TF_EXPECT_OK(tensorflow::ops::ReadVariableOp(ctx_.get(), var.get(),
                                                 &value_ptr, DT_FLOAT));
    value.reset(value_ptr);
  }
  ExpectEquals(value.get(), 2.0f);
  // Destroy variable.
  TF_EXPECT_OK(tensorflow::ops::DestroyResourceOp(ctx_.get(), var.get()));
}

INSTANTIATE_TEST_SUITE_P(TF2CAPI, VariableTest,
                         ::testing::Combine(::testing::Values("graphdef",
                                                              "mlir"),
                                            ::testing::Values(false)));

}  // namespace libtf
}  // namespace tf
