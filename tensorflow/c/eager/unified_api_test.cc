/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
class UnifiedAPI
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(errors::OK, s.code()) << s.message();
  }

 public:
  bool UseMlir() const { return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const { return std::get<2>(GetParam()); }
};

// Checks that inputs[0] is a scalar.
Status TestScalarShape(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> inputs,
                       absl::Span<AbstractTensorHandle*> outputs) {
  PartialTensorShape shape;
  TF_RETURN_IF_ERROR(inputs[0]->Shape(&shape));
  if (shape.dims() != 0) {
    return errors::InvalidArgument(
        "Tensor expected to have scalar shape found rank: ", shape.dims());
  }
  return OkStatus();
}

TEST_P(UnifiedAPI, TestTensorShapeScalar) {
  if (UseFunction() && UseMlir()) {
    // TODO(b/173074167): Remove this.
    GTEST_SKIP() << "MlirTensor::Shape is not implemented yet.";
  }
  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    ctx.reset(ctx_raw);
  }

  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s = TestScalarTensorHandle<float, TF_FLOAT>(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    x.reset(x_raw);
  }

  Status s = RunModel(TestScalarShape, ctx.get(),
                      /*inputs=*/{x.get()},
                      /*outputs=*/{},
                      /*use_function=*/UseFunction());
  ASSERT_EQ(errors::OK, s.code()) << s.message();
}

// Checks that inputs[0] is a matrix with shape 2x4.
Status TestTensorShape2x4(AbstractContext* ctx,
                          absl::Span<AbstractTensorHandle* const> inputs,
                          absl::Span<AbstractTensorHandle*> outputs) {
  PartialTensorShape shape;
  TF_RETURN_IF_ERROR(inputs[0]->Shape(&shape));
  if (shape.dims() != 2) {
    return errors::InvalidArgument(
        "Tensor expected to have rank 2 found rank: ", shape.dims());
  }
  int64_t dim_sizes[] = {2, 4};
  for (int i = 0; i < shape.dims(); i++) {
    if (shape.dim_size(i) != dim_sizes[i]) {
      return errors::InvalidArgument("Dim ", i, " expected to be of size ",
                                     dim_sizes[i],
                                     " found: ", shape.dim_size(i));
    }
  }
  return OkStatus();
}

TEST_P(UnifiedAPI, TestTensorShape2x4) {
  if (UseFunction() && UseMlir()) {
    // TODO(b/173074167): Remove this.
    GTEST_SKIP() << "MlirTensor::Shape is not implemented yet.";
  }
  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    ctx.reset(ctx_raw);
  }

  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    float data[] = {0., 0., 0., 0., 0., 0., 0., 0};
    int64_t dim_sizes[] = {2, 4};
    Status s = TestTensorHandleWithDims<float, TF_FLOAT>(ctx.get(), data,
                                                         dim_sizes, 2, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    x.reset(x_raw);
  }

  Status s = RunModel(TestTensorShape2x4, ctx.get(),
                      /*inputs=*/{x.get()},
                      /*outputs=*/{},
                      /*use_function=*/UseFunction());
  ASSERT_EQ(errors::OK, s.code()) << s.message();
}

TEST_P(UnifiedAPI, TestUnknownShapeTracing) {
  if (!UseFunction()) {
    GTEST_SKIP() << "Tracing only test.";
  }
  if (UseMlir()) {
    // TODO(b/173074167): Remove this.
    GTEST_SKIP() << "MlirTensor::Shape is not implemented yet.";
  }
  AbstractContextPtr ctx(BuildFunction("test_fn"));
  AbstractTensorHandlePtr x;
  {
    tracing::TracingTensorHandle* x_raw = nullptr;
    PartialTensorShape shape;
    Status s = dyn_cast<tracing::TracingContext>(ctx.get())->AddParameter(
        DT_FLOAT, shape, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    x.reset(x_raw);
  }

  PartialTensorShape shape;
  Status s = x->Shape(&shape);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  ASSERT_TRUE(shape.unknown_rank());
}

TEST_P(UnifiedAPI, TestPartialShapeTracing) {
  if (!UseFunction()) {
    GTEST_SKIP() << "Tracing only test.";
  }
  if (UseMlir()) {
    GTEST_SKIP() << "MlirTensor::Shape is not implemented yet.";
  }
  AbstractContextPtr ctx(BuildFunction("test_fn"));
  AbstractTensorHandlePtr x;
  {
    tracing::TracingTensorHandle* x_raw = nullptr;
    PartialTensorShape shape;
    int64_t dim_sizes[] = {2, -1};
    Status s = PartialTensorShape::MakePartialShape(dim_sizes, 2, &shape);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    s = dyn_cast<tracing::TracingContext>(ctx.get())->AddParameter(
        DT_FLOAT, shape, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.message();
    x.reset(x_raw);
  }

  PartialTensorShape shape;
  Status s = x->Shape(&shape);
  ASSERT_EQ(errors::OK, s.code()) << s.message();
  ASSERT_FALSE(shape.unknown_rank());

  ASSERT_EQ(2, shape.dim_size(0));
  ASSERT_EQ(-1, shape.dim_size(1));
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCppAPI, UnifiedAPI,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(true, false),
                       /*use_function*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCppAPI, UnifiedAPI,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace tensorflow
