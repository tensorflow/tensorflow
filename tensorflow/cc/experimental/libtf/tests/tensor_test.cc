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
#include <cstdint>

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {

using AbstractContextPtr = tensorflow::AbstractContextPtr;
using AbstractContext = tensorflow::AbstractContext;
using AbstractTensorHandle = tensorflow::AbstractTensorHandle;
using TF_StatusPtr = tensorflow::TF_StatusPtr;
using Status = tensorflow::Status;

class UnifiedCAPI
    : public ::testing::TestWithParam<std::tuple<const char*, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = tensorflow::StatusFromTF_Status(status.get());
    CHECK_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
  }
};

namespace {
template <class T>
TaggedValue MakeContext(T runtime) {
  AbstractContext* ctx_raw = nullptr;
  Status s = BuildImmediateExecutionContext(runtime, &ctx_raw);
  // ASSERT_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
  return TaggedValue::Capsule(static_cast<void*>(ctx_raw), [](void* p) {
    tensorflow::internal::AbstractContextDeleter()(
        static_cast<AbstractContext*>(p));
  });
}
}  // namespace

TEST_P(UnifiedCAPI, HoldTensors) {
  // Use the parametrized test parameters to make a context.
  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  // Construct a scalar.
  impl::TaggedValueTensor x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s = TestScalarTensorHandle<float, TF_FLOAT>(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    x.reset(x_raw, false);
  }
  // Manually copy pointer so we can later compare the reference count.
  impl::TaggedValueTensor x2(x);

  {
    // Take ownership of x2 pointer. Semantics of AbstractTensorHandlePtr
    // are that it has a reference. Here we steal that reference and put it
    // into TaggedValue. If we used release() we would double free.
    impl::TaggedValue tensor(std::move(x2));
    auto list = TaggedValue::List();
    // Test adding values by copying and moving.
    list.list().emplace_back(3.f);
    list.list().push_back(tensor);
    list.list().emplace_back(std::move(tensor));
    ASSERT_FALSE(x->RefCountIsOne());
  }
  ASSERT_TRUE(x->RefCountIsOne());
}

TaggedValue MakeScalarTensor(TaggedValue self, TaggedValue val) {
  if (val.type() != TaggedValue::FLOAT32) return TaggedValue::None();
  if (self.type() != TaggedValue::DICT) return TaggedValue::None();
  TaggedValue ctx_capsule = (self.dict())[TaggedValue("context")];
  AbstractContext* ctx = static_cast<AbstractContext*>(ctx_capsule.capsule());
  AbstractTensorHandle* x_raw = nullptr;
  Status s = TestScalarTensorHandle<float, TF_FLOAT>(ctx, val.f32(), &x_raw);
  if (!s.ok()) return TaggedValue::None();
  return TaggedValue(impl::TaggedValueTensor(x_raw, false));
}
TEST_P(UnifiedCAPI, SimpleCreationFunctions) {
  // Use the parametrized test parameters to make a context.
  TaggedValue context = MakeContext(std::get<1>(GetParam()));
  Object methods;
  methods.Set(String("context"), Handle(MakeContext(std::get<1>(GetParam()))));
  methods.Set(String("make_scalar_tensor"),
              Callable(TaggedValue(MakeScalarTensor)));

  Handle foo = *methods.Get<Callable>(String("make_scalar_tensor"))
                    ->Call<Handle>(methods, Float(3.f));
}

INSTANTIATE_TEST_SUITE_P(Tracing, UnifiedCAPI,
                         ::testing::Combine(::testing::Values("graphdef",
                                                              "mlir"),
                                            ::testing::Values(true, false)));

}  // namespace libtf
}  // namespace tf
