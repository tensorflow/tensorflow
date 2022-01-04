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

#include "tensorflow/core/function/runtime_client.h"

#include <stdint.h>

#include <memory>
#include <vector>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace core {
namespace function {
namespace {

int IntValue(ImmediateExecutionTensorHandle& h) {
  Status status;
  AbstractTensorPtr t(h.Resolve(&status));
  DCHECK(status.ok());
  switch (h.DataType()) {
    case DT_INT32:
      return *(static_cast<int32_t*>(t->Data()));
    case DT_INT64:
      return *(static_cast<int64_t*>(t->Data()));
    default:
      DCHECK(false) << "invalid data type";
      return 0;
  }
}

FunctionDef MakeSmallestFunction() {
  FunctionDef fd;
  protobuf::TextFormat::Parser parser;
  CHECK(parser.ParseFromString(
      R"pb(signature {
             name: 'SmallestFunction'
             output_arg { name: 'o' type: DT_INT32 }
           }
           node_def {
             name: 'retval'
             op: 'Const'
             attr {
               key: 'dtype'
               value { type: DT_INT32 }
             }
             attr {
               key: 'value'
               value {
                 tensor {
                   dtype: DT_INT32
                   tensor_shape {}
                   int_val: 1
                 }
               }
             }
           }
           ret { key: 'o' value: 'retval:output' })pb",
      &fd));
  return fd;
}

TEST(CreateTest, Call) {
  Runtime rt(GlobalEagerContext());
  TF_ASSERT_OK(rt.CreateFunctionProto(MakeSmallestFunction()));

  StatusOr<ReturnValues> rets = rt.CallFunction("SmallestFunction", {});
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), 1);
  ASSERT_EQ(rets->at(0)->DataType(), DT_INT32);
  EXPECT_EQ(IntValue(*(rets->at(0))), 1);
}

TEST(CreateTest, GetRoundtrip) {
  Runtime rt(GlobalEagerContext());
  TF_ASSERT_OK(rt.CreateFunctionProto(MakeSmallestFunction()));

  StatusOr<FunctionDef> fdef_ret = rt.GetFunctionProto("SmallestFunction");
  TF_ASSERT_OK(fdef_ret.status());

  FunctionDef fdef = *fdef_ret;
  fdef.mutable_signature()->set_name("SecondFunction");

  TF_ASSERT_OK(rt.CreateFunctionProto(fdef));

  StatusOr<ReturnValues> rets = rt.CallFunction("SecondFunction", {});
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), 1);
  ASSERT_EQ(rets->at(0)->DataType(), DT_INT32);
  EXPECT_EQ(IntValue(*(rets->at(0))), 1);
}

}  // namespace
}  // namespace function
}  // namespace core
}  // namespace tensorflow
