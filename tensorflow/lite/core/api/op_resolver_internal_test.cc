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

#include "tensorflow/lite/core/api/op_resolver_internal.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {

using ops::builtin::BuiltinOpResolver;

namespace {

TEST(OpResolverInternal, BuiltinOpResolverContainsOnlyPredefinedOps) {
  BuiltinOpResolver builtin_op_resolver;
  EXPECT_EQ(OpResolverInternal::MayContainUserDefinedOps(builtin_op_resolver),
            false);
}

TEST(OpResolverInternal, EmptyMutableOpResolverContainsOnlyPredefinedOps) {
  MutableOpResolver empty_mutable_op_resolver;
  EXPECT_EQ(
      OpResolverInternal::MayContainUserDefinedOps(empty_mutable_op_resolver),
      false);
}

TEST(OpResolverInternal,
     MutableOpResolverAddBuiltinNullptrContainsOnlyPredefinedOps) {
  MutableOpResolver mutable_op_resolver;
  mutable_op_resolver.AddBuiltin(BuiltinOperator_ADD, nullptr, 1);
  EXPECT_EQ(OpResolverInternal::MayContainUserDefinedOps(mutable_op_resolver),
            false);
}

TEST(OpResolverInternal,
     MutableOpResolverRedefineBuiltinDoesNotContainOnlyPredefinedOps) {
  MutableOpResolver mutable_op_resolver;
  // Redefine the "add" op with a non-standard meaning ("multiply").
  mutable_op_resolver.AddBuiltin(BuiltinOperator_ADD,
                                 tflite::ops::builtin::Register_MUL(), 1);
  EXPECT_EQ(OpResolverInternal::MayContainUserDefinedOps(mutable_op_resolver),
            true);
}

TEST(OpResolverInternal,
     MutableOpResolverAddCustomDoesNotContainOnlyPredefinedOps) {
  MutableOpResolver mutable_op_resolver;
  mutable_op_resolver.AddCustom("my_custom_op",
                                tflite::ops::builtin::Register_ADD(), 1);
  EXPECT_EQ(OpResolverInternal::MayContainUserDefinedOps(mutable_op_resolver),
            true);
}

class ChainableOpResolver : public MutableOpResolver {
 public:
  using MutableOpResolver::ChainOpResolver;
};

TEST(OpResolverInternal, ChainedBuiltinOpResolverContainOnlyPredefinedOps) {
  BuiltinOpResolver builtin_op_resolver;
  ChainableOpResolver chainable_op_resolver;
  chainable_op_resolver.ChainOpResolver(&builtin_op_resolver);
  EXPECT_EQ(OpResolverInternal::MayContainUserDefinedOps(chainable_op_resolver),
            false);
}

TEST(OpResolverInternal,
     ChainedCustomOpResolverDoesNotContainOnlyPredefinedOps) {
  MutableOpResolver mutable_op_resolver;
  mutable_op_resolver.AddCustom("my_custom_op",
                                tflite::ops::builtin::Register_ADD(), 1);
  ChainableOpResolver chainable_op_resolver;
  chainable_op_resolver.ChainOpResolver(&mutable_op_resolver);
  EXPECT_EQ(OpResolverInternal::MayContainUserDefinedOps(chainable_op_resolver),
            true);
}

}  // anonymous namespace

}  // namespace tflite
