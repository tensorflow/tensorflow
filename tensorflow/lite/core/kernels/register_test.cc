/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/kernels/register.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite::ops::builtin {
namespace {

TEST(BuiltinOpResolverTest, SupportsAdd) {
  BuiltinOpResolver builtin_op_resolver;
  const TfLiteRegistration *add =
      builtin_op_resolver.FindOp(::tflite::BuiltinOperator_ADD, 1);
  ASSERT_NE(add, nullptr);
  ASSERT_NE(add->init, nullptr);
  ASSERT_NE(add->free, nullptr);
  ASSERT_NE(add->prepare, nullptr);
  ASSERT_NE(add->invoke, nullptr);
}

TEST(BuiltinOpResolverTest, CopySupportsAdd) {
  BuiltinOpResolver builtin_op_resolver;
  MutableOpResolver copy = builtin_op_resolver;
  const TfLiteRegistration *add = copy.FindOp(::tflite::BuiltinOperator_ADD, 1);
  ASSERT_NE(add, nullptr);
  ASSERT_NE(add->init, nullptr);
  ASSERT_NE(add->free, nullptr);
  ASSERT_NE(add->prepare, nullptr);
  ASSERT_NE(add->invoke, nullptr);
}

#if defined(TFLITE_WITHOUT_XNNPACK)
TEST(BuiltinOpResolverTest, HasXNNPACKDelegate_QS8) {
  BuiltinOpResolver builtin_op_resolver;
  ASSERT_EQ(builtin_op_resolver.GetDelegateCreators().size(), 1);
  BuiltinOpResolver::TfLiteDelegateCreator delegate_creator =
      builtin_op_resolver.GetDelegateCreators()[0];
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate *)> delegate =
      delegate_creator(nullptr);
  const TfLiteXNNPackDelegateOptions *options =
      TfLiteXNNPackDelegateGetOptions(delegate.get());

  ASSERT_EQ(options->flags & TFLITE_XNNPACK_DELEGATE_FLAG_QU8,
            TFLITE_XNNPACK_DELEGATE_FLAG_QU8);

  ASSERT_EQ(options->flags & TFLITE_XNNPACK_DELEGATE_FLAG_QS8,
            TFLITE_XNNPACK_DELEGATE_FLAG_QS8);
}

TEST(BuiltinOpResolverTest, HasXNNPACKDelegate_QS8_QU8) {
  BuiltinOpResolver builtin_op_resolver;
  ASSERT_EQ(builtin_op_resolver.GetDelegateCreators().size(), 1);
  BuiltinOpResolver::TfLiteDelegateCreator delegate_creator =
      builtin_op_resolver.GetDelegateCreators()[0];
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate *)> delegate =
      delegate_creator(nullptr);
  const TfLiteXNNPackDelegateOptions *options =
      TfLiteXNNPackDelegateGetOptions(delegate.get());

  ASSERT_EQ(options->flags & TFLITE_XNNPACK_DELEGATE_FLAG_QU8,
            TFLITE_XNNPACK_DELEGATE_FLAG_QU8);

  ASSERT_EQ(options->flags & TFLITE_XNNPACK_DELEGATE_FLAG_QS8,
            TFLITE_XNNPACK_DELEGATE_FLAG_QS8);
}

TEST(BuiltinOpResolverTest, Disable_QU8) {
  BuiltinOpResolverWithXNNPACK builtin_op_resolver(false);
  ASSERT_EQ(builtin_op_resolver.GetDelegateCreators().size(), 1);
  BuiltinOpResolver::TfLiteDelegateCreator delegate_creator =
      builtin_op_resolver.GetDelegateCreators()[0];
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate *)> delegate =
      delegate_creator(nullptr);
  const TfLiteXNNPackDelegateOptions *options =
      TfLiteXNNPackDelegateGetOptions(delegate.get());

  ASSERT_EQ(options->flags & TFLITE_XNNPACK_DELEGATE_FLAG_QU8, 0);

  ASSERT_EQ(options->flags & TFLITE_XNNPACK_DELEGATE_FLAG_QS8,
            TFLITE_XNNPACK_DELEGATE_FLAG_QS8);
}
#endif  // TFLITE_WITHOUT_XNNPACK
}  // namespace
}  // namespace tflite::ops::builtin
