/* Copyright 2024 The OpenXLA Authors.

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

#define DEFINE_BINARY_TEST_OP(NAME, ENQUEUE, EVALUATE)       \
  template <PrimitiveType T>                                 \
  class NAME final : public BinaryTestOp<T> {                \
   public:                                                   \
    using Traits = BinaryTestOp<T>::Traits;                  \
    using Test = BinaryTestOp<T>::Test;                      \
                                                             \
    explicit NAME(Test* test) : BinaryTestOp<T>(test) {}     \
    ~NAME() override {}                                      \
                                                             \
    Traits::EnqueueOp EnqueueOp() const override ENQUEUE;    \
                                                             \
    Traits::EvaluateOp EvaluateOp() const override EVALUATE; \
  };                                                         \
  static_assert(true, "")

DEFINE_BINARY_TEST_OP(
    AddOp, { return AddEmptyBroadcastDimension(Add); },
    {
      return +[](typename Traits::NativeRefT x, typename Traits::NativeRefT y) {
        return x + y;
      };
    });
DEFINE_BINARY_TEST_OP(
    SubOp, { return AddEmptyBroadcastDimension(Sub); },
    {
      return +[](typename Traits::NativeRefT x, typename Traits::NativeRefT y) {
        return x - y;
      };
    });
DEFINE_BINARY_TEST_OP(
    MulOp, { return AddEmptyBroadcastDimension(Mul); },
    {
      return +[](typename Traits::NativeRefT x, typename Traits::NativeRefT y) {
        return x * y;
      };
    });
DEFINE_BINARY_TEST_OP(
    DivOp, { return AddEmptyBroadcastDimension(Div); },
    {
      return +[](typename Traits::NativeRefT x, typename Traits::NativeRefT y) {
        return x / y;
      };
    });
DEFINE_BINARY_TEST_OP(
    MaxOp, { return AddEmptyBroadcastDimension(Max); },
    { return ReferenceMax<typename Traits::NativeRefT>; });
DEFINE_BINARY_TEST_OP(
    MinOp, { return AddEmptyBroadcastDimension(Min); },
    { return ReferenceMin<typename Traits::NativeRefT>; });
DEFINE_BINARY_TEST_OP(
    PowOp, { return AddEmptyBroadcastDimension(Pow); }, { return std::pow; });
DEFINE_BINARY_TEST_OP(
    Atan2Op, { return AddEmptyBroadcastDimension(Atan2); },
    { return std::atan2; });
DEFINE_BINARY_TEST_OP(
    AbsComplexOp,
    { return +[](XlaOp x, XlaOp y) { return Abs(Complex(x, y)); }; },
    {
      return +[](typename Traits::NativeRefT x, typename Traits::NativeRefT y) {
        return std::abs(std::complex<typename Traits::NativeRefT>(x, y));
      };
    });

#undef DEFINE_BINARY_TEST_OP