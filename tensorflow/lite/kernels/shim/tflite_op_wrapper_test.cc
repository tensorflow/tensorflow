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
#include "tensorflow/lite/kernels/shim/tflite_op_wrapper.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/tflite_op_shim.h"

namespace tflite {
namespace shim {
namespace op_wrapper {
namespace {

// Tests the created type of the variant is correct.
class VariantOpTest : public ::testing::Test {
 public:
  // Fake template op to test against.
  template <shim::Runtime Rt, typename... Ts>
  class TmplOp {};

  // For checking if variant has a member type
  template <typename T, typename VARIANT_T>
  struct isVariantMember;

  template <typename T, typename... ALL_T>
  struct isVariantMember<T, std::variant<ALL_T...>>
      : public std::disjunction<std::is_same<T, ALL_T>...> {};

  // Names for parameters
  static constexpr char kAttrName[] = "AttrName";
};

TEST_F(VariantOpTest, TestVariantOpCreation_1) {
  using VOp = VariantOp<Runtime::kTfLite, TmplOp,
                        Attr<AttrName<kAttrName>, int64_t>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 1);

  bool b;
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t>, VOp>::value;
  EXPECT_TRUE(b);
}

TEST_F(VariantOpTest, TestVariantOpCreation_2) {
  using VOp = VariantOp<Runtime::kTfLite, TmplOp,
                        Attr<AttrName<kAttrName>, int64_t, bool>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 2);

  bool b;
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, bool>, VOp>::value;
  EXPECT_TRUE(b);
}

TEST_F(VariantOpTest, TestVariantOpCreation_1x1) {
  using VOp =
      VariantOp<Runtime::kTfLite, TmplOp, Attr<AttrName<kAttrName>, int64_t>,
                Attr<AttrName<kAttrName>, bool>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 1);

  bool b;
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool>, VOp>::value;
  EXPECT_TRUE(b);
}

TEST_F(VariantOpTest, TestVariantOpCreation_1x1x1) {
  using VOp =
      VariantOp<Runtime::kTfLite, TmplOp, Attr<AttrName<kAttrName>, int64_t>,
                Attr<AttrName<kAttrName>, bool>,
                Attr<AttrName<kAttrName>, bool>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 1);

  bool b;
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool, bool>,
                      VOp>::value;
  EXPECT_TRUE(b);
}

TEST_F(VariantOpTest, TestVariantOpCreation_2x1) {
  using VOp = VariantOp<Runtime::kTfLite, TmplOp,
                        Attr<AttrName<kAttrName>, int64_t, float>,
                        Attr<AttrName<kAttrName>, bool>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 2);

  bool b;
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, float, bool>, VOp>::value;
  EXPECT_TRUE(b);
}

TEST_F(VariantOpTest, TestVariantOpCreation_1x2) {
  using VOp =
      VariantOp<Runtime::kTfLite, TmplOp, Attr<AttrName<kAttrName>, int64_t>,
                Attr<AttrName<kAttrName>, bool, float>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 2);

  bool b;
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, float>, VOp>::value;
  EXPECT_TRUE(b);
}

TEST_F(VariantOpTest, TestVariantOpCreation_2x2) {
  using VOp = VariantOp<Runtime::kTfLite, TmplOp,
                        Attr<AttrName<kAttrName>, int64_t, int32_t>,
                        Attr<AttrName<kAttrName>, bool, float>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 4);

  bool b;
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, float>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, bool>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, float>, VOp>::value;
  EXPECT_TRUE(b);
}

TEST_F(VariantOpTest, TestVariantOpCreation_3x3) {
  using VOp = VariantOp<Runtime::kTfLite, TmplOp,
                        Attr<AttrName<kAttrName>, int64_t, int32_t, int8_t>,
                        Attr<AttrName<kAttrName>, bool, float, char>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 9);

  bool b;
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, float>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, char>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, bool>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, float>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, char>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int8_t, bool>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int8_t, float>, VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int8_t, char>, VOp>::value;
  EXPECT_TRUE(b);
}

TEST_F(VariantOpTest, TestVariantOpCreation_2x2x2) {
  using VOp = VariantOp<Runtime::kTfLite, TmplOp,
                        Attr<AttrName<kAttrName>, int64_t, int32_t>,
                        Attr<AttrName<kAttrName>, bool, float>,
                        Attr<AttrName<kAttrName>, char, int8_t>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 8);

  bool b;
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool, char>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool, int8_t>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, float, char>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, float, int8_t>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, bool, char>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, bool, int8_t>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, float, char>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, float, int8_t>,
                      VOp>::value;
  EXPECT_TRUE(b);
}

TEST_F(VariantOpTest, TestVariantOpCreation_2x1x3x1) {
  using VOp = VariantOp<Runtime::kTfLite, TmplOp,
                        Attr<AttrName<kAttrName>, int64_t, int32_t>,
                        Attr<AttrName<kAttrName>, bool>,
                        Attr<AttrName<kAttrName>, char, int8_t, float>,
                        Attr<AttrName<kAttrName>, uint16_t>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 6);

  bool b;
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool, char, uint16_t>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool, int8_t, uint16_t>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int64_t, bool, float, uint16_t>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, bool, char, uint16_t>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, bool, int8_t, uint16_t>,
                      VOp>::value;
  EXPECT_TRUE(b);
  b = isVariantMember<TmplOp<Runtime::kTfLite, int32_t, bool, float, uint16_t>,
                      VOp>::value;
  EXPECT_TRUE(b);
}

TEST_F(VariantOpTest, TestVariantOpCreation_4x4x6) {
  using VOp =
      VariantOp<Runtime::kTfLite, TmplOp,
                Attr<AttrName<kAttrName>, int64_t, int32_t, int16_t, int8_t>,
                Attr<AttrName<kAttrName>, int64_t, int32_t, int16_t, int8_t>,
                Attr<AttrName<kAttrName>, int64_t, int32_t, int16_t, int8_t,
                     bool, float>>::type;

  EXPECT_EQ(std::variant_size_v<VOp>, 96);
}

// Tests the correct type of the variant is set given the parameter types of the
// context in the Init function.
class SetVariantOpTest : public ::testing::Test {
 public:
  // Extend OpWrapper to get access to underlying op variant for testing.
  template <Runtime Rt, template <Runtime, typename...> typename Op,
            typename... As>
  class OpWrapperFriend : public OpWrapper<Rt, Op, As...> {
   public:
    using TmplOpType = typename VariantOp<Rt, Op, As...>::type;
    TmplOpType* GetOp() { return this->op_.get(); }
  };

  // Fake template op to test against.
  template <Runtime Rt, typename... Ts>
  class TmplOp : public OpKernelShim<TmplOp, Rt, Ts...> {
   public:
    using typename OpKernelShim<TmplOp, Rt, Ts...>::InitContext;
    absl::Status Init(InitContext* ctx) { return absl::OkStatus(); }
  };

  // Fake InitContext used to set the flexbuffer attribute map.
  class FakeInitContext : public TfLiteInitContext {
   public:
    explicit FakeInitContext(const flexbuffers::Map* m)
        : TfLiteInitContext(nullptr, m) {}
  };

  // Helper methods for creating a FakeInitContext
  template <typename T>
  flexbuffers::Map CreateAttrMap() {
    fbb_ = std::make_unique<flexbuffers::Builder>();
    fbb_->Map([&]() {
      fbb_->Int(kAttrName1, static_cast<int>(typeToTfLiteType<T>()));
    });
    fbb_->Finish();
    return flexbuffers::GetRoot(fbb_->GetBuffer()).AsMap();
  }

  template <typename T, typename U>
  flexbuffers::Map CreateAttrMap() {
    fbb_ = std::make_unique<flexbuffers::Builder>();
    fbb_->Map([&]() {
      fbb_->Int(kAttrName1, static_cast<int>(typeToTfLiteType<T>()));
      fbb_->Int(kAttrName2, static_cast<int>(typeToTfLiteType<U>()));
    });
    fbb_->Finish();
    return flexbuffers::GetRoot(fbb_->GetBuffer()).AsMap();
  }

  template <typename T, typename U, typename V>
  flexbuffers::Map CreateAttrMap() {
    fbb_ = std::make_unique<flexbuffers::Builder>();
    fbb_->Map([&]() {
      fbb_->Int(kAttrName1, static_cast<int>(typeToTfLiteType<T>()));
      fbb_->Int(kAttrName2, static_cast<int>(typeToTfLiteType<U>()));
      fbb_->Int(kAttrName3, static_cast<int>(typeToTfLiteType<V>()));
    });
    fbb_->Finish();
    return flexbuffers::GetRoot(fbb_->GetBuffer()).AsMap();
  }

  // Names for parameters
  static constexpr char kAttrName1[] = "AttrName1";
  static constexpr char kAttrName2[] = "AttrName2";
  static constexpr char kAttrName3[] = "AttrName3";

 private:
  // These must exist for length of test
  std::unique_ptr<flexbuffers::Builder> fbb_;
};

TEST_F(SetVariantOpTest, TestSetVariantOp_1) {
  auto op_wrapper = OpWrapperFriend<Runtime::kTfLite, TmplOp,
                                    Attr<AttrName<kAttrName1>, bool>>();

  auto map = CreateAttrMap<bool>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
}

TEST_F(SetVariantOpTest, TestSetVariantOp_1x1) {
  auto op_wrapper = OpWrapperFriend<Runtime::kTfLite, TmplOp,
                                    Attr<AttrName<kAttrName1>, bool>,
                                    Attr<AttrName<kAttrName2>, int32_t>>();

  auto map = CreateAttrMap<bool, int32_t>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
}

TEST_F(SetVariantOpTest, TestSetVariantOp_1x1x1) {
  auto op_wrapper = OpWrapperFriend<
      Runtime::kTfLite, TmplOp, Attr<AttrName<kAttrName1>, bool>,
      Attr<AttrName<kAttrName2>, int32_t>, Attr<AttrName<kAttrName3>, float>>();

  auto map = CreateAttrMap<bool, int32_t, float>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b =
      std::holds_alternative<TmplOp<Runtime::kTfLite, bool, int32_t, float>>(
          *op_wrapper.GetOp());
  EXPECT_TRUE(b);
}

TEST_F(SetVariantOpTest, TestSetVariantOp_2) {
  auto op_wrapper =
      OpWrapperFriend<Runtime::kTfLite, TmplOp,
                      Attr<AttrName<kAttrName1>, bool, int32_t>>();

  auto map = CreateAttrMap<bool>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b;
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
}

TEST_F(SetVariantOpTest, TestSetVariantOp_2x1) {
  auto op_wrapper = OpWrapperFriend<Runtime::kTfLite, TmplOp,
                                    Attr<AttrName<kAttrName1>, bool, int32_t>,
                                    Attr<AttrName<kAttrName2>, float>>();

  auto map = CreateAttrMap<int32_t, float>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b;
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, int32_t, float>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, float>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
}

TEST_F(SetVariantOpTest, TestSetVariantOp_1x2) {
  auto op_wrapper =
      OpWrapperFriend<Runtime::kTfLite, TmplOp,
                      Attr<AttrName<kAttrName1>, bool>,
                      Attr<AttrName<kAttrName2>, float, int32_t>>();

  auto map = CreateAttrMap<bool, float>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b;
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, float>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
}

TEST_F(SetVariantOpTest, TestSetVariantOp_2x2) {
  auto op_wrapper =
      OpWrapperFriend<Runtime::kTfLite, TmplOp,
                      Attr<AttrName<kAttrName1>, bool, int64_t>,
                      Attr<AttrName<kAttrName2>, float, int32_t>>();

  auto map = CreateAttrMap<bool, float>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b;
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, float>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, int64_t, float>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, int64_t, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);

  map = CreateAttrMap<bool, int32_t>();
  context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, float>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, int64_t, float>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, int64_t, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);

  map = CreateAttrMap<int64_t, float>();
  context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, float>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, int64_t, float>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, int64_t, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);

  map = CreateAttrMap<int64_t, int32_t>();
  context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, float>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, bool, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, int64_t, float>>(
      *op_wrapper.GetOp());
  EXPECT_FALSE(b);
  b = std::holds_alternative<TmplOp<Runtime::kTfLite, int64_t, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
}

TEST_F(SetVariantOpTest, TestSetVariantOp_3x3) {
  auto op_wrapper = OpWrapperFriend<
      Runtime::kTfLite, TmplOp,
      Attr<AttrName<kAttrName1>, bool, int64_t, ::tensorflow::tstring>,
      Attr<AttrName<kAttrName2>, float, int32_t, uint32_t>>();

  auto map = CreateAttrMap<::tensorflow::tstring, int32_t>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b;
  b = std::holds_alternative<
      TmplOp<Runtime::kTfLite, ::tensorflow::tstring, int32_t>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
}

TEST_F(SetVariantOpTest, TestSetVariantOp_2x2x2) {
  auto op_wrapper = OpWrapperFriend<
      Runtime::kTfLite, TmplOp, Attr<AttrName<kAttrName1>, bool, int32_t>,
      Attr<AttrName<kAttrName2>, float, uint32_t>,
      Attr<AttrName<kAttrName3>, ::tensorflow::tstring, int64_t>>();

  auto map = CreateAttrMap<int32_t, uint32_t, ::tensorflow::tstring>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b = std::holds_alternative<
      TmplOp<Runtime::kTfLite, int32_t, uint32_t, ::tensorflow::tstring>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
}

TEST_F(SetVariantOpTest, TestSetVariantOp_2x1x3) {
  auto op_wrapper = OpWrapperFriend<
      Runtime::kTfLite, TmplOp, Attr<AttrName<kAttrName1>, bool, int32_t>,
      Attr<AttrName<kAttrName2>, float>,
      Attr<AttrName<kAttrName3>, ::tensorflow::tstring, int64_t, uint32_t>>();

  auto map = CreateAttrMap<int32_t, float, ::tensorflow::tstring>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b = std::holds_alternative<
      TmplOp<Runtime::kTfLite, int32_t, float, ::tensorflow::tstring>>(
      *op_wrapper.GetOp());
  EXPECT_TRUE(b);
}

TEST_F(SetVariantOpTest, TestSetVariantOp_4x4x6) {
  auto op_wrapper = OpWrapperFriend<
      Runtime::kTfLite, TmplOp,
      Attr<AttrName<kAttrName1>, bool, int32_t, uint32_t, int8_t>,
      Attr<AttrName<kAttrName2>, float, int16_t, int32_t, uint32_t>,
      Attr<AttrName<kAttrName3>, int8_t, uint8_t, int64_t, uint64_t, int32_t,
           uint32_t>>();

  auto map = CreateAttrMap<int32_t, float, uint32_t>();
  auto context = FakeInitContext(&map);
  EXPECT_OK(op_wrapper.Init(&context));

  bool b = std::holds_alternative<
      TmplOp<Runtime::kTfLite, int32_t, float, uint32_t>>(*op_wrapper.GetOp());
  EXPECT_TRUE(b);
}

}  // namespace
}  // namespace op_wrapper
}  // namespace shim
}  // namespace tflite
