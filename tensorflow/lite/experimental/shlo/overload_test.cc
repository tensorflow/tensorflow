/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/shlo/overload.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace {

TEST(OverloadTest, DispatchConsidersTypeWithAutoFallback) {
  auto overloaded = shlo_ref::Overload(
      [](int v) -> std::string { return absl::StrCat("int ", v); },        //
      [](double v) -> std::string { return absl::StrCat("double ", v); },  //
      [](const char* v) -> std::string {                                   //
        return absl::StrCat("const char* ", v);                            //
      },                                                                   //
      [](auto v) -> std::string { return absl::StrCat("auto ", v); }       //
  );
  EXPECT_EQ("int 1", overloaded(1));
  EXPECT_EQ("double 2.5", overloaded(2.5));
  EXPECT_EQ("const char* hello", overloaded("hello"));
  EXPECT_EQ("auto 1.5", overloaded(1.5f));
}

TEST(OverloadTest, DispatchConsidersNumberOfArguments) {
  auto overloaded = shlo_ref::Overload(             //
      [](int a) { return a + 1; },                  //
      [](int a, int b) { return a * b; },           //
      []() -> absl::string_view { return "none"; }  //
  );
  EXPECT_EQ(3, overloaded(2));
  EXPECT_EQ(21, overloaded(3, 7));
  EXPECT_EQ("none", overloaded());
}

TEST(OverloadTest, SupportsConstantEvaluation) {
  auto overloaded = shlo_ref::Overload(             //
      [](int a) { return a + 1; },                  //
      [](int a, int b) { return a * b; },           //
      []() -> absl::string_view { return "none"; }  //
  );
  static_assert(overloaded() == "none");
  static_assert(overloaded(2) == 3);
  static_assert(overloaded(3, 7) == 21);
}

TEST(OverloadTest, PropogatesDefaults) {
  auto overloaded = shlo_ref::Overload(        //
      [](int a, int b = 5) { return a * b; },  //
      [](double c) { return c; }               //
  );

  EXPECT_EQ(21, overloaded(3, 7));
  EXPECT_EQ(35, overloaded(7));
  EXPECT_EQ(2.5, overloaded(2.5));
}

TEST(OverloadTest, AmbiguousWithDefaultsNotInvocable) {
  auto overloaded = shlo_ref::Overload(        //
      [](int a, int b = 5) { return a * b; },  //
      [](int c) { return c; }                  //
  );
  static_assert(!std::is_invocable_v<decltype(overloaded), int>);
  static_assert(std::is_invocable_v<decltype(overloaded), int, int>);
}

TEST(OverloadTest, AmbiguousDuplicatesNotInvocable) {
  auto overloaded = shlo_ref::Overload(  //
      [](int a) { return a; },           //
      [](int c) { return c; }            //
  );
  static_assert(!std::is_invocable_v<decltype(overloaded), int>);
}

TEST(OverloadTest, AmbiguousConversionNotInvocable) {
  auto overloaded = shlo_ref::Overload(  //
      [](uint16_t a) { return a; },      //
      [](uint64_t c) { return c; }       //
  );
  static_assert(!std::is_invocable_v<decltype(overloaded), int>);
}

TEST(OverloadTest, AmbiguousConversionWithAutoNotInvocable) {
  auto overloaded = shlo_ref::Overload(  //
      [](auto a) { return a; },          //
      [](auto c) { return c; }           //
  );
  static_assert(!std::is_invocable_v<decltype(overloaded), int>);
}

TEST(OverloadTest, DispatchConsidersSfinae) {
  auto overloaded = shlo_ref::Overload(                //
      [](auto a) -> decltype(a + 1) { return a + 1; }  //
  );
  static_assert(std::is_invocable_v<decltype(overloaded), int>);
  static_assert(!std::is_invocable_v<decltype(overloaded), std::string>);
}

TEST(OverloadTest, VariantVisitDispatchesCorrectly) {
  std::variant<int, double, std::string> v(1);
  auto overloaded = shlo_ref::Overload(
      [](int) -> absl::string_view { return "int"; },                   //
      [](double) -> absl::string_view { return "double"; },             //
      [](const std::string&) -> absl::string_view { return "string"; }  //
  );
  EXPECT_EQ("int", std::visit(overloaded, v));
  v = 1.1;
  EXPECT_EQ("double", std::visit(overloaded, v));
  v = "hello";
  EXPECT_EQ("string", std::visit(overloaded, v));
}

TEST(OverloadTest, VariantVisitWithAutoFallbackDispatchesCorrectly) {
  std::variant<std::string, int32_t, int64_t> v(int32_t{1});
  auto overloaded =
      shlo_ref::Overload([](const std::string& s) { return s.size(); },  //
                         [](const auto& s) { return sizeof(s); }         //
      );
  EXPECT_EQ(4, std::visit(overloaded, v));
  v = int64_t{1};
  EXPECT_EQ(8, std::visit(overloaded, v));
  v = std::string("hello");
  EXPECT_EQ(5, std::visit(overloaded, v));
}

}  // namespace
