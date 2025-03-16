// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_shared_library.h"

#include <dlfcn.h>

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

using testing::Eq;
using testing::NotNull;
using testing::StrEq;
using testing::litert::IsError;

namespace litert {
namespace {

extern "C" {

const char* TestFunction() { return "local_test_function"; }

}  //  extern "C"

TEST(RtldFlagsTest, FlagFactoryWorks) {
  EXPECT_THAT(static_cast<int>(RtldFlags::Now()), Eq(RTLD_NOW));
  EXPECT_THAT(static_cast<int>(RtldFlags::Lazy()), Eq(RTLD_LAZY));
  EXPECT_THAT(static_cast<int>(RtldFlags::Lazy().Global()),
              Eq(RTLD_LAZY | RTLD_GLOBAL));
  EXPECT_THAT(static_cast<int>(RtldFlags::Lazy().Local()),
              Eq(RTLD_LAZY | RTLD_LOCAL));
  EXPECT_THAT(static_cast<int>(RtldFlags::Lazy().NoDelete()),
              Eq(RTLD_LAZY | RTLD_NODELETE));
  EXPECT_THAT(static_cast<int>(RtldFlags::Lazy().NoLoad()),
              Eq(RTLD_LAZY | RTLD_NOLOAD));
  EXPECT_THAT(static_cast<int>(RtldFlags::Lazy().DeepBind()),
              Eq(RTLD_LAZY | RTLD_DEEPBIND));
}

TEST(SharedLibraryTest, LoadRtldDefaultWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      SharedLibrary lib,
      SharedLibrary::Load(RtldFlags::kDefault, RtldFlags::Now().Local()));

  EXPECT_THAT(lib.Path(), StrEq(""));
  EXPECT_EQ(lib.Handle(), RTLD_DEFAULT);

  auto maybe_test_function =
      lib.LookupSymbol<decltype(&TestFunction)>("TestFunction");
  if (!maybe_test_function.HasValue()) {
    GTEST_SKIP() << "TestFunction symbol was stripped from binary.";
  }

  decltype(&TestFunction) test_function = maybe_test_function.Value();
  ASSERT_NE(test_function, nullptr);
  EXPECT_THAT(test_function(), StrEq(TestFunction()));
}

TEST(SharedLibraryTest, LoadRtldNextWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      SharedLibrary lib,
      SharedLibrary::Load(RtldFlags::kNext, RtldFlags::Now().Local()));

  EXPECT_THAT(lib.Path(), StrEq(""));
  EXPECT_EQ(lib.Handle(), RTLD_NEXT);
}

TEST(SharedLibraryTest, LoadEmptyPathFails) {
  EXPECT_THAT(SharedLibrary::Load("", RtldFlags::Now().Local()), IsError());
}

TEST(SharedLibraryTest, LoadPathWorks) {
  const std::string lib_path = absl::StrCat(
      "third_party/tensorflow/lite/experimental/litert/cc/"
      "test_shared_library.so");
  LITERT_ASSERT_OK_AND_ASSIGN(
      SharedLibrary lib,
      SharedLibrary::Load(lib_path, RtldFlags::Now().Local()));

  EXPECT_TRUE(lib.Loaded());
  EXPECT_THAT(lib.Path(), StrEq(lib_path));
  EXPECT_THAT(lib.Handle(), NotNull());

  using TestFunctionSignature = char* (*)();

  LITERT_ASSERT_OK_AND_ASSIGN(TestFunctionSignature test_function,
                              lib.LookupSymbol<char* (*)()>("TestFunction"));
  ASSERT_NE(test_function, nullptr);
  EXPECT_THAT(test_function(), StrEq("test_shared_library"));

  lib.Close();
  EXPECT_THAT(lib.Path(), StrEq(""));
  EXPECT_FALSE(lib.Loaded());
}

TEST(SharedLibraryTest, ConstructionAndAssignmentWork) {
  const std::string lib_path = absl::StrCat(
      "third_party/tensorflow/lite/experimental/litert/cc/"
      "test_shared_library.so");
  LITERT_ASSERT_OK_AND_ASSIGN(
      SharedLibrary lib,
      SharedLibrary::Load(lib_path, RtldFlags::Now().Local()));

  const void* const lib_handle = lib.Handle();

  SharedLibrary lib2(std::move(lib));

  // NOLINTBEGIN(bugprone-use-after-move): Tests that moving clears up the
  // object.
  EXPECT_THAT(lib.Path(), StrEq(""));
  EXPECT_FALSE(lib.Loaded());

  EXPECT_TRUE(lib2.Loaded());
  EXPECT_THAT(lib2.Path(), StrEq(lib_path));
  EXPECT_THAT(lib2.Handle(), Eq(lib_handle));

  lib = std::move(lib2);
  EXPECT_THAT(lib2.Path(), StrEq(""));
  EXPECT_FALSE(lib2.Loaded());

  EXPECT_TRUE(lib.Loaded());
  EXPECT_THAT(lib.Path(), StrEq(lib_path));
  EXPECT_THAT(lib.Handle(), Eq(lib_handle));
  // NOLINTEND(bugprone-use-after-move)
}

}  // namespace
}  // namespace litert
