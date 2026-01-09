<<<<<<< Conflict 1 of 1
%%%%%%% Changes from base to side #1
 /* Copyright 2025 The Abseil Authors & TensorFlow Authors. All Rights Reserved.
 
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
 
 #ifndef GOOGLETEST_WRAPPER_GMOCK_GMOCK_H_
 #define GOOGLETEST_WRAPPER_GMOCK_GMOCK_H_
 
 // gmock/gmock.h wrapper that also provides assert macros.
 //
 // These already exist in internal version of gmock, but upstream version
 // doesn't have them. We use this wrapper to make dependency translation when
 // exporting to OSS easier.
 //
 // - We want to use standard internal header and ASSERT_OK, EXPECT_OK macros
 //   when developing internally.
 // - We want the same macros to work externally, rather than having to add or
 //   strip TF_ prefix.
 // - We want the OSS export to still work after the export and header
 //   translation.
 // - We want to minimize the amount of patching third party projects to reduce
 //   maintenance overhead.
 // - To ensure the OSS patches cleanly apply onto internal repo, we need the
 //   header translation to be reversible, which requires 1:1 header mapping.
 //
 // To achieve this, we swap out gmock.h for this wrapper in all XLA code, which
 // should (TM) make ASSERT_OK/EXPECT_OK "just work" in all XLA tests.
 //
 // The only way to make this work without patching googletest and/or absl is to
 // make XLA *always* use this wrapper, and *never* directly depend on upstream
 // googletest.
 //
 // absl/status/status_matchers.h depends on gmock.h, so we can't simply add it
 // here. This causes either:
 //
 // - A circular dependency between this and absl - which bazel doesn't allow,
 // - absl dependency on the upstream gmock - which depending on the dependency
 //   graph structure may introduce upstream gmock include path *before* one
 //   defined in here, so we end up with *sometimes* including the wrong one and
 //   the entire idea of drop-in replacing gmock.h goes out of the window.
 
 #include_next "gmock/gmock.h"
 
 #include "absl/status/status.h"
 #include "absl/status/statusor.h"
 
 // Macros for testing the results of functions that return absl::Status or
 // absl::StatusOr<T> (for any type T).
 #define EXPECT_OK(expression) \
   EXPECT_THAT(expression, ::xla_testing::internal::IsOk())
 #define ASSERT_OK(expression) \
   ASSERT_THAT(expression, ::xla_testing::internal::IsOk())
 
 #define ASSERT_OK_AND_ASSIGN(lhs, rexpr)                            \
-  TF_ASSERT_OK_AND_ASSIGN_IMPL(                                     \
+  ASSERT_OK_AND_ASSIGN_IMPL(                                        \
       XLA_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), \
       lhs, rexpr);
 
 #define ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr) \
   auto statusor = (rexpr);                              \
   ASSERT_OK(statusor.status());                         \
   lhs = std::move(statusor).value()
 
 #define XLA_STATUS_MACROS_CONCAT_NAME(x, y) XLA_STATUS_MACROS_CONCAT_IMPL(x, y)
 #define XLA_STATUS_MACROS_CONCAT_IMPL(x, y) x##y
 
 namespace xla_testing {
 namespace internal {
 
 // DO NOT USE DIRECTLY. Use absl/status/status_matchers.h instead.
 inline const absl::Status& GetStatus(const absl::Status& status) {
   return status;
 }
 
 // DO NOT USE DIRECTLY. Use absl/status/status_matchers.h instead.
 template <typename T>
 inline const absl::Status& GetStatus(const absl::StatusOr<T>& status) {
   return status.status();
 }
 
 // DO NOT USE DIRECTLY. Use absl/status/status_matchers.h instead.
 //
 // Monomorphic implementation of matcher IsOk() for a given type T.
 // T can be Status, StatusOr<>, or a reference to either of them.
 template <typename T>
 class MonoIsOkMatcherImpl : public ::testing::MatcherInterface<T> {
  public:
   void DescribeTo(std::ostream* os) const override { *os << "is OK"; }
   void DescribeNegationTo(std::ostream* os) const override {
     *os << "is not OK";
   }
   bool MatchAndExplain(T actual_value,
                        ::testing::MatchResultListener*) const override {
     return GetStatus(actual_value).ok();
   }
 };
 
 // DO NOT USE DIRECTLY. Use absl/status/status_matchers.h instead.
 //
 // Implements IsOk() as a polymorphic matcher.
 class IsOkMatcher {
  public:
   template <typename T>
   /*implicit*/ operator ::testing::Matcher<T>() const {  // NOLINT
     return ::testing::Matcher<T>(new MonoIsOkMatcherImpl<const T&>());
   }
 };
 
 // DO NOT USE DIRECTLY. Use absl/status/status_matchers.h instead.
 //
 // Returns a gMock matcher that matches a Status or StatusOr<> which is OK.
 inline ::xla_testing::internal::IsOkMatcher IsOk() {
   return ::xla_testing::internal::IsOkMatcher();
 }
 
 }  // namespace internal
 }  // namespace xla_testing
 
 #endif  // GOOGLETEST_WRAPPER_GMOCK_GMOCK_H_
+++++++ Contents of side #2
>>>>>>> Conflict 1 of 1 ends
