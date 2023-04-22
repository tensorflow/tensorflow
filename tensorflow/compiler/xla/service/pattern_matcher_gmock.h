/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_GMOCK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_GMOCK_H_

#include <ostream>
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

namespace pattern_matcher_gmock_detail {
template <typename Pattern>
class GmockMatcher {
 public:
  explicit GmockMatcher(Pattern p) : pattern_(std::move(p)) {}

  // In service of better error messages, list out the overloads explicitly
  // rather than just using a template.  gMock's polymorphism plus
  // pattern_matcher yields some pretty gnarly stuff.
  bool MatchAndExplain(const Layout& l,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(&l, listener);
  }
  bool MatchAndExplain(const Layout* l,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(l, listener);
  }

  bool MatchAndExplain(const Shape& s,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(&s, listener);
  }
  bool MatchAndExplain(const Shape* s,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(s, listener);
  }

  bool MatchAndExplain(const HloInstruction& instr,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(&instr, listener);
  }
  bool MatchAndExplain(const HloInstruction* instr,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainImpl(instr, listener);
  }

  void DescribeTo(std::ostream* os) const { pattern_.DescribeTo(os); }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "is NOT: ";
    DescribeTo(os);
  }

 private:
  template <typename T>
  bool MatchAndExplainImpl(const T* t,
                           ::testing::MatchResultListener* listener) const {
    MatchOption options{/*.capture=*/true, /*.explain_os=*/listener->stream()};
    return Match(t, pattern_, options);
  }

  Pattern pattern_;
};
}  // namespace pattern_matcher_gmock_detail

template <typename Pattern>
::testing::PolymorphicMatcher<
    pattern_matcher_gmock_detail::GmockMatcher<Pattern>>
GmockMatch(Pattern&& p) {
  return ::testing::MakePolymorphicMatcher(
      pattern_matcher_gmock_detail::GmockMatcher<Pattern>(
          std::forward<Pattern>(p)));
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_GMOCK_H_
