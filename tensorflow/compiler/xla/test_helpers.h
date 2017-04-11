/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TEST_HELPERS_H_
#define TENSORFLOW_COMPILER_XLA_TEST_HELPERS_H_

#include <list>
#include <vector>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/test.h"

// This module contains a minimal subset of gmock functionality just
// sufficient to execute the currently existing tests.
namespace util {
class Status;
}  // namespace util

namespace xla {
template <typename T>
class Array2D;
class Literal;

namespace testing {

class AssertionResult {
 public:
  explicit AssertionResult(bool success) : success_(success) {}

  // Returns true iff the assertion succeeded.
  operator bool() const { return success_; }  // NOLINT

  // Returns the assertion's negation. Used with EXPECT/ASSERT_FALSE.
  AssertionResult operator!() const;

  // Returns the text streamed into this AssertionResult. Test assertions
  // use it when they fail (i.e., the predicate's outcome doesn't match the
  // assertion's expectation). When nothing has been streamed into the
  // object, returns an empty string.
  const char* message() const {
    return message_ != nullptr ? message_->c_str() : "";
  }

  // Streams a custom failure message into this object.
  template <typename T>
  AssertionResult& operator<<(const T& value) {
    AppendMessage(::testing::Message() << value);
    return *this;
  }

  // Allows streaming basic output manipulators such as endl or flush into
  // this object.
  AssertionResult& operator<<(
      std::ostream& (*basic_manipulator)(std::ostream& stream)) {
    AppendMessage(::testing::Message() << basic_manipulator);
    return *this;
  }

  // Copy operator.
  AssertionResult(const AssertionResult& ar);

  // Assignment operator.
  AssertionResult& operator=(const AssertionResult&);

 private:
  // Appends the contents of message to message_.
  void AppendMessage(const ::testing::Message& a_message) {
    if (message_ == nullptr) message_.reset(new std::string);
    message_->append(a_message.GetString().c_str());
  }

  bool success_ = false;

  // Stores the message describing the condition in case the
  // expectation construct is not satisfied with the predicate's
  // outcome.  Referenced via a pointer to avoid taking too much stack
  // frame space with test assertions.
  std::unique_ptr<std::string> message_;
};

AssertionResult AssertionFailure();

AssertionResult AssertionSuccess();

std::function<bool(tensorflow::StringPiece)> ContainsRegex(
    const tensorflow::StringPiece regex);

std::function<bool(tensorflow::StringPiece)> HasSubstr(
    const tensorflow::StringPiece part);

// Matcher for a vector of same-type values for which operator= is
// defined.
template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> VectorMatcher(
    const std::vector<T>& expected) {
  return [expected](const std::vector<T>& actual) -> AssertionResult {
    int len = expected.size();
    if (actual.size() != len) {
      return AssertionFailure() << "Actual values len of " << actual.size()
                                << " != expected.size " << len;
    }
    for (int i = 0; i < len; ++i) {
      if (actual[i] != expected[i]) {
        return AssertionFailure() << "Element " << i << " actual " << actual[i]
                                  << " != " << expected[i];
      }
    }
    return AssertionSuccess();
  };
}

// Approximate matcher for a vector of floats or similar.
template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)>
ApproxVectorMatcher(const std::vector<T>& expected, float abs_diff,
                    float rel_diff) {
  return [abs_diff, rel_diff,
          expected](const std::vector<T>& actual) -> AssertionResult {
    int len = expected.size();
    if (actual.size() != len) {
      AssertionResult ar = AssertionFailure() << "Actual values len of "
                                              << actual.size()
                                              << " != expected.size " << len;
      LOG(ERROR) << ar.message();
      return ar;
    }
    for (int i = 0; i < len; ++i) {
      T diff = actual[i] - expected[i];
      if (diff < 0) {
        diff *= -1;
      }
      if (diff > abs_diff) {
        T rdiff = (expected[i] != 0 ? diff / expected[i] : 0.0 * expected[i]);
        if (rdiff > rel_diff) {
          AssertionResult ar = AssertionFailure()
                               << "Element " << i << " actual " << actual[i]
                               << " != " << expected[i]
                               << "( abs_diff = " << diff
                               << ", rel_diff = " << rdiff << ")";
          LOG(ERROR) << ar.message();
          return ar;
        }
      }
    }
    return AssertionSuccess();
  };
}

// Matches a vector of same-type values against another, succeeding so
// long as they have the same length and every value in 'actual'
// matches one in 'expected.'  Does not verify an exhaustive
// one-to-one mapping between the two.
template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)>
UnorderedElementsAre(const std::vector<T>& expected) {
  return [expected](const std::vector<T>& actual) -> AssertionResult {
    if (actual.size() != expected.size()) {
      return AssertionFailure() << "sizes don't match";
    }
    for (auto a : actual) {
      bool found = false;
      for (auto e : expected) {
        if (a == e) {
          found = true;
          break;
        }
      }
      if (!found) {
        return AssertionFailure() << "actual element " << a
                                  << " not in expected";
      }
    }
    return AssertionSuccess();
  };
}

// Overloaded cover functions for UnorderedElementsAre, for the numbers
// of values used in practice.
template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> UnorderedMatcher(
    T a) {
  std::vector<T> expected;
  expected.push_back(a);
  return testing::UnorderedElementsAre<T>(expected);
}

template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> UnorderedMatcher(
    T a, T b) {
  std::vector<T> expected;
  expected.push_back(a);
  expected.push_back(b);
  return testing::UnorderedElementsAre<T>(expected);
}

template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> UnorderedMatcher(
    T a, T b, T c) {
  std::vector<T> expected;
  expected.push_back(a);
  expected.push_back(b);
  expected.push_back(c);
  return testing::UnorderedElementsAre<T>(expected);
}

template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> UnorderedMatcher(
    T a, T b, T c, T d) {
  std::vector<T> expected;
  expected.push_back(a);
  expected.push_back(b);
  expected.push_back(c);
  expected.push_back(d);
  return testing::UnorderedElementsAre<T>(expected);
}

template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> UnorderedMatcher(
    T a, T b, T c, T d, T e) {
  std::vector<T> expected;
  expected.push_back(a);
  expected.push_back(b);
  expected.push_back(c);
  expected.push_back(d);
  expected.push_back(e);
  return testing::UnorderedElementsAre<T>(expected);
}

template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> UnorderedMatcher(
    T a, T b, T c, T d, T e, T f) {
  std::vector<T> expected;
  expected.push_back(a);
  expected.push_back(b);
  expected.push_back(c);
  expected.push_back(d);
  expected.push_back(e);
  expected.push_back(f);
  return testing::UnorderedElementsAre<T>(expected);
}

// Overloaded cover functions for VectorMatcher for the numbers of
// elements used in practice.
template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> OrderedMatcher(
    T a) {
  std::vector<T> expected;
  expected.push_back(a);
  return testing::VectorMatcher<T>(expected);
}

template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> OrderedMatcher(
    T a, T b) {
  std::vector<T> expected;
  expected.push_back(a);
  expected.push_back(b);
  return testing::VectorMatcher<T>(expected);
}

template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> OrderedMatcher(
    T a, T b, T c) {
  std::vector<T> expected;
  expected.push_back(a);
  expected.push_back(b);
  expected.push_back(c);
  return testing::VectorMatcher<T>(expected);
}

template <typename T>
std::function<AssertionResult(const std::vector<T>& actual)> OrderedMatcher(
    T a, T b, T c, T d) {
  std::vector<T> expected;
  expected.push_back(a);
  expected.push_back(b);
  expected.push_back(c);
  expected.push_back(d);
  return testing::VectorMatcher<T>(expected);
}

// Convert a RepeatedField to a flat vector.
template <typename T>
std::vector<T> PBToVec(const tensorflow::protobuf::RepeatedField<T> rf) {
  return std::vector<T>(rf.begin(), rf.end());
}

// Convert a List to a flat vector.
template <typename T>
std::vector<T> ListToVec(const std::list<T>& l) {
  return std::vector<T>(l.begin(), l.end());
}

// Convert a Set to a flat vector.
template <typename T>
std::vector<T> SetToVec(const std::set<T>& c) {
  return std::vector<T>(c.begin(), c.end());
}

// Convert an Array to a flat vector.
template <typename T>
std::vector<T> Array2DToVec(const Array2D<T>& a) {
  return std::vector<T>(a.data(), a.data() + a.num_elements());
}

namespace internal_status {
inline const ::tensorflow::Status& GetStatus(
    const ::tensorflow::Status& status) {
  return status;
}

template <typename T>
inline const ::tensorflow::Status& GetStatus(const StatusOr<T>& status) {
  return status.status();
}
}  // namespace internal_status

}  // namespace testing
}  // namespace xla

// The following macros are similar to macros in gmock, but deliberately named
// differently in order to avoid conflicts in files which include both.

// Macros for testing the results of functions that return tensorflow::Status or
// StatusOr<T> (for any type T).
#define EXPECT_IS_OK(expression)      \
  EXPECT_EQ(tensorflow::Status::OK(), \
            xla::testing::internal_status::GetStatus(expression))
#undef ASSERT_IS_OK
#define ASSERT_IS_OK(expression)      \
  ASSERT_EQ(tensorflow::Status::OK(), \
            xla::testing::internal_status::GetStatus(expression))

// Macros that apply a Matcher to a Value, returning an
// AssertionResult which gets digested by a standard gunit macro.
#define EXPECT_MATCH(V, M) EXPECT_TRUE((M)((V)))
#define ASSERT_MATCH(V, M) ASSERT_TRUE(M(V))

#endif  // TENSORFLOW_COMPILER_XLA_TEST_HELPERS_H_
