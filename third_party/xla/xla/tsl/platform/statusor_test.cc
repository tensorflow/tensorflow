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

// Unit tests for StatusOr

#include "xla/tsl/platform/statusor.h"

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/config.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace tsl {
namespace {

class Base1 {
 public:
  virtual ~Base1() {}
  int pad_;
};

class Base2 {
 public:
  virtual ~Base2() {}
  int yetotherpad_;
};

class Derived : public Base1, public Base2 {
 public:
  ~Derived() override {}
  int evenmorepad_;
};

class CopyNoAssign {
 public:
  explicit CopyNoAssign(int value) : foo_(value) {}
  CopyNoAssign(const CopyNoAssign& other) : foo_(other.foo_) {}
  int foo_;

 private:
  const CopyNoAssign& operator=(const CopyNoAssign&);
};

class NoDefaultConstructor {
 public:
  explicit NoDefaultConstructor(int foo);
};

static_assert(!std::is_default_constructible<NoDefaultConstructor>(),
              "Should not be default-constructible.");

absl::StatusOr<std::unique_ptr<int>> ReturnUniquePtr() {
  // Uses implicit constructor from T&&
  return std::unique_ptr<int>(new int(0));
}

TEST(StatusOr, NullPointerStatusOr) {
  // As a very special case, null-plain-pointer StatusOr used to be an
  // error. Test that it no longer is.
  absl::StatusOr<int*> null_status(nullptr);
  EXPECT_TRUE(null_status.ok());
  EXPECT_EQ(null_status.value(), nullptr);
}

TEST(StatusOr, TestNoDefaultConstructorInitialization) {
  // Explicitly initialize it with an error code.
  absl::StatusOr<NoDefaultConstructor> statusor(errors::Cancelled(""));
  EXPECT_FALSE(statusor.ok());
  EXPECT_EQ(statusor.status().code(), absl::StatusCode::kCancelled);

  // Default construction of StatusOr initializes it with an UNKNOWN error code.
  absl::StatusOr<NoDefaultConstructor> statusor2;
  EXPECT_FALSE(statusor2.ok());
  EXPECT_EQ(statusor2.status().code(), absl::StatusCode::kUnknown);
}

TEST(StatusOr, TestMoveOnlyInitialization) {
  absl::StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.value());
  int* previous = thing.value().get();

  thing = ReturnUniquePtr();
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.value());
  EXPECT_NE(previous, thing.value().get());
}

TEST(StatusOr, TestMoveOnlyStatusCtr) {
  absl::StatusOr<std::unique_ptr<int>> thing(errors::Cancelled(""));
  ASSERT_FALSE(thing.ok());
}

TEST(StatusOr, TestMoveOnlyValueExtraction) {
  absl::StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  std::unique_ptr<int> ptr = std::move(thing).value();
  EXPECT_EQ(0, *ptr);

  thing = std::move(ptr);
  ptr = std::move(thing.value());
  EXPECT_EQ(0, *ptr);
}

TEST(StatusOr, TestMoveOnlyConversion) {
  absl::StatusOr<std::unique_ptr<const int>> const_thing(ReturnUniquePtr());
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.value());

  // Test rvalue converting assignment
  const int* const_previous = const_thing.value().get();
  const_thing = ReturnUniquePtr();
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.value());
  EXPECT_NE(const_previous, const_thing.value().get());
}

TEST(StatusOr, TestMoveOnlyVector) {
  // Sanity check that StatusOr<MoveOnly> works in vector.
  std::vector<absl::StatusOr<std::unique_ptr<int>>> vec;
  vec.push_back(ReturnUniquePtr());
  vec.resize(2);
  auto another_vec = std::move(vec);
  EXPECT_EQ(0, *another_vec[0].value());
  EXPECT_EQ(absl::StatusCode::kUnknown, another_vec[1].status().code());
}

TEST(StatusOr, TestMoveWithValuesAndErrors) {
  absl::StatusOr<std::string> status_or(std::string(1000, '0'));
  absl::StatusOr<std::string> value1(std::string(1000, '1'));
  absl::StatusOr<std::string> value2(std::string(1000, '2'));
  absl::StatusOr<std::string> error1(
      absl::Status(absl::StatusCode::kUnknown, "error1"));
  absl::StatusOr<std::string> error2(
      absl::Status(absl::StatusCode::kUnknown, "error2"));

  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '0'), status_or.value());

  // Overwrite the value in status_or with another value.
  status_or = std::move(value1);
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '1'), status_or.value());

  // Overwrite the value in status_or with an error.
  status_or = std::move(error1);
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error1", status_or.status().message());

  // Overwrite the error in status_or with another error.
  status_or = std::move(error2);
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error2", status_or.status().message());

  // Overwrite the error with a value.
  status_or = std::move(value2);
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '2'), status_or.value());
}

TEST(StatusOr, TestCopyWithValuesAndErrors) {
  absl::StatusOr<std::string> status_or(std::string(1000, '0'));
  absl::StatusOr<std::string> value1(std::string(1000, '1'));
  absl::StatusOr<std::string> value2(std::string(1000, '2'));
  absl::StatusOr<std::string> error1(
      absl::Status(absl::StatusCode::kUnknown, "error1"));
  absl::StatusOr<std::string> error2(
      absl::Status(absl::StatusCode::kUnknown, "error2"));

  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '0'), status_or.value());

  // Overwrite the value in status_or with another value.
  status_or = value1;
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '1'), status_or.value());

  // Overwrite the value in status_or with an error.
  status_or = error1;
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error1", status_or.status().message());

  // Overwrite the error in status_or with another error.
  status_or = error2;
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error2", status_or.status().message());

  // Overwrite the error with a value.
  status_or = value2;
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '2'), status_or.value());

  // Verify original values unchanged.
  EXPECT_EQ(std::string(1000, '1'), value1.value());
  EXPECT_EQ("error1", error1.status().message());
  EXPECT_EQ("error2", error2.status().message());
  EXPECT_EQ(std::string(1000, '2'), value2.value());
}

TEST(StatusOr, TestDefaultCtor) {
  absl::StatusOr<int> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), absl::StatusCode::kUnknown);
}

TEST(StatusOrDeathTest, TestDefaultCtorValue) {
  absl::StatusOr<int> thing;
#ifdef ABSL_HAVE_EXCEPTIONS
  try {
    thing.value();
    ADD_FAILURE()
        << "value() returned successfully while the access is illegal";
  } catch (absl::BadStatusOrAccess& ex) {
  }
#else
  EXPECT_DEATH(thing.value(), "");
#endif

  const absl::StatusOr<int> thing2;
#ifdef ABSL_HAVE_EXCEPTIONS
  try {
    thing.value();
    ADD_FAILURE()
        << "value() returned successfully while the access is illegal";
  } catch (absl::BadStatusOrAccess& ex) {
  }
#else
  EXPECT_DEATH(thing.value(), "");
#endif
}

TEST(StatusOr, TestStatusCtor) {
  absl::StatusOr<int> thing(absl::Status(absl::StatusCode::kCancelled, ""));
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), absl::StatusCode::kCancelled);
}

TEST(StatusOr, TestValueCtor) {
  const int kI = 4;
  const absl::StatusOr<int> thing(kI);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(kI, thing.value());
}

TEST(StatusOr, TestCopyCtorStatusOk) {
  const int kI = 4;
  const absl::StatusOr<int> original(kI);
  const absl::StatusOr<int> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.value(), copy.value());
}

TEST(StatusOr, TestCopyCtorStatusNotOk) {
  absl::StatusOr<int> original(absl::Status(absl::StatusCode::kCancelled, ""));
  absl::StatusOr<int> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestCopyCtorNonAssignable) {
  const int kI = 4;
  CopyNoAssign value(kI);
  absl::StatusOr<CopyNoAssign> original(value);
  absl::StatusOr<CopyNoAssign> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.value().foo_, copy.value().foo_);
}

TEST(StatusOr, TestCopyCtorStatusOKConverting) {
  const int kI = 4;
  absl::StatusOr<int> original(kI);
  absl::StatusOr<double> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_DOUBLE_EQ(original.value(), copy.value());
}

TEST(StatusOr, TestCopyCtorStatusNotOkConverting) {
  absl::StatusOr<int> original(absl::Status(absl::StatusCode::kCancelled, ""));
  absl::StatusOr<double> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestAssignmentStatusOk) {
  const int kI = 4;
  absl::StatusOr<int> source(kI);
  absl::StatusOr<int> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
  EXPECT_EQ(source.value(), target.value());
}

TEST(StatusOr, TestAssignmentStatusNotOk) {
  absl::StatusOr<int> source(absl::Status(absl::StatusCode::kCancelled, ""));
  absl::StatusOr<int> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
}

TEST(StatusOr, TestStatus) {
  absl::StatusOr<int> good(4);
  EXPECT_TRUE(good.ok());
  absl::StatusOr<int> bad(absl::Status(absl::StatusCode::kCancelled, ""));
  EXPECT_FALSE(bad.ok());
  EXPECT_EQ(bad.status(), absl::Status(absl::StatusCode::kCancelled, ""));
}

TEST(StatusOr, TestValue) {
  const int kI = 4;
  absl::StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.value());
}

TEST(StatusOr, TestValueConst) {
  const int kI = 4;
  const absl::StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.value());
}

TEST(StatusOrDeathTest, TestValueNotOk) {
  absl::StatusOr<int> thing(
      absl::Status(absl::StatusCode::kCancelled, "cancelled"));
#ifdef ABSL_HAVE_EXCEPTIONS
  try {
    thing.value();
    ADD_FAILURE()
        << "value() returned successfully while the access is illegal";
  } catch (absl::BadStatusOrAccess& ex) {
  }
#else
  EXPECT_DEATH(thing.value(), "cancelled");
#endif
}

TEST(StatusOrDeathTest, TestValueNotOkConst) {
  const absl::StatusOr<int> thing(absl::Status(absl::StatusCode::kUnknown, ""));
#ifdef ABSL_HAVE_EXCEPTIONS
  try {
    thing.value();
    ADD_FAILURE()
        << "value() returned successfully while the access is illegal";
  } catch (absl::BadStatusOrAccess& ex) {
  }
#else
  EXPECT_DEATH(thing.value(), "");
#endif
}

TEST(StatusOr, TestPointerDefaultCtor) {
  absl::StatusOr<int*> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), absl::StatusCode::kUnknown);
}

TEST(StatusOrDeathTest, TestPointerDefaultCtorValue) {
  absl::StatusOr<int*> thing;
#ifdef ABSL_HAVE_EXCEPTIONS
  try {
    thing.value();
    ADD_FAILURE()
        << "value() returned successfully while the access is illegal";
  } catch (absl::BadStatusOrAccess& ex) {
  }
#else
  EXPECT_DEATH(thing.value(), "");
#endif
}

TEST(StatusOr, TestPointerStatusCtor) {
  absl::StatusOr<int*> thing(absl::Status(absl::StatusCode::kCancelled, ""));
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status(), absl::Status(absl::StatusCode::kCancelled, ""));
}

TEST(StatusOr, TestPointerValueCtor) {
  const int kI = 4;
  absl::StatusOr<const int*> thing(&kI);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(&kI, thing.value());
}

TEST(StatusOr, TestPointerCopyCtorStatusOk) {
  const int kI = 0;
  absl::StatusOr<const int*> original(&kI);
  absl::StatusOr<const int*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.value(), copy.value());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOk) {
  absl::StatusOr<int*> original(absl::Status(absl::StatusCode::kCancelled, ""));
  absl::StatusOr<int*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestPointerCopyCtorStatusOKConverting) {
  Derived derived;
  absl::StatusOr<Derived*> original(&derived);
  absl::StatusOr<Base2*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(static_cast<const Base2*>(original.value()), copy.value());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOkConverting) {
  absl::StatusOr<Derived*> original(
      absl::Status(absl::StatusCode::kCancelled, ""));
  absl::StatusOr<Base2*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestPointerAssignmentStatusOk) {
  const int kI = 0;
  absl::StatusOr<const int*> source(&kI);
  absl::StatusOr<const int*> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
  EXPECT_EQ(source.value(), target.value());
}

TEST(StatusOr, TestPointerAssignmentStatusNotOk) {
  absl::StatusOr<int*> source(absl::Status(absl::StatusCode::kCancelled, ""));
  absl::StatusOr<int*> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
}

TEST(StatusOr, TestPointerStatus) {
  const int kI = 0;
  absl::StatusOr<const int*> good(&kI);
  EXPECT_TRUE(good.ok());
  absl::StatusOr<const int*> bad(
      absl::Status(absl::StatusCode::kCancelled, ""));
  EXPECT_EQ(bad.status(), absl::Status(absl::StatusCode::kCancelled, ""));
}

TEST(StatusOr, TestPointerValue) {
  const int kI = 0;
  absl::StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.value());
}

TEST(StatusOr, TestPointerValueConst) {
  const int kI = 0;
  const absl::StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.value());
}

TEST(StatusOr, TestArrowOperator) {
  absl::StatusOr<std::unique_ptr<int>> uptr = ReturnUniquePtr();
  EXPECT_EQ(*uptr->get(), 0);
}

TEST(StatusOr, TestStarOperator) {
  absl::StatusOr<std::unique_ptr<int>> uptr = ReturnUniquePtr();
  EXPECT_EQ(**uptr, 0);
}

TEST(StatusOr, TestStarOperatorDeath) {
  absl::StatusOr<Base1> error(
      absl::Status(absl::StatusCode::kCancelled, "cancelled"));
  EXPECT_DEATH(*error, "cancelled");
}

// NOTE(tucker): StatusOr does not support this kind
// of resize op.
// TEST(StatusOr, StatusOrVectorOfUniquePointerCanResize) {
//   using EvilType = std::vector<std::unique_ptr<int>>;
//   static_assert(std::is_copy_constructible<EvilType>::value, "");
//   std::vector<StatusOr<EvilType>> v(5);
//   v.reserve(v.capacity() + 10);
// }

static absl::StatusOr<int> MakeStatus() { return 100; }
// A factory to help us benchmark the various factory styles. All of
// the factory methods are marked as non-inlineable so as to more
// accurately simulate calling a factory for which you do not have
// visibility of implementation. Similarly, the value_ variable is
// marked volatile to prevent the compiler from getting too clever
// about detecting that the same value is used in all loop iterations.
template <typename T>
class BenchmarkFactory {
 public:
  // Construct a new factory. Allocate an object which will always
  // be the result of the factory methods.
  BenchmarkFactory() : value_(new T) {}

  // Destroy this factory, including the result value.
  ~BenchmarkFactory() { delete value_; }

  // A trivial factory that just returns the value. There is no status
  // object that could be returned to encapsulate an error
  T* TrivialFactory() TF_ATTRIBUTE_NOINLINE { return value_; }

  // A more sophisticated factory, which returns a status to indicate
  // the result of the operation. The factory result is populated into
  // the user provided pointer result.
  absl::Status ArgumentFactory(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = value_;
    return absl::OkStatus();
  }

  absl::Status ArgumentFactoryFail(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = nullptr;
    return absl::Status(absl::StatusCode::kCancelled, "");
  }

  absl::Status ArgumentFactoryFailShortMsg(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = nullptr;
    return absl::Status(absl::StatusCode::kInternal, "");
  }

  absl::Status ArgumentFactoryFailLongMsg(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = nullptr;
    return absl::Status(absl::StatusCode::kInternal,
                        "a big string of message junk that will never be read");
  }

  // A factory that returns a StatusOr<T*>. If the factory operation
  // is OK, then the StatusOr<T*> will hold a T*. Otherwise, it will
  // hold a status explaining the error.
  StatusOr<T*> StatusOrFactory() TF_ATTRIBUTE_NOINLINE {
    return static_cast<T*>(value_);
  }

  StatusOr<T*> StatusOrFactoryFail() TF_ATTRIBUTE_NOINLINE {
    return absl::Status(absl::StatusCode::kCancelled, "");
  }

  StatusOr<T*> StatusOrFactoryFailShortMsg() TF_ATTRIBUTE_NOINLINE {
    return absl::Status(absl::StatusCode::kInternal, "");
  }

  StatusOr<T*> StatusOrFactoryFailLongMsg() TF_ATTRIBUTE_NOINLINE {
    return absl::Status(absl::StatusCode::kInternal,
                        "a big string of message junk that will never be read");
  }

 private:
  T* volatile value_;
  BenchmarkFactory(const BenchmarkFactory&) = delete;
  void operator=(const BenchmarkFactory&) = delete;
};

// A simple type we use with the factory.
class BenchmarkType {
 public:
  BenchmarkType() {}
  virtual ~BenchmarkType() {}
  virtual void DoWork() TF_ATTRIBUTE_NOINLINE {}

 private:
  BenchmarkType(const BenchmarkType&) = delete;
  void operator=(const BenchmarkType&) = delete;
};

// Calibrate the amount of time spent just calling DoWork, since each of our
// tests will do this, we can subtract this out of benchmark results.
void BM_CalibrateWorkLoop(::testing::benchmark::State& state) {
  BenchmarkFactory<BenchmarkType> factory;
  BenchmarkType* result = factory.TrivialFactory();
  for (auto s : state) {
    if (result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_CalibrateWorkLoop);

// Measure the time taken to call into the factory, return the value,
// determine that it is OK, and invoke a trivial function.
void BM_TrivialFactory(::testing::benchmark::State& state) {
  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    BenchmarkType* result = factory.TrivialFactory();
    if (result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_TrivialFactory);

// Measure the time taken to call into the factory, providing an
// out-param for the result, evaluating the status result and the
// result pointer, and invoking the trivial function.
void BM_ArgumentFactory(::testing::benchmark::State& state) {
  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    BenchmarkType* result = nullptr;
    absl::Status status = factory.ArgumentFactory(&result);
    if (status.ok() && result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_ArgumentFactory);

// Measure the time to use the StatusOr<T*> factory, evaluate the result,
// and invoke the trivial function.
void BM_StatusOrFactory(::testing::benchmark::State& state) {
  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    absl::StatusOr<BenchmarkType*> result = factory.StatusOrFactory();
    if (result.ok()) {
      result.value()->DoWork();
    }
  }
}
BENCHMARK(BM_StatusOrFactory);

// Measure the time taken to call into the factory, providing an
// out-param for the result, evaluating the status result and the
// result pointer, and invoking the trivial function.
void BM_ArgumentFactoryFail(::testing::benchmark::State& state) {
  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    BenchmarkType* result = nullptr;
    absl::Status status = factory.ArgumentFactoryFail(&result);
    if (status.ok() && result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_ArgumentFactoryFail);

// Measure the time to use the StatusOr<T*> factory, evaluate the result,
// and invoke the trivial function.
void BM_StatusOrFactoryFail(::testing::benchmark::State& state) {
  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    absl::StatusOr<BenchmarkType*> result = factory.StatusOrFactoryFail();
    if (result.ok()) {
      result.value()->DoWork();
    }
  }
}
BENCHMARK(BM_StatusOrFactoryFail);

// Measure the time taken to call into the factory, providing an
// out-param for the result, evaluating the status result and the
// result pointer, and invoking the trivial function.
void BM_ArgumentFactoryFailShortMsg(::testing::benchmark::State& state) {
  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    BenchmarkType* result = nullptr;
    absl::Status status = factory.ArgumentFactoryFailShortMsg(&result);
    if (status.ok() && result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_ArgumentFactoryFailShortMsg);

// Measure the time to use the StatusOr<T*> factory, evaluate the result,
// and invoke the trivial function.
void BM_StatusOrFactoryFailShortMsg(::testing::benchmark::State& state) {
  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    absl::StatusOr<BenchmarkType*> result =
        factory.StatusOrFactoryFailShortMsg();
    if (result.ok()) {
      result.value()->DoWork();
    }
  }
}
BENCHMARK(BM_StatusOrFactoryFailShortMsg);

// Measure the time taken to call into the factory, providing an
// out-param for the result, evaluating the status result and the
// result pointer, and invoking the trivial function.
void BM_ArgumentFactoryFailLongMsg(::testing::benchmark::State& state) {
  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    BenchmarkType* result = nullptr;
    absl::Status status = factory.ArgumentFactoryFailLongMsg(&result);
    if (status.ok() && result != nullptr) {
      result->DoWork();
    }
  }
}
BENCHMARK(BM_ArgumentFactoryFailLongMsg);

// Measure the time to use the StatusOr<T*> factory, evaluate the result,
// and invoke the trivial function.
void BM_StatusOrFactoryFailLongMsg(::testing::benchmark::State& state) {
  BenchmarkFactory<BenchmarkType> factory;
  for (auto s : state) {
    absl::StatusOr<BenchmarkType*> result =
        factory.StatusOrFactoryFailLongMsg();
    if (result.ok()) {
      result.value()->DoWork();
    }
  }
}
BENCHMARK(BM_StatusOrFactoryFailLongMsg);

#if defined(PLATFORM_GOOGLE)

absl::StatusOr<int> GetError() {
  return absl::InvalidArgumentError("An invalid argument error");
}

absl::StatusOr<int> PropagateError() {
  TF_ASSIGN_OR_RETURN(int a, GetError());
  return a;
}

absl::StatusOr<int> PropagateError2() {
  TF_ASSIGN_OR_RETURN(int a, PropagateError());
  return a;
}

TEST(Status, StackTracePropagation) {
  absl::StatusOr<int> s = PropagateError2();
  auto sources = s.status().GetSourceLocations();
  ASSERT_EQ(sources.size(), 3);

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(
        sources[i].file_name(),
        "third_party/tensorflow/compiler/xla/tsl/platform/statusor_test.cc");
  }
}

#endif

}  // namespace
}  // namespace tsl
