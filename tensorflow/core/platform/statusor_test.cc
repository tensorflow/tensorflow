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

#include "tensorflow/core/platform/statusor.h"

#include <memory>
#include <type_traits>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
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

StatusOr<std::unique_ptr<int>> ReturnUniquePtr() {
  // Uses implicit constructor from T&&
  return std::unique_ptr<int>(new int(0));
}

TEST(StatusOr, ElementType) {
  static_assert(std::is_same<StatusOr<int>::element_type, int>(), "");
  static_assert(std::is_same<StatusOr<char>::element_type, char>(), "");
}

TEST(StatusOr, NullPointerStatusOr) {
  // As a very special case, null-plain-pointer StatusOr used to be an
  // error. Test that it no longer is.
  StatusOr<int*> null_status(nullptr);
  EXPECT_TRUE(null_status.ok());
  EXPECT_EQ(null_status.ValueOrDie(), nullptr);
}

TEST(StatusOr, TestNoDefaultConstructorInitialization) {
  // Explicitly initialize it with an error code.
  StatusOr<NoDefaultConstructor> statusor(tensorflow::errors::Cancelled(""));
  EXPECT_FALSE(statusor.ok());
  EXPECT_EQ(statusor.status().code(), tensorflow::error::CANCELLED);

  // Default construction of StatusOr initializes it with an UNKNOWN error code.
  StatusOr<NoDefaultConstructor> statusor2;
  EXPECT_FALSE(statusor2.ok());
  EXPECT_EQ(statusor2.status().code(), tensorflow::error::UNKNOWN);
}

TEST(StatusOr, TestMoveOnlyInitialization) {
  StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.ValueOrDie());
  int* previous = thing.ValueOrDie().get();

  thing = ReturnUniquePtr();
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.ValueOrDie());
  EXPECT_NE(previous, thing.ValueOrDie().get());
}

TEST(StatusOr, TestMoveOnlyStatusCtr) {
  StatusOr<std::unique_ptr<int>> thing(tensorflow::errors::Cancelled(""));
  ASSERT_FALSE(thing.ok());
}

TEST(StatusOr, TestMoveOnlyValueExtraction) {
  StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  std::unique_ptr<int> ptr = thing.ConsumeValueOrDie();
  EXPECT_EQ(0, *ptr);

  thing = std::move(ptr);
  ptr = std::move(thing.ValueOrDie());
  EXPECT_EQ(0, *ptr);
}

TEST(StatusOr, TestMoveOnlyConversion) {
  StatusOr<std::unique_ptr<const int>> const_thing(ReturnUniquePtr());
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.ValueOrDie());

  // Test rvalue converting assignment
  const int* const_previous = const_thing.ValueOrDie().get();
  const_thing = ReturnUniquePtr();
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.ValueOrDie());
  EXPECT_NE(const_previous, const_thing.ValueOrDie().get());
}

TEST(StatusOr, TestMoveOnlyVector) {
  // Sanity check that StatusOr<MoveOnly> works in vector.
  std::vector<StatusOr<std::unique_ptr<int>>> vec;
  vec.push_back(ReturnUniquePtr());
  vec.resize(2);
  auto another_vec = std::move(vec);
  EXPECT_EQ(0, *another_vec[0].ValueOrDie());
  EXPECT_EQ(tensorflow::error::UNKNOWN, another_vec[1].status().code());
}

TEST(StatusOr, TestMoveWithValuesAndErrors) {
  StatusOr<std::string> status_or(std::string(1000, '0'));
  StatusOr<std::string> value1(std::string(1000, '1'));
  StatusOr<std::string> value2(std::string(1000, '2'));
  StatusOr<std::string> error1(Status(tensorflow::error::UNKNOWN, "error1"));
  StatusOr<std::string> error2(Status(tensorflow::error::UNKNOWN, "error2"));

  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '0'), status_or.ValueOrDie());

  // Overwrite the value in status_or with another value.
  status_or = std::move(value1);
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '1'), status_or.ValueOrDie());

  // Overwrite the value in status_or with an error.
  status_or = std::move(error1);
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error1", status_or.status().error_message());

  // Overwrite the error in status_or with another error.
  status_or = std::move(error2);
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error2", status_or.status().error_message());

  // Overwrite the error with a value.
  status_or = std::move(value2);
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '2'), status_or.ValueOrDie());
}

TEST(StatusOr, TestCopyWithValuesAndErrors) {
  StatusOr<std::string> status_or(std::string(1000, '0'));
  StatusOr<std::string> value1(std::string(1000, '1'));
  StatusOr<std::string> value2(std::string(1000, '2'));
  StatusOr<std::string> error1(Status(tensorflow::error::UNKNOWN, "error1"));
  StatusOr<std::string> error2(Status(tensorflow::error::UNKNOWN, "error2"));

  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '0'), status_or.ValueOrDie());

  // Overwrite the value in status_or with another value.
  status_or = value1;
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '1'), status_or.ValueOrDie());

  // Overwrite the value in status_or with an error.
  status_or = error1;
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error1", status_or.status().error_message());

  // Overwrite the error in status_or with another error.
  status_or = error2;
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ("error2", status_or.status().error_message());

  // Overwrite the error with a value.
  status_or = value2;
  ASSERT_TRUE(status_or.ok());
  EXPECT_EQ(std::string(1000, '2'), status_or.ValueOrDie());

  // Verify original values unchanged.
  EXPECT_EQ(std::string(1000, '1'), value1.ValueOrDie());
  EXPECT_EQ("error1", error1.status().error_message());
  EXPECT_EQ("error2", error2.status().error_message());
  EXPECT_EQ(std::string(1000, '2'), value2.ValueOrDie());
}

TEST(StatusOr, TestDefaultCtor) {
  StatusOr<int> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), tensorflow::error::UNKNOWN);
}

TEST(StatusOrDeathTest, TestDefaultCtorValue) {
  StatusOr<int> thing;
  EXPECT_DEATH(thing.ValueOrDie(), "");

  const StatusOr<int> thing2;
  EXPECT_DEATH(thing.ValueOrDie(), "");
}

TEST(StatusOr, TestStatusCtor) {
  StatusOr<int> thing(Status(tensorflow::error::CANCELLED, ""));
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), tensorflow::error::CANCELLED);
}

TEST(StatusOr, TestValueCtor) {
  const int kI = 4;
  const StatusOr<int> thing(kI);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusOk) {
  const int kI = 4;
  const StatusOr<int> original(kI);
  const StatusOr<int> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusNotOk) {
  StatusOr<int> original(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<int> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestCopyCtorNonAssignable) {
  const int kI = 4;
  CopyNoAssign value(kI);
  StatusOr<CopyNoAssign> original(value);
  StatusOr<CopyNoAssign> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.ValueOrDie().foo_, copy.ValueOrDie().foo_);
}

TEST(StatusOr, TestCopyCtorStatusOKConverting) {
  const int kI = 4;
  StatusOr<int> original(kI);
  StatusOr<double> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_DOUBLE_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusNotOkConverting) {
  StatusOr<int> original(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<double> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestAssignmentStatusOk) {
  const int kI = 4;
  StatusOr<int> source(kI);
  StatusOr<int> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
  EXPECT_EQ(source.ValueOrDie(), target.ValueOrDie());
}

TEST(StatusOr, TestAssignmentStatusNotOk) {
  StatusOr<int> source(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<int> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
}

TEST(StatusOr, TestStatus) {
  StatusOr<int> good(4);
  EXPECT_TRUE(good.ok());
  StatusOr<int> bad(Status(tensorflow::error::CANCELLED, ""));
  EXPECT_FALSE(bad.ok());
  EXPECT_EQ(bad.status(), Status(tensorflow::error::CANCELLED, ""));
}

TEST(StatusOr, TestValue) {
  const int kI = 4;
  StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOr, TestValueConst) {
  const int kI = 4;
  const StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOrDeathTest, TestValueNotOk) {
  StatusOr<int> thing(Status(tensorflow::error::CANCELLED, "cancelled"));
  EXPECT_DEATH(thing.ValueOrDie(), "cancelled");
}

TEST(StatusOrDeathTest, TestValueNotOkConst) {
  const StatusOr<int> thing(Status(tensorflow::error::UNKNOWN, ""));
  EXPECT_DEATH(thing.ValueOrDie(), "");
}

TEST(StatusOr, TestPointerDefaultCtor) {
  StatusOr<int*> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status().code(), tensorflow::error::UNKNOWN);
}

TEST(StatusOrDeathTest, TestPointerDefaultCtorValue) {
  StatusOr<int*> thing;
  EXPECT_DEATH(thing.ValueOrDie(), "");
}

TEST(StatusOr, TestPointerStatusCtor) {
  StatusOr<int*> thing(Status(tensorflow::error::CANCELLED, ""));
  EXPECT_FALSE(thing.ok());
  EXPECT_EQ(thing.status(), Status(tensorflow::error::CANCELLED, ""));
}

TEST(StatusOr, TestPointerValueCtor) {
  const int kI = 4;
  StatusOr<const int*> thing(&kI);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusOk) {
  const int kI = 0;
  StatusOr<const int*> original(&kI);
  StatusOr<const int*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOk) {
  StatusOr<int*> original(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<int*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestPointerCopyCtorStatusOKConverting) {
  Derived derived;
  StatusOr<Derived*> original(&derived);
  StatusOr<Base2*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
  EXPECT_EQ(static_cast<const Base2*>(original.ValueOrDie()),
            copy.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOkConverting) {
  StatusOr<Derived*> original(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<Base2*> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestPointerAssignmentStatusOk) {
  const int kI = 0;
  StatusOr<const int*> source(&kI);
  StatusOr<const int*> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
  EXPECT_EQ(source.ValueOrDie(), target.ValueOrDie());
}

TEST(StatusOr, TestPointerAssignmentStatusNotOk) {
  StatusOr<int*> source(Status(tensorflow::error::CANCELLED, ""));
  StatusOr<int*> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
}

TEST(StatusOr, TestPointerStatus) {
  const int kI = 0;
  StatusOr<const int*> good(&kI);
  EXPECT_TRUE(good.ok());
  StatusOr<const int*> bad(Status(tensorflow::error::CANCELLED, ""));
  EXPECT_EQ(bad.status(), Status(tensorflow::error::CANCELLED, ""));
}

TEST(StatusOr, TestPointerValue) {
  const int kI = 0;
  StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOr, TestPointerValueConst) {
  const int kI = 0;
  const StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOr, TestArrowOperator) {
  StatusOr<std::unique_ptr<int>> uptr = ReturnUniquePtr();
  EXPECT_EQ(*uptr->get(), 0);
}

TEST(StatusOr, TestArrowOperatorNotOk) {
  StatusOr<Base1> error(Status(tensorflow::error::CANCELLED, "cancelled"));
  EXPECT_DEATH(error->pad_++, "cancelled");
}

TEST(StatusOr, TestStarOperator) {
  StatusOr<std::unique_ptr<int>> uptr = ReturnUniquePtr();
  EXPECT_EQ(**uptr, 0);
}

TEST(StatusOr, TestStarOperatorDeath) {
  StatusOr<Base1> error(Status(tensorflow::error::CANCELLED, "cancelled"));
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

TEST(StatusOrDeathTest, TestPointerValueNotOk) {
  StatusOr<int*> thing(Status(tensorflow::error::CANCELLED, "cancelled"));
  EXPECT_DEATH(thing.ValueOrDie(), "cancelled");
}

TEST(StatusOrDeathTest, TestPointerValueNotOkConst) {
  const StatusOr<int*> thing(Status(tensorflow::error::CANCELLED, "cancelled"));
  EXPECT_DEATH(thing.ValueOrDie(), "cancelled");
}

static StatusOr<int> MakeStatus() { return 100; }
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
  Status ArgumentFactory(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = value_;
    return OkStatus();
  }

  Status ArgumentFactoryFail(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = nullptr;
    return Status(tensorflow::error::CANCELLED, "");
  }

  Status ArgumentFactoryFailShortMsg(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = nullptr;
    return Status(::tensorflow::error::INTERNAL, "");
  }

  Status ArgumentFactoryFailLongMsg(T** result) TF_ATTRIBUTE_NOINLINE {
    *result = nullptr;
    return Status(::tensorflow::error::INTERNAL,
                  "a big string of message junk that will never be read");
  }

  // A factory that returns a StatusOr<T*>. If the factory operation
  // is OK, then the StatusOr<T*> will hold a T*. Otherwise, it will
  // hold a status explaining the error.
  StatusOr<T*> StatusOrFactory() TF_ATTRIBUTE_NOINLINE {
    return static_cast<T*>(value_);
  }

  StatusOr<T*> StatusOrFactoryFail() TF_ATTRIBUTE_NOINLINE {
    return Status(tensorflow::error::CANCELLED, "");
  }

  StatusOr<T*> StatusOrFactoryFailShortMsg() TF_ATTRIBUTE_NOINLINE {
    return Status(::tensorflow::error::INTERNAL, "");
  }

  StatusOr<T*> StatusOrFactoryFailLongMsg() TF_ATTRIBUTE_NOINLINE {
    return Status(::tensorflow::error::INTERNAL,
                  "a big string of message junk that will never be read");
  }

 private:
  T* volatile value_;
  TF_DISALLOW_COPY_AND_ASSIGN(BenchmarkFactory);
};

// A simple type we use with the factory.
class BenchmarkType {
 public:
  BenchmarkType() {}
  virtual ~BenchmarkType() {}
  virtual void DoWork() TF_ATTRIBUTE_NOINLINE {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BenchmarkType);
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
    Status status = factory.ArgumentFactory(&result);
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
    StatusOr<BenchmarkType*> result = factory.StatusOrFactory();
    if (result.ok()) {
      result.ValueOrDie()->DoWork();
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
    Status status = factory.ArgumentFactoryFail(&result);
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
    StatusOr<BenchmarkType*> result = factory.StatusOrFactoryFail();
    if (result.ok()) {
      result.ValueOrDie()->DoWork();
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
    Status status = factory.ArgumentFactoryFailShortMsg(&result);
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
    StatusOr<BenchmarkType*> result = factory.StatusOrFactoryFailShortMsg();
    if (result.ok()) {
      result.ValueOrDie()->DoWork();
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
    Status status = factory.ArgumentFactoryFailLongMsg(&result);
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
    StatusOr<BenchmarkType*> result = factory.StatusOrFactoryFailLongMsg();
    if (result.ok()) {
      result.ValueOrDie()->DoWork();
    }
  }
}
BENCHMARK(BM_StatusOrFactoryFailLongMsg);

}  // namespace
}  // namespace tensorflow
