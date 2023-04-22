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
#include "tensorflow/cc/experimental/libtf/object.h"

#include <cstdint>

#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/cc/experimental/libtf/value_iostream.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {

TEST(ObjectTest, TestDictionary) {
  Dictionary foo;
  foo.Set(String("a"), Integer(33));
  foo.Set(String("b"), Integer(33));
  EXPECT_EQ(foo.Get<Integer>(String("b"))->get(), 33);
}

TEST(ObjectTest, TestTuple) {
  Tuple foo(String("a"), Integer(33), Float(10.f));
  EXPECT_EQ(foo.size(), 3);
  EXPECT_EQ(foo.Get<Integer>(1)->get(), 33);
}

TEST(ObjectTest, TestList) {
  List l;
  EXPECT_EQ(l.size(), 0);
  l.append(Integer(3));
  EXPECT_EQ(l.Get<Integer>(0)->get(), 3);
  EXPECT_EQ(l.size(), 1);
}

TaggedValue AddIntegers(TaggedValue args_, TaggedValue kwargs_) {
  auto& args = args_.tuple();
  // auto& kwargs = kwargs_.dict();
  return TaggedValue(args[0].i64() + args[1].i64());
}

TEST(ObjectTest, TestCast) {
  Integer i(3);
  auto result = Cast<String>(i);
  ASSERT_TRUE(!result.ok());
}

TEST(ObjectTest, TestCall) {
  TaggedValue add_func(AddIntegers);
  Callable add(add_func);
  TF_ASSERT_OK_AND_ASSIGN(Integer i,
                          add.Call<Integer>(Integer(1), Integer(10)));
  EXPECT_EQ(i.get(), 11);

  TF_ASSERT_OK_AND_ASSIGN(
      Integer i2, add.Call<Integer>(1, Integer(10), KeywordArg("foo") = 3));
  EXPECT_EQ(i2.get(), 11);
}

TEST(ObjectTest, MakeObject) {
  // TaggedValue func(f);
  Object parent;
  parent.Set(String("test3"), Integer(3));
  Object child;
  child.Set(String("test1"), Integer(1));
  child.Set(String("test2"), Integer(2));
  child.Set(*Object::parent_, parent);
  EXPECT_EQ(child.Get<Integer>(String("test1"))->get(), 1);
  EXPECT_EQ(child.Get<Integer>(String("test2"))->get(), 2);
  EXPECT_EQ(child.Get<Integer>(String("test3"))->get(), 3);
  ASSERT_FALSE(child.Get<Integer>(String("test4")).status().ok());
  TF_ASSERT_OK(child.Get(String("test3")).status());
}

TEST(ObjectTest, CallFunctionOnObject) {
  Object module;
  module.Set(String("add"), Callable(TaggedValue(AddIntegers)));
  TF_ASSERT_OK_AND_ASSIGN(Callable method, module.Get<Callable>(String("add")));

  TF_ASSERT_OK_AND_ASSIGN(Integer val, method.Call<Integer>(1, 2));
  EXPECT_EQ(val.get(), 3);
}

TEST(ObjectTest, Capsule) {
  Object obj;
  int* hundred = new int(100);
  Handle capsule =
      Handle(TaggedValue::Capsule(static_cast<void*>(hundred), [](void* p) {
        delete static_cast<int*>(p);
      }));
  obj.Set(String("hundred"), capsule);
  EXPECT_EQ(*static_cast<int*>(
                obj.Get<internal::Capsule>(String("hundred"))->cast<int*>()),
            100);
}

None AppendIntegerToList(List a, Integer b) {
  a.append(b);
  return None();
}
Integer AddIntegersTyped(Integer a, Integer b) {
  return Integer(a.get() + b.get());
}
Integer ReturnFive() { return Integer(5); }

TEST(TypeUneraseCallTest, TestCallable) {
  // Add two integers.
  Callable add(TFLIB_CALLABLE_ADAPTOR(AddIntegersTyped));
  auto res = add.Call<Integer>(Integer(3), Integer(1));
  EXPECT_EQ(res->get(), 4);
}

TEST(TypeUneraseCallTest, TestAppend) {
  // Append some indices to a list.
  Callable append(TFLIB_CALLABLE_ADAPTOR(AppendIntegerToList));
  List l;
  TF_ASSERT_OK(append.Call<None>(l, Integer(3)).status());
  TF_ASSERT_OK(append.Call<None>(l, Integer(6)).status());
  EXPECT_EQ(l.size(), 2);
  EXPECT_EQ(l.Get<Integer>(0)->get(), 3);
  EXPECT_EQ(l.Get<Integer>(1)->get(), 6);
}

TEST(TypeUneraseCallTest, TestCallableWrongArgs) {
  // Try variants of wrong argument types.
  Callable append(TFLIB_CALLABLE_ADAPTOR(AddIntegersTyped));
  ASSERT_FALSE(append.Call<None>(Object(), Integer(3)).ok());
  ASSERT_FALSE(append.Call<None>(Object(), Object()).ok());
  // Try variants of wrong numbers of arguments.
  ASSERT_FALSE(append.Call().ok());
  ASSERT_FALSE(append.Call(Integer(3)).ok());
  ASSERT_FALSE(append.Call(Integer(3), Integer(4), Integer(5)).ok());
}

Handle Polymorph(Handle a) {
  auto i = Cast<Integer>(a);
  if (i.ok()) {
    return Integer(i->get() * 2);
  }
  auto f = Cast<Float>(a);
  if (f.ok()) {
    return Float(f->get() * 2.f);
  }
  return None();
}

TEST(TypeUneraseCallTest, TestCallableGeneric) {
  Callable f(TFLIB_CALLABLE_ADAPTOR(Polymorph));
  EXPECT_EQ(f.Call<Float>(Float(.2))->get(), .4f);
  EXPECT_EQ(Cast<Float>(*f.Call(Float(.2)))->get(), .4f);
  EXPECT_EQ(f.Call<Integer>(Integer(3))->get(), 6);
}

TEST(TypeUneraseCallTest, TestLambda) {
  // Test a trivial lambda that doubles an integer.
  Callable c(
      TFLIB_CALLABLE_ADAPTOR([](Integer a) { return Integer(a.get() * 2); }));
  EXPECT_EQ(c.Call<Integer>(Integer(3))->get(), 6);
  // Testa lambda that has captured state (call count).
  int call_count = 0;
  Callable f(TFLIB_CALLABLE_ADAPTOR([&call_count](Integer a, Integer b) {
    call_count++;
    return Integer(a.get() + b.get());
  }));
  EXPECT_EQ(f.Call<Integer>(Integer(3), Integer(-1))->get(), 2);
  EXPECT_EQ(f.Call<Integer>(Integer(3), Integer(-3))->get(), 0);
  EXPECT_EQ(call_count, 2);
}

}  // namespace libtf
}  // namespace tf
