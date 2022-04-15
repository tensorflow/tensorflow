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
#include <cstdint>

#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/cc/experimental/libtf/value_iostream.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tf {
namespace libtf {

namespace {

// AddTagged using tagged values
TaggedValue AddTagged(TaggedValue args, TaggedValue kwargs) {
  return TaggedValue(args.tuple()[0].i64() + args.tuple()[1].i64());
}

int64_t AddRaw(int64_t a, int64_t b) { return a + b; }

}  // namespace

// Add numbers in a loop by calling a callable.
void CallFunctions(::testing::benchmark::State& state) {
  Integer sum(0);
  Callable callable((impl::TaggedValue(impl::Func(AddTagged))));
  *callable.Call<Integer>(sum, Integer(30));
  size_t i = 0;
  for (auto dummy : state) {
    sum = *callable.Call<Integer>(sum, Integer(i));
    i++;
  }
}

// Add numbers in a loop by calling a callable, looking up method every
// time by tokenized string.
void CallFunctionsIndirect(::testing::benchmark::State& state) {
  Integer sum(0);
  Callable callable((impl::TaggedValue(impl::Func(AddTagged))));
  Object o;
  String name("f");
  o.Set(name, callable);
  size_t i = 0;
  for (auto dummy : state) {
    sum = *(o.Get<Callable>(name))->Call<Integer>(sum, Integer(i));
    i++;
  }
}

// Add numbers in a loop by calling a callable, looking up method every
// time by non-tokenized string.
void CallFunctionsIndirectNaive(::testing::benchmark::State& state) {
  Integer sum(0);
  Callable callable((impl::TaggedValue(impl::Func(AddTagged))));
  Object o;
  o.Set(String("f"), callable);
  size_t i = 0;
  for (auto dummy : state) {
    sum = *(o.Get<Callable>(String("f")))->Call<Integer>(sum, Integer(i));
    i++;
  }
}

// Add numbers in a loop by calling a raw C++ function with a function
// pointer.
void CallFunctionsBase(::testing::benchmark::State& state) {
  int64_t sum = 0;
  typedef int64_t (*Func)(int64_t a, int64_t b);
  volatile Func f_raw = AddRaw;
  Func f = f_raw;
  size_t i = 0;
  for (auto dummy : state) {
    sum = f(sum, i);
    i++;
  }
  // volatile int64_t result = sum;
}

BENCHMARK(CallFunctions)->Arg(1 << 10);
BENCHMARK(CallFunctionsIndirect)->Arg(1 << 10);
BENCHMARK(CallFunctionsIndirectNaive)->Arg(1 << 10);
BENCHMARK(CallFunctionsBase)->Arg(1 << 10);

}  // namespace libtf
}  // namespace tf
