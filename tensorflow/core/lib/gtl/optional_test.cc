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

#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/test.h"

namespace {

using tensorflow::gtl::optional;
using tensorflow::gtl::nullopt;
using tensorflow::gtl::nullopt_t;
using tensorflow::gtl::in_place;
using tensorflow::gtl::make_optional;

template <typename T> string TypeQuals(T&) { return "&"; }
template <typename T> string TypeQuals(T&&) { return "&&"; }
template <typename T> string TypeQuals(const T&) { return "c&"; }
template <typename T> string TypeQuals(const T&&) { return "c&&"; }

struct StructorListener {
  int construct0 = 0;
  int construct1 = 0;
  int construct2 = 0;
  int listinit = 0;
  int copy = 0;
  int move = 0;
  int copy_assign = 0;
  int move_assign = 0;
  int destruct = 0;
};

struct Listenable {
  static StructorListener* listener;

  Listenable() { ++listener->construct0; }
  Listenable(int /*unused*/) { ++listener->construct1; }  // NOLINT
  Listenable(int /*unused*/, int /*unused*/) { ++listener->construct2; }
  Listenable(std::initializer_list<int> /*unused*/) { ++listener->listinit; }
  Listenable(const Listenable& /*unused*/) { ++listener->copy; }
  Listenable(Listenable&& /*unused*/) { ++listener->move; }  // NOLINT
  Listenable& operator=(const Listenable& /*unused*/) {
    ++listener->copy_assign;
    return *this;
  }
  Listenable& operator=(Listenable&& /*unused*/) {  // NOLINT
    ++listener->move_assign;
    return *this;
  }
  ~Listenable() { ++listener->destruct; }
};

StructorListener* Listenable::listener = nullptr;

TEST(optionalTest, DefaultConstructor) {
  optional<int> empty;
  EXPECT_FALSE(empty);
}

TEST(optionalTest, NullOptConstructor) {
  optional<int> empty(nullopt);
  EXPECT_FALSE(empty);
}

TEST(optionalTest, CopyConstructor) {
  optional<int> empty, opt42 = 42;
  optional<int> empty_copy(empty);
  EXPECT_FALSE(empty_copy);
  optional<int> opt42_copy(opt42);
  EXPECT_TRUE(opt42_copy);
  EXPECT_EQ(42, opt42_copy);
}

TEST(optionalTest, StructorBasic) {
  StructorListener listener;
  Listenable::listener = &listener;
  {
    optional<Listenable> empty;
    EXPECT_FALSE(empty);
    optional<Listenable> opt0(in_place);
    EXPECT_TRUE(opt0);
    optional<Listenable> opt1(in_place, 1);
    EXPECT_TRUE(opt1);
    optional<Listenable> opt2(in_place, 1, 2);
    EXPECT_TRUE(opt2);
  }
  EXPECT_EQ(1, listener.construct0);
  EXPECT_EQ(1, listener.construct1);
  EXPECT_EQ(1, listener.construct2);
  EXPECT_EQ(3, listener.destruct);
}

TEST(optionalTest, CopyMoveStructor) {
  StructorListener listener;
  Listenable::listener = &listener;
  optional<Listenable> original(in_place);
  EXPECT_EQ(1, listener.construct0);
  EXPECT_EQ(0, listener.copy);
  EXPECT_EQ(0, listener.move);
  optional<Listenable> copy(original);
  EXPECT_EQ(1, listener.construct0);
  EXPECT_EQ(1, listener.copy);
  EXPECT_EQ(0, listener.move);
  optional<Listenable> move(std::move(original));
  EXPECT_EQ(1, listener.construct0);
  EXPECT_EQ(1, listener.copy);
  EXPECT_EQ(1, listener.move);
}

TEST(optionalTest, ListInit) {
  StructorListener listener;
  Listenable::listener = &listener;
  optional<Listenable> listinit1(in_place, {1});
  optional<Listenable> listinit2(in_place, {1, 2});
  EXPECT_EQ(2, listener.listinit);
}

TEST(optionalTest, CopyAssignment) {
  const optional<int> empty, opt1 = 1, opt2 = 2;
  optional<int> empty_to_opt1, opt1_to_opt2, opt2_to_empty;

  EXPECT_FALSE(empty_to_opt1);
  empty_to_opt1 = empty;
  EXPECT_FALSE(empty_to_opt1);
  empty_to_opt1 = opt1;
  EXPECT_TRUE(empty_to_opt1);
  EXPECT_EQ(1, empty_to_opt1.value());

  EXPECT_FALSE(opt1_to_opt2);
  opt1_to_opt2 = opt1;
  EXPECT_TRUE(opt1_to_opt2);
  EXPECT_EQ(1, opt1_to_opt2.value());
  opt1_to_opt2 = opt2;
  EXPECT_TRUE(opt1_to_opt2);
  EXPECT_EQ(2, opt1_to_opt2.value());

  EXPECT_FALSE(opt2_to_empty);
  opt2_to_empty = opt2;
  EXPECT_TRUE(opt2_to_empty);
  EXPECT_EQ(2, opt2_to_empty.value());
  opt2_to_empty = empty;
  EXPECT_FALSE(opt2_to_empty);
}

TEST(optionalTest, MoveAssignment) {
  StructorListener listener;
  Listenable::listener = &listener;

  optional<Listenable> empty1, empty2, set1(in_place), set2(in_place);
  EXPECT_EQ(2, listener.construct0);
  optional<Listenable> empty_to_empty, empty_to_set, set_to_empty(in_place),
      set_to_set(in_place);
  EXPECT_EQ(4, listener.construct0);
  empty_to_empty = std::move(empty1);
  empty_to_set = std::move(set1);
  set_to_empty = std::move(empty2);
  set_to_set = std::move(set2);
  EXPECT_EQ(0, listener.copy);
  EXPECT_EQ(1, listener.move);
  EXPECT_EQ(1, listener.destruct);
  EXPECT_EQ(1, listener.move_assign);
}

TEST(optionalTest, AssignmentVarious) {
  optional<int> opt;
  EXPECT_FALSE(opt);
  opt = 42;
  EXPECT_TRUE(opt);
  EXPECT_EQ(42, opt.value());
  opt = nullopt;
  EXPECT_FALSE(opt);
  opt = 42;
  EXPECT_TRUE(opt);
  EXPECT_EQ(42, opt.value());
  opt = 43;
  EXPECT_TRUE(opt);
  EXPECT_EQ(43, opt.value());
}

TEST(optionalTest, ResetAndHasValue) {
  StructorListener listener;
  Listenable::listener = &listener;
  optional<Listenable> opt;
  EXPECT_FALSE(opt);
  EXPECT_FALSE(opt.has_value());
  opt.emplace();
  EXPECT_TRUE(opt);
  EXPECT_TRUE(opt.has_value());
  opt.reset();
  EXPECT_FALSE(opt);
  EXPECT_FALSE(opt.has_value());
  EXPECT_EQ(1, listener.destruct);
  opt.reset();
  EXPECT_FALSE(opt);
  EXPECT_FALSE(opt.has_value());
}

TEST(optionalTest, Emplace) {
  StructorListener listener;
  Listenable::listener = &listener;
  optional<Listenable> opt;
  EXPECT_FALSE(opt);
  opt.emplace(1);
  EXPECT_TRUE(opt);
  opt.emplace(1, 2);
  EXPECT_EQ(1, listener.construct1);
  EXPECT_EQ(1, listener.construct2);
  EXPECT_EQ(1, listener.destruct);
}

TEST(optionalTest, Swap) {
  optional<int> opt_empty, opt1 = 1, opt2 = 2;
  EXPECT_FALSE(opt_empty);
  EXPECT_TRUE(opt1);
  EXPECT_EQ(1, opt1.value());
  EXPECT_TRUE(opt2);
  EXPECT_EQ(2, opt2.value());
  swap(opt_empty, opt1);
  EXPECT_FALSE(opt1);
  EXPECT_TRUE(opt_empty);
  EXPECT_EQ(1, opt_empty.value());
  EXPECT_TRUE(opt2);
  EXPECT_EQ(2, opt2.value());
  swap(opt_empty, opt1);
  EXPECT_FALSE(opt_empty);
  EXPECT_TRUE(opt1);
  EXPECT_EQ(1, opt1.value());
  EXPECT_TRUE(opt2);
  EXPECT_EQ(2, opt2.value());
  swap(opt1, opt2);
  EXPECT_FALSE(opt_empty);
  EXPECT_TRUE(opt1);
  EXPECT_EQ(2, opt1.value());
  EXPECT_TRUE(opt2);
  EXPECT_EQ(1, opt2.value());
}

TEST(optionalTest, PointerStuff) {
  optional<string> opt(in_place, "foo");
  EXPECT_EQ("foo", *opt);
  const auto& opt_const = opt;
  EXPECT_EQ("foo", *opt_const);
  EXPECT_EQ(opt->size(), 3);
  EXPECT_EQ(opt_const->size(), 3);
}

// gcc has a bug pre 4.9 where it doesn't do correct overload resolution
// between rvalue reference qualified member methods. Skip that test to make
// the build green again when using the old compiler.
#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
#define SKIP_OVERLOAD_TEST_DUE_TO_GCC_BUG
#endif
#endif

TEST(optionalTest, Value) {
  using O = optional<string>;
  using CO = const optional<string>;
  O lvalue(in_place, "lvalue");
  CO clvalue(in_place, "clvalue");
  EXPECT_EQ("lvalue", lvalue.value());
  EXPECT_EQ("clvalue", clvalue.value());
  EXPECT_EQ("xvalue", O(in_place, "xvalue").value());
#ifndef SKIP_OVERLOAD_TEST_DUE_TO_GCC_BUG
  EXPECT_EQ("cxvalue", CO(in_place, "cxvalue").value());
#endif
  EXPECT_EQ("&", TypeQuals(lvalue.value()));
  EXPECT_EQ("c&", TypeQuals(clvalue.value()));
  EXPECT_EQ("&&", TypeQuals(O(in_place, "xvalue").value()));
  EXPECT_EQ("c&&", TypeQuals(CO(in_place, "cxvalue").value()));
}

TEST(optionalTest, DerefOperator) {
  using O = optional<string>;
  using CO = const optional<string>;
  O lvalue(in_place, "lvalue");
  CO clvalue(in_place, "clvalue");
  EXPECT_EQ("lvalue", *lvalue);
  EXPECT_EQ("clvalue", *clvalue);
  EXPECT_EQ("xvalue", *O(in_place, "xvalue"));
#ifndef SKIP_OVERLOAD_TEST_DUE_TO_GCC_BUG
  EXPECT_EQ("cxvalue", *CO(in_place, "cxvalue"));
#endif
  EXPECT_EQ("&", TypeQuals(*lvalue));
  EXPECT_EQ("c&", TypeQuals(*clvalue));
  EXPECT_EQ("&&", TypeQuals(*O(in_place, "xvalue")));
  EXPECT_EQ("c&&", TypeQuals(*CO(in_place, "cxvalue")));
}

TEST(optionalTest, ValueOr) {
  optional<double> opt_empty, opt_set = 1.2;
  EXPECT_EQ(42.0, opt_empty.value_or(42));
  EXPECT_EQ(1.2, opt_set.value_or(42));
  EXPECT_EQ(42.0, optional<double>().value_or(42));
  EXPECT_EQ(1.2, optional<double>(1.2).value_or(42));
}

TEST(optionalTest, make_optional) { EXPECT_EQ(42, make_optional(42).value()); }

TEST(optionalTest, Comparisons) {
  optional<int> ae, be, a2 = 2, b2 = 2, a4 = 4, b4 = 4;

#define optionalTest_Comparisons_EXPECT_LESS(x, y) \
  EXPECT_FALSE((x) == (y));                        \
  EXPECT_TRUE((x) != (y));                         \
  EXPECT_TRUE((x) < (y));                          \
  EXPECT_FALSE((x) > (y));                         \
  EXPECT_TRUE((x) <= (y));                         \
  EXPECT_FALSE((x) >= (y));

#define optionalTest_Comparisons_EXPECT_SAME(x, y) \
  EXPECT_TRUE((x) == (y));                         \
  EXPECT_FALSE((x) != (y));                        \
  EXPECT_FALSE((x) < (y));                         \
  EXPECT_FALSE((x) > (y));                         \
  EXPECT_TRUE((x) <= (y));                         \
  EXPECT_TRUE((x) >= (y));

#define optionalTest_Comparisons_EXPECT_GREATER(x, y) \
  EXPECT_FALSE((x) == (y));                           \
  EXPECT_TRUE((x) != (y));                            \
  EXPECT_FALSE((x) < (y));                            \
  EXPECT_TRUE((x) > (y));                             \
  EXPECT_FALSE((x) <= (y));                           \
  EXPECT_TRUE((x) >= (y));

  // LHS: nullopt, ae, a2, 3, a4
  // RHS: nullopt, be, b2, 3, b4

  // optionalTest_Comparisons_EXPECT_NOT_TO_WORK(nullopt,nullopt);
  optionalTest_Comparisons_EXPECT_SAME(nullopt, be);
  optionalTest_Comparisons_EXPECT_LESS(nullopt, b2);
  // optionalTest_Comparisons_EXPECT_NOT_TO_WORK(nullopt,3);
  optionalTest_Comparisons_EXPECT_LESS(nullopt, b4);

  optionalTest_Comparisons_EXPECT_SAME(ae, nullopt);
  optionalTest_Comparisons_EXPECT_SAME(ae, be);
  optionalTest_Comparisons_EXPECT_LESS(ae, b2);
  optionalTest_Comparisons_EXPECT_LESS(ae, 3);
  optionalTest_Comparisons_EXPECT_LESS(ae, b4);

  optionalTest_Comparisons_EXPECT_GREATER(a2, nullopt);
  optionalTest_Comparisons_EXPECT_GREATER(a2, be);
  optionalTest_Comparisons_EXPECT_SAME(a2, b2);
  optionalTest_Comparisons_EXPECT_LESS(a2, 3);
  optionalTest_Comparisons_EXPECT_LESS(a2, b4);

  // optionalTest_Comparisons_EXPECT_NOT_TO_WORK(3,nullopt);
  optionalTest_Comparisons_EXPECT_GREATER(3, be);
  optionalTest_Comparisons_EXPECT_GREATER(3, b2);
  optionalTest_Comparisons_EXPECT_SAME(3, 3);
  optionalTest_Comparisons_EXPECT_LESS(3, b4);

  optionalTest_Comparisons_EXPECT_GREATER(a4, nullopt);
  optionalTest_Comparisons_EXPECT_GREATER(a4, be);
  optionalTest_Comparisons_EXPECT_GREATER(a4, b2);
  optionalTest_Comparisons_EXPECT_GREATER(a4, 3);
  optionalTest_Comparisons_EXPECT_SAME(a4, b4);
}

TEST(optionalTest, SwapRegression) {
  StructorListener listener;
  Listenable::listener = &listener;

  {
    optional<Listenable> a;
    optional<Listenable> b(in_place);
    a.swap(b);
  }

  EXPECT_EQ(1, listener.construct0);
  EXPECT_EQ(1, listener.move);
  EXPECT_EQ(2, listener.destruct);

  {
    optional<Listenable> a(in_place);
    optional<Listenable> b;
    a.swap(b);
  }

  EXPECT_EQ(2, listener.construct0);
  EXPECT_EQ(2, listener.move);
  EXPECT_EQ(4, listener.destruct);
}

TEST(optionalTest, BigStringLeakCheck) {
  constexpr size_t n = 1 << 16;

  using OS = optional<string>;

  OS a;
  OS b = nullopt;
  OS c = string(n, 'c');
  string sd(n, 'd');
  OS d = sd;
  OS e(in_place, n, 'e');
  OS f;
  f.emplace(n, 'f');

  OS ca(a);
  OS cb(b);
  OS cc(c);
  OS cd(d);
  OS ce(e);

  OS oa;
  OS ob = nullopt;
  OS oc = string(n, 'c');
  string sod(n, 'd');
  OS od = sod;
  OS oe(in_place, n, 'e');
  OS of;
  of.emplace(n, 'f');

  OS ma(std::move(oa));
  OS mb(std::move(ob));
  OS mc(std::move(oc));
  OS md(std::move(od));
  OS me(std::move(oe));
  OS mf(std::move(of));

  OS aa1;
  OS ab1 = nullopt;
  OS ac1 = string(n, 'c');
  string sad1(n, 'd');
  OS ad1 = sad1;
  OS ae1(in_place, n, 'e');
  OS af1;
  af1.emplace(n, 'f');

  OS aa2;
  OS ab2 = nullopt;
  OS ac2 = string(n, 'c');
  string sad2(n, 'd');
  OS ad2 = sad2;
  OS ae2(in_place, n, 'e');
  OS af2;
  af2.emplace(n, 'f');

  aa1 = af2;
  ab1 = ae2;
  ac1 = ad2;
  ad1 = ac2;
  ae1 = ab2;
  af1 = aa2;

  OS aa3;
  OS ab3 = nullopt;
  OS ac3 = string(n, 'c');
  string sad3(n, 'd');
  OS ad3 = sad3;
  OS ae3(in_place, n, 'e');
  OS af3;
  af3.emplace(n, 'f');

  aa3 = nullopt;
  ab3 = nullopt;
  ac3 = nullopt;
  ad3 = nullopt;
  ae3 = nullopt;
  af3 = nullopt;

  OS aa4;
  OS ab4 = nullopt;
  OS ac4 = string(n, 'c');
  string sad4(n, 'd');
  OS ad4 = sad4;
  OS ae4(in_place, n, 'e');
  OS af4;
  af4.emplace(n, 'f');

  aa4 = OS(in_place, n, 'a');
  ab4 = OS(in_place, n, 'b');
  ac4 = OS(in_place, n, 'c');
  ad4 = OS(in_place, n, 'd');
  ae4 = OS(in_place, n, 'e');
  af4 = OS(in_place, n, 'f');

  OS aa5;
  OS ab5 = nullopt;
  OS ac5 = string(n, 'c');
  string sad5(n, 'd');
  OS ad5 = sad5;
  OS ae5(in_place, n, 'e');
  OS af5;
  af5.emplace(n, 'f');

  string saa5(n, 'a');
  string sab5(n, 'a');
  string sac5(n, 'a');
  string sad52(n, 'a');
  string sae5(n, 'a');
  string saf5(n, 'a');

  aa5 = saa5;
  ab5 = sab5;
  ac5 = sac5;
  ad5 = sad52;
  ae5 = sae5;
  af5 = saf5;

  OS aa6;
  OS ab6 = nullopt;
  OS ac6 = string(n, 'c');
  string sad6(n, 'd');
  OS ad6 = sad6;
  OS ae6(in_place, n, 'e');
  OS af6;
  af6.emplace(n, 'f');

  aa6 = string(n, 'a');
  ab6 = string(n, 'b');
  ac6 = string(n, 'c');
  ad6 = string(n, 'd');
  ae6 = string(n, 'e');
  af6 = string(n, 'f');

  OS aa7;
  OS ab7 = nullopt;
  OS ac7 = string(n, 'c');
  string sad7(n, 'd');
  OS ad7 = sad7;
  OS ae7(in_place, n, 'e');
  OS af7;
  af7.emplace(n, 'f');

  aa7.emplace(n, 'A');
  ab7.emplace(n, 'B');
  ac7.emplace(n, 'C');
  ad7.emplace(n, 'D');
  ae7.emplace(n, 'E');
  af7.emplace(n, 'F');
}

TEST(optionalTest, MoveAssignRegression) {
  StructorListener listener;
  Listenable::listener = &listener;

  {
    optional<Listenable> a;
    Listenable b;
    a = std::move(b);
  }

  EXPECT_EQ(1, listener.construct0);
  EXPECT_EQ(1, listener.move);
  EXPECT_EQ(2, listener.destruct);
}

TEST(optionalTest, ValueType) {
  EXPECT_TRUE((std::is_same<optional<int>::value_type, int>::value));
  EXPECT_TRUE((std::is_same<optional<string>::value_type, string>::value));
  EXPECT_FALSE((std::is_same<optional<int>::value_type, nullopt_t>::value));
}

TEST(optionalTest, Hash) {
  std::hash<optional<int>> hash;
  std::set<size_t> hashcodes;
  hashcodes.insert(hash(nullopt));
  for (int i = 0; i < 100; ++i) {
    hashcodes.insert(hash(i));
  }
  EXPECT_GT(hashcodes.size(), 90);
}

struct MoveMeNoThrow {
  MoveMeNoThrow() : x(0) {}
  MoveMeNoThrow(const MoveMeNoThrow& other) : x(other.x) {
    LOG(FATAL) << "Should not be called.";
  }
  MoveMeNoThrow(MoveMeNoThrow&& other) noexcept : x(other.x) {}
  int x;
};

struct MoveMeThrow {
  MoveMeThrow() : x(0) {}
  MoveMeThrow(const MoveMeThrow& other) : x(other.x) {}
  MoveMeThrow(MoveMeThrow&& other) : x(other.x) {}
  int x;
};

TEST(optionalTest, NoExcept) {
  static_assert(
      std::is_nothrow_move_constructible<optional<MoveMeNoThrow>>::value, "");
  static_assert(
      !std::is_nothrow_move_constructible<optional<MoveMeThrow>>::value, "");
  std::vector<optional<MoveMeNoThrow>> v;
  for (int i = 0; i < 10; ++i) v.emplace_back();
}

}  // namespace
