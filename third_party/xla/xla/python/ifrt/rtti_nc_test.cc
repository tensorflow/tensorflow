/* Copyright 2026 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "xla/python/ifrt/rtti.h"

namespace xla {
namespace ifrt {

namespace {

class Depth0 : public RTTIExtends<Depth0, RTTIRoot> {
  static char ID;  // NOLINT
};
class Depth1 : public RTTIExtends<Depth1, Depth0> {
  static char ID;  // NOLINT
};
class Depth2 : public RTTIExtends<Depth2, Depth1> {
  static char ID;  // NOLINT
};
class Depth3 : public RTTIExtends<Depth3, Depth2> {
  static char ID;  // NOLINT
};
class Depth4 : public RTTIExtends<Depth4, Depth3> {
  static char ID;  // NOLINT
};
class Depth5 : public RTTIExtends<Depth5, Depth4> {
  static char ID;  // NOLINT
};
class Depth6 : public RTTIExtends<Depth6, Depth5> {
  static char ID;  // NOLINT
};

[[maybe_unused]] char Depth0::ID = 0;
[[maybe_unused]] char Depth1::ID = 0;
[[maybe_unused]] char Depth2::ID = 0;
[[maybe_unused]] char Depth3::ID = 0;
[[maybe_unused]] char Depth4::ID = 0;
[[maybe_unused]] char Depth5::ID = 0;
[[maybe_unused]] char Depth6::ID = 0;

// expected-error@* {{Exceeded maximum supported IFRT RTTI inheritance depth}}
class Depth7 : public RTTIExtends<Depth7, Depth6> {
  static char ID;  // NOLINT
};

}  // namespace

namespace {

// Regular classes that do not use `tsl::RCReference`.

class Base : public RTTIExtends<Base, RTTIRoot> {
 public:
  static char ID;  // NOLINT
};

class DerivedA : public RTTIExtends<DerivedA, Base> {
 public:
  static char ID;  // NOLINT
};

class DerivedB : public RTTIExtends<DerivedB, Base> {
 public:
  static char ID;  // NOLINT
};

[[maybe_unused]] char Base::ID = 0;
[[maybe_unused]] char DerivedA::ID = 0;
[[maybe_unused]] char DerivedB::ID = 0;

TEST(RttiNcTest, UnrelatedType) {
  DerivedA derived_a;
  DerivedB derived_b;

  // expected-error@* {{which are not related by inheritance}}
  isa<DerivedB>(&derived_a);
  // expected-error@* {{which are not related by inheritance}}
  isa<DerivedA>(&derived_b);

  // expected-error@* {{which are not related by inheritance}}
  cast<DerivedB>(&derived_a);
  // expected-error@* {{which are not related by inheritance}}
  cast<DerivedA>(&derived_b);

  // expected-error@* {{Casting between disjoint/unrelated types}}
  dyn_cast<DerivedB>(&derived_a);
  // expected-error@* {{Casting between disjoint/unrelated types}}
  dyn_cast<DerivedA>(&derived_b);
}

}  // namespace

}  // namespace ifrt
}  // namespace xla
