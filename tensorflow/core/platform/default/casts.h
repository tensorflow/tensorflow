/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_CASTS_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_CASTS_H_

#include <assert.h>  // for use with down_cast<>

#include <type_traits>

namespace tensorflow {

// An "upcast", i.e. a conversion from a pointer to an object to a pointer to a
// base subobject, always succeeds if the base is unambiguous and accessible,
// and so it's fine to use implicit_cast.
//
// A "downcast", i.e. a conversion from a pointer to an object to a pointer
// to a more-derived object that may contain the original object as a base
// subobject, cannot safely be done using static_cast, because you do not
// generally know whether the source object is really the base subobject of
// a containing, more-derived object of the target type. Thus, when you
// downcast in a polymorphic type hierarchy, you should use the following
// function template.
//
// In debug mode, we use dynamic_cast to double-check whether the downcast is
// legal (we die if it's not). In normal mode, we do the efficient static_cast
// instead. Thus, it's important to test in debug mode to make sure the cast is
// legal!
//
// This is the only place in the codebase we should use dynamic_cast.
// In particular, you should NOT use dynamic_cast for RTTI, e.g. for
// code like this:
//    if (auto* p = dynamic_cast<Subclass1*>(foo)) HandleASubclass1Object(p);
//    if (auto* p = dynamic_cast<Subclass2*>(foo)) HandleASubclass2Object(p);
// You should design the code some other way not to need this.

template <typename To, typename From>  // use like this: down_cast<T*>(foo);
inline To down_cast(From* f) {         // so we only accept pointers
  static_assert(
      (std::is_base_of<From, typename std::remove_pointer<To>::type>::value),
      "target type not derived from source type");

  // We skip the assert and hence the dynamic_cast if RTTI is disabled.
#if !defined(__GNUC__) || defined(__GXX_RTTI)
  // Uses RTTI in dbg and fastbuild. asserts are disabled in opt builds.
  assert(f == nullptr || dynamic_cast<To>(f) != nullptr);
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)

  return static_cast<To>(f);
}

// Overload of down_cast for references. Use like this: down_cast<T&>(foo).
// The code is slightly convoluted because we're still using the pointer
// form of dynamic cast. (The reference form throws an exception if it
// fails.)
//
// There's no need for a special const overload either for the pointer
// or the reference form. If you call down_cast with a const T&, the
// compiler will just bind From to const T.
template <typename To, typename From>
inline To down_cast(From& f) {
  static_assert(std::is_lvalue_reference<To>::value,
                "target type not a reference");
  static_assert(
      (std::is_base_of<From, typename std::remove_reference<To>::type>::value),
      "target type not derived from source type");

  // We skip the assert and hence the dynamic_cast if RTTI is disabled.
#if !defined(__GNUC__) || defined(__GXX_RTTI)
  // RTTI: debug mode only
  assert(dynamic_cast<typename std::remove_reference<To>::type*>(&f) !=
         nullptr);
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)

  return static_cast<To>(f);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_CASTS_H_
