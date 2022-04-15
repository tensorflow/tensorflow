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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_VALUE_IOSTREAM_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_VALUE_IOSTREAM_H_

#include <iostream>

#include "tensorflow/cc/experimental/libtf/value.h"

namespace tf {
namespace libtf {
namespace impl {

inline std::ostream& operator<<(std::ostream& o, const Dict& v) {
  o << "{";
  for (auto& x : v) {
    o << x.first;
    o << ": ";
    o << x.second;
    o << ", ";
  }
  o << "}";
  return o;
}
template <class IT>
inline std::ostream& OutList(std::ostream& o, IT v_start, IT const v_end,
                             char start, char end) {
  o << start;
  for (IT p = v_start; p != v_end; ++p) {
    o << *p;
    o << ", ";
  }
  o << end;
  return o;
}

class TaggedValueIOStreamVisitor {
  std::ostream& o_;

 public:
  explicit TaggedValueIOStreamVisitor(std::ostream& o) : o_(o) {}

  std::ostream& operator()(const ListPtr& x) {
    OutList(o_, x->begin(), x->end(), '[', ']');
    return o_;
  }
  std::ostream& operator()(const TuplePtr& x) {
    OutList(o_, x->begin(), x->end(), '(', ')');
    return o_;
  }
  std::ostream& operator()(const DictPtr& x) {
    o_ << *x;
    return o_;
  }
  std::ostream& operator()(const Capsule& x) {
    o_ << "Capsule(" << x.get() << ")";
    return o_;
  }
  std::ostream& operator()(const Func& x) {
    o_ << "Func";
    return o_;
  }
  std::ostream& operator()(const TaggedValueTensor& x) {
    o_ << "Tensor";
    return o_;
  }

  template <class T>
  std::ostream& operator()(const T& x) {
    o_ << x;
    return o_;
  }
};

inline std::ostream& operator<<(std::ostream& o, const TaggedValue& v) {
  return v.visit<std::ostream&>(TaggedValueIOStreamVisitor(o));
}
}  // namespace impl
}  // namespace libtf
}  // namespace tf
#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_VALUE_IOSTREAM_H_
