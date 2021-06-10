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

#include "tensorflow/core/framework/experimental/type_inference.h"

#include <iterator>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_util.h"

namespace tensorflow {

namespace full_type {

using Lattice = absl::flat_hash_map<Type, std::vector<Type>>;
Lattice MakeLattice();
Lattice MakeLatticeClosure(const Lattice& lattice);

std::string ShortName(Type type) {
  switch (type) {
    case TFT_BOOL:
      return "b";
    case TFT_UINT8:
      return "u1";
    case TFT_UINT16:
      return "u2";
    case TFT_UINT32:
      return "u4";
    case TFT_UINT64:
      return "u8";
    case TFT_INT8:
      return "i1";
    case TFT_INT16:
      return "i2";
    case TFT_INT32:
      return "i4";
    case TFT_INT64:
      return "i8";
    case TFT_FLOAT:
      return "f4";
    case TFT_HALF:
      return "f2";
    case TFT_DOUBLE:
      return "f8";
    case TFT_COMPLEX64:
      return "c4";
    case TFT_COMPLEX128:
      return "c8";
    case TFT_COMPLEX_WEAK:
      return "c*";
    case TFT_FLOAT_WEAK:
      return "f*";
    case TFT_INT_WEAK:
      return "i*";
    case TFT_BOOL_WEAK:
      return "b*";
    case TFT_BFLOAT16:
      return "bf";
  }
  return "!!";
}

Type Canonical(Type t) {
  switch (t) {
    case TFT_COMPLEX_WEAK:
      return TFT_COMPLEX64;
    case TFT_FLOAT_WEAK:
      return TFT_FLOAT;
    case TFT_INT_WEAK:
      return TFT_INT32;
    case TFT_BOOL_WEAK:
      return TFT_BOOL;
  }
  return t;
}

std::string Name(Type type) {
  switch (type) {
    case TFT_COMPLEX_WEAK:
      return "TFT_COMPLEX_WEAK";
    case TFT_FLOAT_WEAK:
      return "TFT_FLOAT_WEAK";
    case TFT_INT_WEAK:
      return "TFT_INT_WEAK";
    case TFT_BOOL_WEAK:
      return "TFT_BOOL_WEAK";
    case TFT_BFLOAT16:
      return "TFT_BFLOAT16";
  }
  auto* descriptor = FullTypeId_descriptor();
  if (const auto* value = descriptor->FindValueByNumber(type)) {
    return value->name();
  }
  return "__ERROR_UNKNOWN__";
}

Lattice MakeLattice() {
  Lattice types;
  types[TFT_BOOL_WEAK] = {TFT_BOOL};
  types[TFT_BOOL] = {TFT_INT_WEAK};
  types[TFT_INT_WEAK] = {TFT_INT8, TFT_UINT8};
  types[TFT_FLOAT_WEAK] = {TFT_HALF, TFT_BFLOAT16, TFT_COMPLEX_WEAK};
  types[TFT_BFLOAT16] = {TFT_FLOAT};
  types[TFT_HALF] = {TFT_FLOAT};
  types[TFT_COMPLEX_WEAK] = {TFT_COMPLEX64};
  types[TFT_COMPLEX64] = {TFT_COMPLEX128};
  types[TFT_FLOAT] = {TFT_DOUBLE, TFT_COMPLEX64};
  types[TFT_INT8] = {TFT_INT16};
  types[TFT_INT16] = {TFT_INT32};
  types[TFT_INT32] = {TFT_INT64};
  types[TFT_UINT8] = {TFT_INT16, TFT_UINT16};
  types[TFT_UINT16] = {TFT_INT32, TFT_UINT32};
  types[TFT_UINT32] = {TFT_INT64, TFT_UINT64};
  types[TFT_UINT64] = {TFT_FLOAT_WEAK};
  types[TFT_INT64] = {TFT_FLOAT_WEAK};
  types[TFT_DOUBLE] = {TFT_COMPLEX128};
  types[TFT_COMPLEX128] = {};
  for (auto& it : types) std::sort(it.second.begin(), it.second.end());
  return types;
}

Lattice MakeLatticeClosure(const Lattice& lattice) {
  using Set = std::set<Type>;
  Lattice result;
  for (auto& l : lattice) {
    auto type = l.first;
    Set current;
    current.insert(type);

    for (;;) {
      Set additions;
      for (const auto& i : current) {
        const auto& lat = lattice.find(i)->second;
        additions.insert(lat.begin(), lat.end());
      }
      // Check for cycles, crash since the lattice is static data.
      CHECK(additions.find(l.first) == additions.end());  // Crash OK
      // Check if we actually got any new types.
      size_t old_length = current.size();
      current.insert(additions.begin(), additions.end());
      if (old_length == current.size()) break;
    }
    result[type] = std::vector<Type>(current.begin(), current.end());
  }
  return result;
}

Lattice& LatticeSingleton() {
  static Lattice* _lattice = new Lattice(MakeLatticeClosure(MakeLattice()));
  return *_lattice;
}

Type ReturnType(Type t1, Type t2) {
  auto& closure_lattice = LatticeSingleton();
  auto it1 = closure_lattice.find(t1);
  auto it2 = closure_lattice.find(t2);
  // Check if both types are supported by promotion lattices
  if (it1 == closure_lattice.end() || it2 == closure_lattice.end()) {
    return TFT_ANY;  // TODO(aselle): mdan, do we need an error type?
  }
  std::vector<Type> t1_t2_reachable;
  std::set_intersection(it1->second.begin(), it1->second.end(),
                        it2->second.begin(), it2->second.end(),
                        std::back_inserter(t1_t2_reachable));
  constexpr Type NOT_FOUND = std::numeric_limits<Type>::max();
  Type final_type = NOT_FOUND;
  for (auto t : t1_t2_reachable) {
    // this must exist, by construction.
    auto t_reachable_it = closure_lattice.find(t);
    if (t_reachable_it->second == t1_t2_reachable) {
      if (final_type != NOT_FOUND) {
        LOG(ERROR) << "Ambiguous promotion type.";
        return TFT_ANY;
      }
      final_type = t;
    }
  }
  return Canonical(final_type);
}

FullTypeDef ReturnType(FullTypeDef t1, FullTypeDef t2) {
  auto ret = FullTypeDef();
  if (t1.type_id() != TFT_TENSOR && t2.type_id() != TFT_TENSOR) {
    ret.set_type_id(TFT_ANY);
  } else {
    auto* arg = ret.add_args();
    auto id1 = t1.args()[0].type_id(), id2 = t2.args()[0].type_id();
    ret.set_type_id(TFT_TENSOR);
    arg->set_type_id(static_cast<FullTypeId>(
        ReturnType(static_cast<Type>(id1), static_cast<Type>(id2))));
  }
  return ret;
}

}  // namespace full_type

}  // namespace tensorflow
