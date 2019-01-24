//===- BlockAndValueMapping.h -----------------------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines a utility class for maintaining a mapping for multiple
// value types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BLOCKANDVALUEMAPPING_H
#define MLIR_IR_BLOCKANDVALUEMAPPING_H

#include "mlir/IR/Block.h"

namespace mlir {
// This is a utility class for mapping one set of values to another. New
// mappings can be inserted via 'map'. Existing mappings can be
// found via the 'lookup*' functions. There are two variants that differ only in
// return value when an existing is not found for the provided key.
// 'lookupOrNull' returns nullptr where as 'lookupOrDefault' will return the
// lookup key.
class BlockAndValueMapping {
public:
  /// Inserts a new mapping for 'from' to 'to'. If there is an existing mapping,
  /// it is overwritten.
  void map(const Block *from, Block *to) { valueMap[from] = to; }
  void map(const Value *from, Value *to) { valueMap[from] = to; }

  /// Erases a mapping for 'from'.
  void erase(const IRObjectWithUseList *from) { valueMap.erase(from); }

  /// Checks to see if a mapping for 'from' exists.
  bool contains(const IRObjectWithUseList *from) const {
    return valueMap.count(from);
  }

  /// Lookup a mapped value within the map. If a mapping for the provided value
  /// does not exist then return nullptr.
  Block *lookupOrNull(const Block *from) const {
    return lookupOrValue(from, (Block *)nullptr);
  }
  Value *lookupOrNull(const Value *from) const {
    return lookupOrValue(from, (Value *)nullptr);
  }

  /// Lookup a mapped value within the map. If a mapping for the provided value
  /// does not exist then return the provided value.
  Block *lookupOrDefault(Block *from) const {
    return lookupOrValue(from, from);
  }
  Value *lookupOrDefault(Value *from) const {
    return lookupOrValue(from, from);
  }

  /// Clears all mappings held by the mapper.
  void clear() { valueMap.clear(); }

private:
  /// Utility lookupOrValue that looks up an existing key or returns the
  /// provided value. This function assumes that if a mapping does exist, then
  /// it is of 'T' type.
  template <typename T> T *lookupOrValue(const T *from, T *value) const {
    auto it = valueMap.find(from);
    return it != valueMap.end() ? static_cast<T *>(it->second) : value;
  }

  llvm::DenseMap<const IRObjectWithUseList *, IRObjectWithUseList *> valueMap;
};

} // end namespace mlir

#endif // MLIR_IR_BLOCKANDVALUEMAPPING_H
