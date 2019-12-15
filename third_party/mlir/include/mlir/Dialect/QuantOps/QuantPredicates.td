//===- QuantPredicates.td - Predicates for dialect types ---*- tablegen -*-===//
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
// Predicates for types in the Quantization dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QUANTOPS_QUANT_PREDICATES_
#define DIALECT_QUANTOPS_QUANT_PREDICATES_

//===----------------------------------------------------------------------===//
// Quantization type definitions
//===----------------------------------------------------------------------===//

class quant_TypedPrimitiveOrContainer<Type etype> :
    Type<Or<[etype.predicate,
                TensorOf<[etype]>.predicate,
                VectorOf<[etype]>.predicate]>,
         "primitive/tensor/vector of " # etype.description>;

// An implementation of QuantizedType.
def quant_QuantizedType :
    Type<CPred<"$_self.isa<mlir::quant::QuantizedType>()">, "QuantizedType">;

// A primitive type that can represent a real value. This is either a
// floating point value or a quantized type.
def quant_RealPrimitiveType :
    Type<Or<[AnyFloat.predicate, quant_QuantizedType.predicate]>,
    "real valued primitive (float or quantized type)">;

// A primitive type that can represent a storage value. This is either an
// integer or quantized type.
def quant_StoragePrimitiveType :
    Type<Or<[AnyInteger.predicate, quant_QuantizedType.predicate]>,
    "quantized storage primitive (integer or quantized type)">;

// A primitive or container of RealPrimitiveType.
def quant_RealValueType :
    quant_TypedPrimitiveOrContainer<quant_RealPrimitiveType>;

// A primitive or container of StoragePrimitiveType.
def quant_StorageValueType :
    quant_TypedPrimitiveOrContainer<quant_StoragePrimitiveType>;

// Either a real valued or storage primitive or container type.
def quant_RealOrStorageValueType :
    Type<Or<[quant_RealValueType.predicate,
                quant_StorageValueType.predicate]>>;

// An implementation of UniformQuantizedType.
def quant_UniformQuantizedType :
    Type<CPred<"$_self.isa<UniformQuantizedType>()">, "UniformQuantizedType">;

// Predicate for detecting a container or primitive of UniformQuantizedType.
def quant_UniformQuantizedValueType :
    quant_TypedPrimitiveOrContainer<quant_UniformQuantizedType>;

#endif // DIALECT_QUANTOPS_QUANT_PREDICATES_
