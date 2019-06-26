//===- ToyDialect.cpp - Toy IR Dialect registration in MLIR ---------------===//
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
// This file implements the dialect for the Toy IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace toy {
namespace detail {

/// This class holds the implementation of the ToyArrayType.
/// It is intended to be uniqued based on its content and owned by the context.
struct ToyArrayTypeStorage : public mlir::TypeStorage {
  /// This defines how we unique this type in the context: our key contains
  /// only the shape, a more complex type would have multiple entries in the
  /// tuple here.
  /// The element of the tuples usually matches 1-1 the arguments from the
  /// public `get()` method arguments from the facade.
  using KeyTy = std::tuple<ArrayRef<int64_t>>;
  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key));
  }
  /// When the key hash hits an existing type, we compare the shape themselves
  /// to confirm we have the right type.
  bool operator==(const KeyTy &key) const { return key == KeyTy(getShape()); }

  /// This is a factory method to create our type storage. It is only
  /// invoked after looking up the type in the context using the key and not
  /// finding it.
  static ToyArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    // Copy the shape array into the bumpptr allocator owned by the context.
    ArrayRef<int64_t> shape = allocator.copyInto(std::get<0>(key));

    // Allocate the instance for the ToyArrayTypeStorage itself
    auto *storage = allocator.allocate<ToyArrayTypeStorage>();
    // Initialize the instance using placement new.
    return new (storage) ToyArrayTypeStorage(shape);
  }

  ArrayRef<int64_t> getShape() const { return shape; }

private:
  ArrayRef<int64_t> shape;

  /// Constructor is only invoked from the `construct()` method above.
  ToyArrayTypeStorage(ArrayRef<int64_t> shape) : shape(shape) {}
};

} // namespace detail

mlir::Type ToyArrayType::getElementType() {
  return mlir::FloatType::getF64(getContext());
}

ToyArrayType ToyArrayType::get(mlir::MLIRContext *context,
                               ArrayRef<int64_t> shape) {
  return Base::get(context, ToyTypeKind::TOY_ARRAY, shape);
}

ArrayRef<int64_t> ToyArrayType::getShape() { return getImpl()->getShape(); }

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
ToyDialect::ToyDialect(mlir::MLIRContext *ctx) : mlir::Dialect("toy", ctx) {
  addOperations<ConstantOp, GenericCallOp, PrintOp, TransposeOp, ReshapeOp,
                MulOp, AddOp, ReturnOp>();
  addTypes<ToyArrayType>();
}

/// Parse a type registered to this dialect, we expect only Toy arrays.
mlir::Type ToyDialect::parseType(StringRef tyData, mlir::Location loc) const {
  // Sanity check: we only support array or array<...>
  if (!tyData.startswith("array")) {
    emitError(loc, "Invalid Toy type '" + tyData + "', array expected");
    return nullptr;
  }
  // Drop the "array" prefix from the type name, we expect either an empty
  // string or just the shape.
  tyData = tyData.drop_front(StringRef("array").size());
  // This is the generic array case without shape, early return it.
  if (tyData.empty())
    return ToyArrayType::get(getContext());

  // Use a regex to parse the shape (for efficient we should store this regex in
  // the dialect itself).
  SmallVector<StringRef, 4> matches;
  auto shapeRegex = llvm::Regex("^<([0-9]+)(, ([0-9]+))*>$");
  if (!shapeRegex.match(tyData, &matches)) {
    emitError(loc, "Invalid toy array shape '" + tyData + "'");
    return nullptr;
  }
  SmallVector<int64_t, 4> shape;
  // Iterate through the captures, skip the first one which is the full string.
  for (auto dimStr :
       llvm::make_range(std::next(matches.begin()), matches.end())) {
    if (dimStr.startswith(","))
      continue; // POSIX misses non-capturing groups.
    if (dimStr.empty())
      continue; // '*' makes it an optional group capture
    // Convert the capture to an integer
    unsigned long long dim;
    if (getAsUnsignedInteger(dimStr, /* Radix = */ 10, dim)) {
      emitError(loc, "Couldn't parse dimension as integer, matched: " + dimStr);
      return mlir::Type();
    }
    shape.push_back(dim);
  }
  // Finally we collected all the dimensions in the shape,
  // create the array type.
  return ToyArrayType::get(getContext(), shape);
}

/// Print a Toy array type, for example `array<2, 3, 4>`
void ToyDialect::printType(mlir::Type type, raw_ostream &os) const {
  auto arrayTy = type.dyn_cast<ToyArrayType>();
  if (!arrayTy) {
    os << "unknown toy type";
    return;
  }
  os << "array";
  if (!arrayTy.getShape().empty()) {
    os << "<";
    mlir::interleaveComma(arrayTy.getShape(), os);
    os << ">";
  }
}

////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Helper to verify that the result of an operation is a Toy array type.
template <typename T> static mlir::LogicalResult verifyToyReturnArray(T *op) {
  if (!op->getResult()->getType().template isa<ToyArrayType>()) {
    std::string msg;
    raw_string_ostream os(msg);
    os << "expects a Toy Array for its argument, got "
       << op->getResult()->getType();
    return op->emitOpError(os.str());
  }
  return mlir::success();
}

/// Helper to verify that the two operands of a binary operation are Toy
/// arrays..
template <typename T> static mlir::LogicalResult verifyToyBinOperands(T *op) {
  if (!op->getOperand(0)->getType().template isa<ToyArrayType>()) {
    std::string msg;
    raw_string_ostream os(msg);
    os << "expects a Toy Array for its LHS, got "
       << op->getOperand(0)->getType();
    return op->emitOpError(os.str());
  }
  if (!op->getOperand(1)->getType().template isa<ToyArrayType>()) {
    std::string msg;
    raw_string_ostream os(msg);
    os << "expects a Toy Array for its LHS, got "
       << op->getOperand(0)->getType();
    return op->emitOpError(os.str());
  }
  return mlir::success();
}

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::Builder *builder, mlir::OperationState *state,
                       ArrayRef<int64_t> shape, mlir::DenseElementsAttr value) {
  state->types.push_back(ToyArrayType::get(builder->getContext(), shape));
  auto dataAttribute = builder->getNamedAttr("value", value);
  state->attributes.push_back(dataAttribute);
}

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::Builder *builder, mlir::OperationState *state,
                       mlir::FloatAttr value) {
  // Broadcast and forward to the other build factory
  mlir::Type elementType = mlir::FloatType::getF64(builder->getContext());
  auto dataType = builder->getTensorType({1}, elementType);
  auto dataAttribute = builder->getDenseElementsAttr(dataType, {value})
                           .cast<mlir::DenseElementsAttr>();

  ConstantOp::build(builder, state, {1}, dataAttribute);
}

/// Verifier for constant operation.
mlir::LogicalResult ConstantOp::verify() {
  // Ensure that the return type is a Toy array
  if (failed(verifyToyReturnArray(this)))
    return mlir::failure();

  // We expect the constant itself to be stored as an attribute.
  auto dataAttr = getAttr("value").dyn_cast<mlir::DenseElementsAttr>();
  if (!dataAttr) {
    return emitOpError(
        "missing valid `value` DenseElementsAttribute on toy.constant()");
  }
  auto attrType = dataAttr.getType().dyn_cast<mlir::TensorType>();
  if (!attrType) {
    return emitOpError(
        "missing valid `value` DenseElementsAttribute on toy.constant()");
  }

  // If the return type of the constant is not a generic array, the shape must
  // match the shape of the attribute holding the data.
  auto resultType = getResult()->getType().cast<ToyArrayType>();
  if (!resultType.isGeneric()) {
    if (attrType.getRank() != resultType.getRank()) {
      return emitOpError("The rank of the toy.constant return type must match "
                         "the one of the attached value attribute: " +
                         Twine(attrType.getRank()) +
                         " != " + Twine(resultType.getRank()));
    }
    for (int dim = 0; dim < attrType.getRank(); ++dim) {
      if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
        std::string msg;
        raw_string_ostream os(msg);
        return emitOpError(
            "Shape mismatch between toy.constant return type and its "
            "attribute at dimension " +
            Twine(dim) + ": " + Twine(attrType.getShape()[dim]) +
            " != " + Twine(resultType.getShape()[dim]));
      }
    }
  }
  return mlir::success();
}

void GenericCallOp::build(mlir::Builder *builder, mlir::OperationState *state,
                          StringRef callee, ArrayRef<mlir::Value *> arguments) {
  // Generic call always returns a generic ToyArray initially
  state->types.push_back(ToyArrayType::get(builder->getContext()));
  state->operands.assign(arguments.begin(), arguments.end());
  auto calleeAttr = builder->getStringAttr(callee);
  state->attributes.push_back(builder->getNamedAttr("callee", calleeAttr));
}

mlir::LogicalResult GenericCallOp::verify() {
  // Verify that every operand is a Toy Array
  for (int opId = 0, num = getNumOperands(); opId < num; ++opId) {
    if (!getOperand(opId)->getType().template isa<ToyArrayType>()) {
      std::string msg;
      raw_string_ostream os(msg);
      os << "expects a Toy Array for its " << opId << " operand, got "
         << getOperand(opId)->getType();
      return emitOpError(os.str());
    }
  }
  return mlir::success();
}

/// Return the name of the callee.
StringRef GenericCallOp::getCalleeName() {
  return getAttr("callee").cast<mlir::StringAttr>().getValue();
}

template <typename T> static mlir::LogicalResult verifyToySingleOperand(T *op) {
  if (!op->getOperand()->getType().template isa<ToyArrayType>()) {
    std::string msg;
    raw_string_ostream os(msg);
    os << "expects a Toy Array for its argument, got "
       << op->getOperand()->getType();
    return op->emitOpError(os.str());
  }
  return mlir::success();
}

void ReturnOp::build(mlir::Builder *builder, mlir::OperationState *state,
                     mlir::Value *value) {
  // Return does not return any value and has an optional single argument
  if (value)
    state->operands.push_back(value);
}

mlir::LogicalResult ReturnOp::verify() {
  if (getNumOperands() > 1) {
    std::string msg;
    raw_string_ostream os(msg);
    os << "expects zero or one operand, got " << getNumOperands();
    return emitOpError(os.str());
  }
  if (hasOperand() && failed(verifyToySingleOperand(this)))
    return mlir::failure();
  return mlir::success();
}

void PrintOp::build(mlir::Builder *builder, mlir::OperationState *state,
                    mlir::Value *value) {
  // Print does not return any value and has a single argument
  state->operands.push_back(value);
}

mlir::LogicalResult PrintOp::verify() {
  if (failed(verifyToySingleOperand(this)))
    return mlir::failure();
  return mlir::success();
}

void TransposeOp::build(mlir::Builder *builder, mlir::OperationState *state,
                        mlir::Value *value) {
  state->types.push_back(ToyArrayType::get(builder->getContext()));
  state->operands.push_back(value);
}

mlir::LogicalResult TransposeOp::verify() {
  if (failed(verifyToySingleOperand(this)))
    return mlir::failure();
  return mlir::success();
}

void ReshapeOp::build(mlir::Builder *builder, mlir::OperationState *state,
                      mlir::Value *value, ToyArrayType reshapedType) {
  state->types.push_back(reshapedType);
  state->operands.push_back(value);
}

mlir::LogicalResult ReshapeOp::verify() {
  if (failed(verifyToySingleOperand(this)))
    return mlir::failure();
  auto retTy = getResult()->getType().dyn_cast<ToyArrayType>();
  if (!retTy)
    return emitOpError("toy.reshape is expected to produce a Toy array");
  if (retTy.isGeneric())
    return emitOpError("toy.reshape is expected to produce a shaped Toy array, "
                       "got a generic one.");
  return mlir::success();
}

void AddOp::build(mlir::Builder *builder, mlir::OperationState *state,
                  mlir::Value *lhs, mlir::Value *rhs) {
  state->types.push_back(ToyArrayType::get(builder->getContext()));
  state->operands.push_back(lhs);
  state->operands.push_back(rhs);
}

mlir::LogicalResult AddOp::verify() {
  if (failed(verifyToyBinOperands(this)))
    return mlir::failure();
  return mlir::success();
}

void MulOp::build(mlir::Builder *builder, mlir::OperationState *state,
                  mlir::Value *lhs, mlir::Value *rhs) {
  state->types.push_back(ToyArrayType::get(builder->getContext()));
  state->operands.push_back(lhs);
  state->operands.push_back(rhs);
}

mlir::LogicalResult MulOp::verify() {
  if (failed(verifyToyBinOperands(this)))
    return mlir::failure();
  return mlir::success();
}

} // namespace toy
