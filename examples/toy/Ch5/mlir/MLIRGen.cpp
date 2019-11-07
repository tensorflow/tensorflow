//===- MLIRGen.cpp - MLIR Generation from a Toy AST -----------------------===//
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
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "toy/MLIRGen.h"
#include "toy/AST.h"
#include "toy/Dialect.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace mlir::toy;
using namespace toy;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Toy AST.
///
/// This will emit operations that are specific to the Toy language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (FunctionAST &F : moduleAST) {
      auto func = mlirGen(F);
      if (!func)
        return nullptr;
      theModule.push_back(func);
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value *> symbolTable;

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(Location loc) {
    return builder.getFileLineColLoc(builder.getIdentifier(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value *value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  mlir::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> arg_types(proto.getArgs().size(),
                                               getType(VarType{}));
    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    return mlir::FuncOp::create(location, proto.getName(), func_type);
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value *> var_scope(symbolTable);

    // Create an MLIR function for the given prototype.
    mlir::FuncOp function(mlirGen(*funcAST.getProto()));
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();
    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto &name_value :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(name_value)->getName(),
                         std::get<1>(name_value))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(function.getType().getInputs(),
                                               getType(VarType{})));
    }

    return function;
  }

  /// Emit a binary operation
  mlir::Value *mlirGen(BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value *lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value *rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      return builder.create<AddOp>(location, lhs, rhs);
    case '*':
      return builder.create<MulOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value *mlirGen(VariableExprAST &expr) {
    if (auto *variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value *expr = nullptr;
    if (ret.getExpr().hasValue()) {
      if (!(expr = mlirGen(*ret.getExpr().getValue())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    builder.create<ReturnOp>(location, expr ? makeArrayRef(expr)
                                            : ArrayRef<mlir::Value *>());
    return mlir::success();
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `toy.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::Value *mlirGen(LiteralExprAST &lit) {
    auto type = getType(lit.getDims());

    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                 std::multiplies<int>()));
    collectData(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

    // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
    // method.
    return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(cast<NumberExprAST>(expr).getValue());
  }

  /// Emit a call expression. It emits specific operations for the `transpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value *mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    // Codegen the operands first.
    SmallVector<mlir::Value *, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto *arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    // Builting calls have their custom operation, meaning this is a
    // straightforward emission.
    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: toy.transpose "
                            "does not accept multiple arguments");
        return nullptr;
      }
      return builder.create<TransposeOp>(location, operands[0]);
    }

    // Otherwise this is a call to a user-defined function. Calls to ser-defined
    // functions are mapped to a custom call that takes the callee name as an
    // attribute.
    return builder.create<GenericCallOp>(location, callee, operands);
  }

  /// Emit a print expression. It emits specific operations for two builtins:
  /// transpose(x) and print(x).
  mlir::LogicalResult mlirGen(PrintExprAST &call) {
    auto *arg = mlirGen(*call.getArg());
    if (!arg)
      return mlir::failure();

    builder.create<PrintOp>(loc(call.loc()), arg);
    return mlir::success();
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value *mlirGen(NumberExprAST &num) {
    return builder.create<ConstantOp>(loc(num.loc()), num.getValue());
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value *mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case toy::ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr));
    case toy::ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case toy::ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case toy::ExprAST::Expr_Call:
      return mlirGen(cast<CallExprAST>(expr));
    case toy::ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value *mlirGen(VarDeclExprAST &vardecl) {
    auto init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value *value = mlirGen(*init);
    if (!value)
      return nullptr;

    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (!vardecl.getType().shape.empty()) {
      value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                        getType(vardecl.getType()), value);
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
    ScopedHashTableScope<StringRef, mlir::Value *> var_scope(symbolTable);
    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  /// Build an MLIR type from a Toy AST variable type (forward to the generic
  /// getType above).
  mlir::Type getType(const VarType &type) { return getType(type.shape); }
};

} // namespace

namespace toy {

// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace toy
