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
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/StandardOps/Ops.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace toy;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::make_unique;
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
///
/// At this point we take advantage of the "raw" MLIR APIs to create operations
/// that haven't been registered in any way with MLIR. These operations are
/// unknown to MLIR, custom passes could operate by string-matching the name of
/// these operations, but no other type checking or semantic is associated with
/// them natively by MLIR.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : context(context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module.
  mlir::OwningModuleRef mlirGen(ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

    for (FunctionAST &F : moduleAST) {
      auto func = mlirGen(F);
      if (!func)
        return nullptr;
      theModule->push_back(func);
    }

    // FIXME: (in the next chapter...) without registering a dialect in MLIR,
    // this won't do much, but it should at least check some structural
    // properties.
    if (failed(mlir::verify(*theModule))) {
      emitError(mlir::UnknownLoc::get(&context), "Module verification error");
      return nullptr;
    }

    return std::move(theModule);
  }

private:
  /// In MLIR (like in LLVM) a "context" object holds the memory allocation and
  /// the ownership of many internal structure of the IR and provide a level
  /// of "uniquing" across multiple modules (types for instance).
  mlir::MLIRContext &context;

  /// A "module" matches a source file: it contains a list of functions.
  mlir::OwningModuleRef theModule;

  /// The builder is a helper class to create IR inside a function. It is
  /// re-initialized every time we enter a function and kept around as a
  /// convenience for emitting individual operations.
  /// The builder is stateful, in particular it keeeps an "insertion point":
  /// this is where the next operations will be introduced.
  std::unique_ptr<mlir::OpBuilder> builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value *> symbolTable;

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(Location loc) {
    return mlir::FileLineColLoc::get(mlir::Identifier::get(*loc.file, &context),
                                     loc.line, loc.col, &context);
  }

  /// Declare a variable in the current scope, return true if the variable
  /// wasn't declared yet.
  bool declare(llvm::StringRef var, mlir::Value *value) {
    if (symbolTable.count(var)) {
      return false;
    }
    symbolTable.insert(var, value);
    return true;
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  mlir::FuncOp mlirGen(PrototypeAST &proto) {
    // This is a generic function, the return type will be inferred later.
    llvm::SmallVector<mlir::Type, 4> ret_types;
    // Arguments type is uniformly a generic array.
    llvm::SmallVector<mlir::Type, 4> arg_types(proto.getArgs().size(),
                                               getType(VarType{}));
    auto func_type = mlir::FunctionType::get(arg_types, ret_types, &context);
    auto function = mlir::FuncOp::create(loc(proto.loc()), proto.getName(),
                                         func_type, /* attrs = */ {});

    // Mark the function as generic: it'll require type specialization for every
    // call site.
    if (function.getNumArguments())
      function.setAttr("toy.generic", mlir::BoolAttr::get(true, &context));

    return function;
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
    function.addEntryBlock();

    auto &entryBlock = function.front();
    auto &protoArgs = funcAST.getProto()->getArgs();
    // Declare all the function arguments in the symbol table.
    for (const auto &name_value :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      declare(std::get<0>(name_value)->getName(), std::get<1>(name_value));
    }

    // Create a builder for the function, it will be used throughout the codegen
    // to create operations in this function.
    builder = llvm::make_unique<mlir::OpBuilder>(function.getBody());

    // Emit the body of the function.
    if (!mlirGen(*funcAST.getBody())) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    if (function.getBlocks().back().back().getName().getStringRef() !=
        "toy.return") {
      ReturnExprAST fakeRet(funcAST.getProto()->loc(), llvm::None);
      mlirGen(fakeRet);
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
    mlir::Value *L = mlirGen(*binop.getLHS());
    if (!L)
      return nullptr;
    mlir::Value *R = mlirGen(*binop.getRHS());
    if (!R)
      return nullptr;
    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      return builder->create<AddOp>(location, L, R).getResult();
      break;
    case '*':
      return builder->create<MulOp>(location, L, R).getResult();
    default:
      emitError(loc(binop.loc()), "Error: invalid binary operator '")
          << binop.getOp() << "'";
      return nullptr;
    }
  }

  // This is a reference to a variable in an expression. The variable is
  // expected to have been declared and so should have a value in the symbol
  // table, otherwise emit an error and return nullptr.
  mlir::Value *mlirGen(VariableExprAST &expr) {
    if (symbolTable.count(expr.getName()))
      return symbolTable.lookup(expr.getName());
    emitError(loc(expr.loc()), "Error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  // Emit a return operation, return true on success.
  bool mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());
    // `return` takes an optional expression, we need to account for it here.
    if (!ret.getExpr().hasValue()) {
      builder->create<ReturnOp>(location);
      return true;
    }
    auto *expr = mlirGen(*ret.getExpr().getValue());
    if (!expr)
      return false;
    builder->create<ReturnOp>(location, expr);
    return true;
  }

  // Emit a literal/constant array. It will be emitted as a flattened array of
  // data in an Attribute attached to a `toy.constant` operation.
  // See documentation on [Attributes](LangRef.md#attributes) for more details.
  // Here is an excerpt:
  //
  //   Attributes are the mechanism for specifying constant data in MLIR in
  //   places where a variable is never allowed [...]. They consist of a name
  //   and a [concrete attribute value](#attribute-values). It is possible to
  //   attach attributes to operations, functions, and function arguments. The
  //   set of expected attributes, their structure, and their interpretation
  //   are all contextually dependent on what they are attached to.
  //
  // Example, the source level statement:
  //   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  // will be converted to:
  //   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  //     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  //      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> memref<2x3xf64>
  //
  mlir::Value *mlirGen(LiteralExprAST &lit) {
    auto location = loc(lit.loc());
    // The attribute is a vector with an attribute per element (number) in the
    // array, see `collectData()` below for more details.
    std::vector<mlir::Attribute> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                 std::multiplies<int>()));
    collectData(lit, data);

    // FIXME: using a tensor type is a HACK here.
    // Can we do differently without registering a dialect? Using a string blob?
    mlir::Type elementType = mlir::FloatType::getF64(&context);
    auto dataType = builder->getTensorType(lit.getDims(), elementType);

    // This is the actual attribute that actually hold the list of values for
    // this array literal.
    auto dataAttribute = builder->getDenseElementsAttr(dataType, data)
                             .cast<mlir::DenseElementsAttr>();

    // Build the MLIR op `toy.constant`, only boilerplate below.
    return builder->create<ConstantOp>(location, lit.getDims(), dataAttribute)
        .getResult();
  }

  // Recursive helper function to accumulate the data that compose an array
  // literal. It flattens the nested structure in the supplied vector. For
  // example with this array:
  //  [[1, 2], [3, 4]]
  // we will generate:
  //  [ 1, 2, 3, 4 ]
  // Individual numbers are wrapped in a light wrapper `mlir::FloatAttr`.
  // Attributes are the way MLIR attaches constant to operations and functions.
  void collectData(ExprAST &expr, std::vector<mlir::Attribute> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }
    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    mlir::Type elementType = mlir::FloatType::getF64(&context);
    auto attr = mlir::FloatAttr::getChecked(
        elementType, cast<NumberExprAST>(expr).getValue(), loc(expr.loc()));
    data.push_back(attr);
  }

  // Emit a call expression. It emits specific operations for the `transpose`
  // builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value *mlirGen(CallExprAST &call) {
    auto location = loc(call.loc());
    std::string callee = call.getCallee();
    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: toy.transpose "
                            "does not accept multiple arguments");
        return nullptr;
      }
      mlir::Value *arg = mlirGen(*call.getArgs()[0]);
      return builder->create<TransposeOp>(location, arg).getResult();
    }

    // Codegen the operands first
    SmallVector<mlir::Value *, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto *arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }
    // Calls to user-defined function are mapped to a custom call that takes
    // the callee name as an attribute.
    return builder->create<GenericCallOp>(location, call.getCallee(), operands)
        .getResult();
  }

  // Emit a call expression. It emits specific operations for two builtins:
  // transpose(x) and print(x). Other identifiers are assumed to be user-defined
  // functions. Return false on failure.
  bool mlirGen(PrintExprAST &call) {
    auto *arg = mlirGen(*call.getArg());
    if (!arg)
      return false;
    auto location = loc(call.loc());
    builder->create<PrintOp>(location, arg);
    return true;
  }

  // Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value *mlirGen(NumberExprAST &num) {
    auto location = loc(num.loc());
    mlir::Type elementType = mlir::FloatType::getF64(&context);
    auto attr = mlir::FloatAttr::getChecked(elementType, num.getValue(),
                                            loc(num.loc()));
    return builder->create<ConstantOp>(location, attr).getResult();
  }

  // Dispatch codegen for the right expression subclass using RTTI.
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

  // Handle a variable declaration, we'll codegen the expression that forms the
  // initializer and record the value in the symbol table before returning it.
  // Future expressions will be able to reference this variable through symbol
  // table lookup.
  mlir::Value *mlirGen(VarDeclExprAST &vardecl) {
    mlir::Value *value = nullptr;
    auto location = loc(vardecl.loc());
    if (auto init = vardecl.getInitVal()) {
      value = mlirGen(*init);
      if (!value)
        return nullptr;
      // We have the initializer value, but in case the variable was declared
      // with specific shape, we emit a "reshape" operation. It will get
      // optimized out later as needed.
      if (!vardecl.getType().shape.empty()) {
        value = builder
                    ->create<ReshapeOp>(
                        location, value,
                        getType(vardecl.getType()).cast<ToyArrayType>())
                    .getResult();
      }
    } else {
      emitError(loc(vardecl.loc()),
                "Missing initializer in variable declaration");
      return nullptr;
    }
    // Register the value in the symbol table
    declare(vardecl.getName(), value);
    return value;
  }

  /// Codegen a list of expression, return false if one of them hit an error.
  bool mlirGen(ExprASTList &blockAST) {
    ScopedHashTableScope<llvm::StringRef, mlir::Value *> var_scope(symbolTable);
    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return false;
        continue;
      }
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get())) {
        if (!mlirGen(*ret))
          return false;
        return true;
      }
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (!mlirGen(*print))
          return false;
        continue;
      }
      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return false;
    }
    return true;
  }

  /// Build a type from a list of shape dimensions. Types are `array` followed
  /// by an optional dimension list, example: array<2, 2>
  /// They are wrapped in a `toy` dialect (see next chapter) and get printed:
  ///   !toy.array<2, 2>
  template <typename T> mlir::Type getType(T shape) {
    SmallVector<int64_t, 8> shape64(shape.begin(), shape.end());
    return ToyArrayType::get(&context, shape64);
  }

  /// Build an MLIR type from a Toy AST variable type
  /// (forward to the generic getType(T) above).
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
