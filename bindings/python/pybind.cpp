#include "third_party/llvm/llvm/include/llvm/ADT/SmallVector.h"
#include "third_party/llvm/llvm/include/llvm/ADT/StringRef.h"
#include "third_party/llvm/llvm/include/llvm/IR/Module.h"
#include "third_party/llvm/llvm/include/llvm/Support/TargetSelect.h"
#include "third_party/llvm/llvm/include/llvm/Support/raw_ostream.h"
#include <cstddef>

#include "third_party/llvm/llvm/projects/google_mlir/include/mlir-c/Core.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/EDSC/MLIREmitter.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/EDSC/Types.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/ExecutionEngine/ExecutionEngine.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Module.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Target/LLVMIR.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Transforms/Passes.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

#include "mlir/IR/Function.h"
#include "mlir/IR/Types.h"

static bool inited = [] {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  return true;
}();

namespace mlir {
namespace edsc {
namespace python {

static std::vector<std::unique_ptr<mlir::Pass>> getDefaultPasses(
    const std::vector<const mlir::PassInfo *> &mlirPassInfoList = {}) {
  std::vector<std::unique_ptr<mlir::Pass>> passList;
  passList.reserve(mlirPassInfoList.size() + 4);
  // Run each of the passes that were selected.
  for (const auto *passInfo : mlirPassInfoList) {
    passList.emplace_back(passInfo->createPass());
  }
  // Append the extra passes for lowering to MLIR.
  passList.emplace_back(mlir::createConstantFoldPass());
  passList.emplace_back(mlir::createCSEPass());
  passList.emplace_back(mlir::createCanonicalizerPass());
  passList.emplace_back(mlir::createLowerAffinePass());
  return passList;
}

// Run the passes sequentially on the given module.
// Return `nullptr` immediately if any of the passes fails.
static bool runPasses(const std::vector<std::unique_ptr<mlir::Pass>> &passes,
                      Module *module) {
  for (const auto &pass : passes) {
    mlir::PassResult result = pass->runOnModule(module);
    if (result == mlir::PassResult::Failure || module->verify()) {
      llvm::errs() << "Pass failed\n";
      return true;
    }
  }
  return false;
}

namespace py = pybind11;

struct PythonBindable;
struct PythonExpr;
struct PythonStmt;

struct PythonFunction {
  PythonFunction() : function{nullptr} {}
  PythonFunction(mlir_func_t f) : function{f} {}
  PythonFunction(mlir::Function *f) : function{f} {}
  operator mlir_func_t() { return function; }
  std::string str() {
    mlir::Function *f = reinterpret_cast<mlir::Function *>(function);
    std::string res;
    llvm::raw_string_ostream os(res);
    f->print(os);
    return res;
  }
  mlir_func_t function;
};

struct PythonType {
  PythonType() : type{nullptr} {}
  PythonType(mlir_type_t t) : type{t} {}
  operator mlir_type_t() { return type; }
  std::string str() {
    mlir::Type f = mlir::Type::getFromOpaquePointer(type);
    std::string res;
    llvm::raw_string_ostream os(res);
    f.print(os);
    return res;
  }
  mlir_type_t type;
};

/// Trivial C++ wrappers make use of the EDSC C API.
struct PythonMLIRModule {
  PythonMLIRModule() : mlirContext(), module(new mlir::Module(&mlirContext)) {}

  PythonType makeScalarType(const std::string &mlirElemType,
                            unsigned bitwidth) {
    return ::makeScalarType(mlir_context_t{&mlirContext}, mlirElemType.c_str(),
                            bitwidth);
  }
  PythonType makeMemRefType(PythonType elemType, std::vector<int64_t> sizes) {
    return ::makeMemRefType(mlir_context_t{&mlirContext}, elemType,
                            int64_list_t{sizes.data(), sizes.size()});
  }
  PythonFunction makeFunction(const std::string &name,
                              std::vector<PythonType> &inputTypes,
                              std::vector<PythonType> &outputTypes) {
    std::vector<mlir_type_t> ins(inputTypes.begin(), inputTypes.end());
    std::vector<mlir_type_t> outs(outputTypes.begin(), outputTypes.end());
    auto funcType = ::makeFunctionType(
        mlir_context_t{&mlirContext}, mlir_type_list_t{ins.data(), ins.size()},
        mlir_type_list_t{outs.data(), outs.size()});
    auto *func = new mlir::Function(
        UnknownLoc::get(&mlirContext), name,
        mlir::Type::getFromOpaquePointer(funcType).cast<FunctionType>());
    func->addEntryBlock();
    module->getFunctions().push_back(func);
    return mlir_func_t{func};
  }

  void compile() {
    auto created = mlir::ExecutionEngine::create(module.get());
    llvm::handleAllErrors(created.takeError(),
                          [](const llvm::ErrorInfoBase &b) {
                            b.log(llvm::errs());
                            assert(false);
                          });
    engine = std::move(*created);
  }

  std::string getIR() {
    std::string res;
    llvm::raw_string_ostream os(res);
    module->print(os);
    return res;
  }

  uint64_t getEngineAddress() {
    assert(engine && "module must be compiled into engine first");
    return reinterpret_cast<uint64_t>(reinterpret_cast<void *>(engine.get()));
  }

private:
  mlir::MLIRContext mlirContext;
  // One single module in a python-exposed MLIRContext for now.
  std::unique_ptr<mlir::Module> module;
  std::unique_ptr<mlir::ExecutionEngine> engine;
};

struct ContextManager {
  void enter() { context = new ScopedEDSCContext(); }
  void exit(py::object, py::object, py::object) {
    delete context;
    context = nullptr;
  }
  mlir::edsc::ScopedEDSCContext *context;
};

struct PythonExpr {
  PythonExpr() : expr{nullptr} {}
  PythonExpr(const PythonBindable &bindable);
  PythonExpr(const edsc_expr_t &expr) : expr{expr} {}
  operator edsc_expr_t() { return expr; }
  std::string str() {
    assert(expr && "unexpected empty expr");
    return Expr(*this).str();
  }
  edsc_expr_t expr;
};

struct PythonBindable : public PythonExpr {
  PythonBindable() : PythonExpr(edsc_expr_t{makeBindable()}) {}
  PythonBindable(PythonExpr expr) : PythonExpr(expr) {
    assert(Expr(expr).isa<Bindable>() && "Expected Bindable");
  }
  std::string str() {
    assert(expr && "unexpected empty expr");
    return Expr(expr).str();
  }
};

struct PythonStmt {
  PythonStmt() : stmt{nullptr} {}
  PythonStmt(const edsc_stmt_t &stmt) : stmt{stmt} {}
  PythonStmt(const PythonExpr &e) : stmt{makeStmt(e.expr)} {}
  operator edsc_stmt_t() { return stmt; }
  std::string str() {
    assert(stmt && "unexpected empty stmt");
    return Stmt(stmt).str();
  }
  edsc_stmt_t stmt;
};

struct PythonIndexed : public edsc_indexed_t {
  PythonIndexed() : edsc_indexed_t{makeIndexed(PythonBindable())} {}
  PythonIndexed(PythonExpr e) : edsc_indexed_t{makeIndexed(e)} {}
  PythonIndexed(PythonBindable b) : edsc_indexed_t{makeIndexed(b)} {}
  operator PythonExpr() { return PythonExpr(base); }
};

struct MLIRFunctionEmitter {
  MLIRFunctionEmitter(PythonFunction f)
      : currentFunction(reinterpret_cast<mlir::Function *>(f.function)),
        currentBuilder(currentFunction),
        emitter(&currentBuilder, currentFunction->getLoc()) {}

  PythonExpr bindConstantBF16(double value);
  PythonExpr bindConstantF16(float value);
  PythonExpr bindConstantF32(float value);
  PythonExpr bindConstantF64(double value);
  PythonExpr bindConstantInt(int64_t value, unsigned bitwidth);
  PythonExpr bindConstantIndex(int64_t value);
  PythonExpr bindFunctionArgument(unsigned pos);
  py::list bindFunctionArguments();
  py::list bindFunctionArgumentView(unsigned pos);
  py::list bindMemRefShape(PythonExpr boundMemRef);
  py::list bindIndexedMemRefShape(PythonIndexed boundMemRef) {
    return bindMemRefShape(boundMemRef.base);
  }
  py::list bindMemRefView(PythonExpr boundMemRef);
  py::list bindIndexedMemRefView(PythonIndexed boundMemRef) {
    return bindMemRefView(boundMemRef.base);
  }
  void emit(PythonStmt stmt);

private:
  mlir::Function *currentFunction;
  mlir::FuncBuilder currentBuilder;
  mlir::edsc::MLIREmitter emitter;
  edsc_mlir_emitter_t c_emitter;
};

static edsc_stmt_list_t makeCStmts(llvm::SmallVectorImpl<edsc_stmt_t> &owning,
                                   const py::list &stmts) {
  for (auto &inp : stmts) {
    owning.push_back(edsc_stmt_t{inp.cast<PythonStmt>()});
  }
  return edsc_stmt_list_t{owning.data(), owning.size()};
}

static edsc_expr_list_t makeCExprs(llvm::SmallVectorImpl<edsc_expr_t> &owning,
                                   const py::list &exprs) {
  for (auto &inp : exprs) {
    owning.push_back(edsc_expr_t{inp.cast<PythonExpr>()});
  }
  return edsc_expr_list_t{owning.data(), owning.size()};
}

PythonExpr::PythonExpr(const PythonBindable &bindable) : expr{bindable.expr} {}

PythonExpr MLIRFunctionEmitter::bindConstantBF16(double value) {
  return ::bindConstantBF16(edsc_mlir_emitter_t{&emitter}, value);
}

PythonExpr MLIRFunctionEmitter::bindConstantF16(float value) {
  return ::bindConstantF16(edsc_mlir_emitter_t{&emitter}, value);
}

PythonExpr MLIRFunctionEmitter::bindConstantF32(float value) {
  return ::bindConstantF32(edsc_mlir_emitter_t{&emitter}, value);
}

PythonExpr MLIRFunctionEmitter::bindConstantF64(double value) {
  return ::bindConstantF64(edsc_mlir_emitter_t{&emitter}, value);
}

PythonExpr MLIRFunctionEmitter::bindConstantInt(int64_t value,
                                                unsigned bitwidth) {
  return ::bindConstantInt(edsc_mlir_emitter_t{&emitter}, value, bitwidth);
}

PythonExpr MLIRFunctionEmitter::bindConstantIndex(int64_t value) {
  return ::bindConstantIndex(edsc_mlir_emitter_t{&emitter}, value);
}

PythonExpr MLIRFunctionEmitter::bindFunctionArgument(unsigned pos) {
  return ::bindFunctionArgument(edsc_mlir_emitter_t{&emitter},
                                mlir_func_t{currentFunction}, pos);
}

PythonExpr getPythonType(edsc_expr_t e) { return PythonExpr(e); }

template <typename T> py::list makePyList(llvm::ArrayRef<T> owningResults) {
  py::list res;
  for (auto e : owningResults) {
    res.append(getPythonType(e));
  }
  return res;
}

py::list MLIRFunctionEmitter::bindFunctionArguments() {
  auto arity = getFunctionArity(mlir_func_t{currentFunction});
  llvm::SmallVector<edsc_expr_t, 8> owningResults(arity);
  edsc_expr_list_t results{owningResults.data(), owningResults.size()};
  ::bindFunctionArguments(edsc_mlir_emitter_t{&emitter},
                          mlir_func_t{currentFunction}, &results);
  return makePyList(ArrayRef<edsc_expr_t>{owningResults});
}

py::list MLIRFunctionEmitter::bindMemRefShape(PythonExpr boundMemRef) {
  auto rank = getBoundMemRefRank(edsc_mlir_emitter_t{&emitter}, boundMemRef);
  llvm::SmallVector<edsc_expr_t, 8> owningShapes(rank);
  edsc_expr_list_t resultShapes{owningShapes.data(), owningShapes.size()};
  ::bindMemRefShape(edsc_mlir_emitter_t{&emitter}, boundMemRef, &resultShapes);
  return makePyList(ArrayRef<edsc_expr_t>{owningShapes});
}

py::list MLIRFunctionEmitter::bindMemRefView(PythonExpr boundMemRef) {
  auto rank = getBoundMemRefRank(edsc_mlir_emitter_t{&emitter}, boundMemRef);
  // Own the PythonExpr for the arg as well as all its dims.
  llvm::SmallVector<edsc_expr_t, 8> owningLbs(rank);
  llvm::SmallVector<edsc_expr_t, 8> owningUbs(rank);
  llvm::SmallVector<edsc_expr_t, 8> owningSteps(rank);
  edsc_expr_list_t resultLbs{owningLbs.data(), owningLbs.size()};
  edsc_expr_list_t resultUbs{owningUbs.data(), owningUbs.size()};
  edsc_expr_list_t resultSteps{owningSteps.data(), owningSteps.size()};
  ::bindMemRefView(edsc_mlir_emitter_t{&emitter}, boundMemRef, &resultLbs,
                   &resultUbs, &resultSteps);
  py::list res;
  res.append(makePyList(ArrayRef<edsc_expr_t>{owningLbs}));
  res.append(makePyList(ArrayRef<edsc_expr_t>{owningUbs}));
  res.append(makePyList(ArrayRef<edsc_expr_t>{owningSteps}));
  return res;
}

void MLIRFunctionEmitter::emit(PythonStmt stmt) {
  emitter.emitStmt(Stmt(stmt));
}

PYBIND11_MODULE(pybind, m) {
  m.doc() =
      "Python bindings for MLIR Embedded Domain-Specific Components (EDSCs)";
  m.def("version", []() { return "EDSC Python extensions v0.0"; });
  m.def("initContext",
        []() { return static_cast<void *>(new ScopedEDSCContext()); });
  m.def("deleteContext",
        [](void *ctx) { delete reinterpret_cast<ScopedEDSCContext *>(ctx); });

  m.def("Block", [](const py::list &stmts) {
    SmallVector<edsc_stmt_t, 8> owning;
    return PythonStmt(::StmtList(makeCStmts(owning, stmts)));
  });
  m.def("For", [](const py::list &ivs, const py::list &lbs, const py::list &ubs,
                  const py::list &steps, const py::list &stmts) {
    SmallVector<edsc_expr_t, 8> owningIVs;
    SmallVector<edsc_expr_t, 8> owningLBs;
    SmallVector<edsc_expr_t, 8> owningUBs;
    SmallVector<edsc_expr_t, 8> owningSteps;
    SmallVector<edsc_stmt_t, 8> owningStmts;
    return PythonStmt(
        ::ForNest(makeCExprs(owningIVs, ivs), makeCExprs(owningLBs, lbs),
                  makeCExprs(owningUBs, ubs), makeCExprs(owningSteps, steps),
                  makeCStmts(owningStmts, stmts)));
  });
  m.def("For", [](PythonExpr iv, PythonExpr lb, PythonExpr ub, PythonExpr step,
                  const py::list &stmts) {
    SmallVector<edsc_stmt_t, 8> owning;
    return PythonStmt(::For(iv, lb, ub, step, makeCStmts(owning, stmts)));
  });
  m.def("Select", [](PythonExpr cond, PythonExpr e1, PythonExpr e2) {
    return PythonExpr(::Select(cond, e1, e2));
  });
  m.def("Return", []() {
    return PythonStmt(::Return(edsc_expr_list_t{nullptr, 0}));
  });
  m.def("Return", [](const py::list &returns) {
    SmallVector<edsc_expr_t, 8> owningExprs;
    return PythonStmt(::Return(makeCExprs(owningExprs, returns)));
  });

#define DEFINE_PYBIND_BINARY_OP(PYTHON_NAME, C_NAME)                           \
  m.def(PYTHON_NAME, [](PythonExpr e1, PythonExpr e2) {                        \
    return PythonExpr(::C_NAME(e1, e2));                                       \
  });

  DEFINE_PYBIND_BINARY_OP("Add", Add);
  DEFINE_PYBIND_BINARY_OP("Mul", Mul);
  DEFINE_PYBIND_BINARY_OP("Sub", Sub);
  // DEFINE_PYBIND_BINARY_OP("Div", Div);
  DEFINE_PYBIND_BINARY_OP("LT", LT);
  DEFINE_PYBIND_BINARY_OP("LE", LE);
  DEFINE_PYBIND_BINARY_OP("GT", GT);
  DEFINE_PYBIND_BINARY_OP("GE", GE);
  DEFINE_PYBIND_BINARY_OP("EQ", EQ);
  DEFINE_PYBIND_BINARY_OP("NE", NE);
  DEFINE_PYBIND_BINARY_OP("And", And);
  DEFINE_PYBIND_BINARY_OP("Or", Or);

#undef DEFINE_PYBIND_BINARY_OP

#define DEFINE_PYBIND_UNARY_OP(PYTHON_NAME, C_NAME)                            \
  m.def(PYTHON_NAME, [](PythonExpr e1) { return PythonExpr(::C_NAME(e1)); });

  DEFINE_PYBIND_UNARY_OP("Negate", Negate);

#undef DEFINE_PYBIND_UNARY_OP

  py::class_<PythonFunction>(m, "Function",
                             "Wrapping class for mlir::Function.")
      .def(py::init<PythonFunction>())
      .def("__str__", &PythonFunction::str);

  py::class_<PythonType>(m, "Type", "Wrapping class for mlir::Type.")
      .def(py::init<PythonType>())
      .def("__str__", &PythonType::str);

  py::class_<PythonMLIRModule>(
      m, "MLIRModule",
      "An MLIRModule is the abstraction that owns the allocations to support "
      "compilation of a single mlir::Module into an ExecutionEngine backed by "
      "the LLVM ORC JIT. A typical flow consists in creating an MLIRModule, "
      "adding functions, compiling the module to obtain an ExecutionEngine on "
      "which named functions may be called. For now the only means to retrieve "
      "the ExecutionEngine is by calling `get_engine_address`. This mode of "
      "execution is limited to passing the pointer to C++ where the function "
      "is called. Extending the API to allow calling JIT compiled functions "
      "directly require integration with a tensor library (e.g. numpy). This "
      "is left as the prerogative of libraries and frameworks for now.")
      .def(py::init<>())
      .def("make_function", &PythonMLIRModule::makeFunction,
           "Creates a new mlir::Function in the current mlir::Module.")
      .def(
          "make_scalar_type",
          [](PythonMLIRModule &instance, const std::string &type,
             unsigned bitwidth) {
            return instance.makeScalarType(type, bitwidth);
          },
          py::arg("type"), py::arg("bitwidth") = 0,
          "Returns a scalar mlir::Type using the following convention:\n"
          "  - makeScalarType(c, \"bf16\") return an `mlir::Type::getBF16`\n"
          "  - makeScalarType(c, \"f16\") return an `mlir::Type::getF16`\n"
          "  - makeScalarType(c, \"f32\") return an `mlir::Type::getF32`\n"
          "  - makeScalarType(c, \"f64\") return an `mlir::Type::getF64`\n"
          "  - makeScalarType(c, \"index\") return an `mlir::Type::getIndex`\n"
          "  - makeScalarType(c, \"i\", bitwidth) return an "
          "`mlir::Type::getInteger(bitwidth)`\n\n"
          " No other combinations are currently supported.")
      .def("make_memref_type", &PythonMLIRModule::makeMemRefType,
           "Returns an mlir::MemRefType of an elemental scalar. -1 is used to "
           "denote symbolic dimensions in the resulting memref shape.")
      .def("compile", &PythonMLIRModule::compile,
           "Compiles the mlir::Module to LLVMIR a creates new opaque "
           "ExecutionEngine backed by the ORC JIT.")
      .def("get_ir", &PythonMLIRModule::getIR,
           "Returns a dump of the MLIR representation of the module. This is "
           "used for serde to support out-of-process execution as well as "
           "debugging purposes.")
      .def("get_engine_address", &PythonMLIRModule::getEngineAddress,
           "Returns the address of the compiled ExecutionEngine. This is used "
           "for in-process execution.");

  py::class_<ContextManager>(
      m, "ContextManager",
      "An EDSC context manager is the memory arena containing all the EDSC "
      "allocations.\nUsage:\n\n"
      "with E.ContextManager() as _:\n  i = E.Expr(E.Bindable())\n  ...")
      .def(py::init<>())
      .def("__enter__", &ContextManager::enter)
      .def("__exit__", &ContextManager::exit);

  py::class_<MLIRFunctionEmitter>(
      m, "MLIRFunctionEmitter",
      "An MLIRFunctionEmitter is used to fill an empty function body. This is "
      "a staged process:\n"
      "  1. create or retrieve an mlir::Function `f` with an empty body;\n"
      "  2. make an `MLIRFunctionEmitter(f)` to build the current function;\n"
      "  3. create leaf Expr that are either Bindable or already Expr that are"
      "     bound to constants and function arguments by using methods of "
      "     `MLIRFunctionEmitter`;\n"
      "  4. build the function body using Expr, Indexed and Stmt;\n"
      "  5. emit the MLIR to implement the function body.")
      .def(py::init<PythonFunction>())
      .def("bind_constant_bf16", &MLIRFunctionEmitter::bindConstantBF16)
      .def("bind_constant_f16", &MLIRFunctionEmitter::bindConstantF16)
      .def("bind_constant_f32", &MLIRFunctionEmitter::bindConstantF32)
      .def("bind_constant_f64", &MLIRFunctionEmitter::bindConstantF64)
      .def("bind_constant_int", &MLIRFunctionEmitter::bindConstantInt)
      .def("bind_constant_index", &MLIRFunctionEmitter::bindConstantIndex)
      .def("bind_function_argument", &MLIRFunctionEmitter::bindFunctionArgument,
           "Returns an Expr that has been bound to a positional argument in "
           "the current Function.")
      .def("bind_function_arguments",
           &MLIRFunctionEmitter::bindFunctionArguments,
           "Returns a list of Expr where each Expr has been bound to the "
           "corresponding positional argument in the current Function.")
      .def("bind_memref_shape", &MLIRFunctionEmitter::bindMemRefShape,
           "Returns a list of Expr where each Expr has been bound to the "
           "corresponding dimension of the memref.")
      .def("bind_memref_view", &MLIRFunctionEmitter::bindMemRefView,
           "Returns three lists (lower bound, upper bound and step) of Expr "
           "where each triplet of Expr has been bound to the minimal offset, "
           "extent and stride of the corresponding dimension of the memref.")
      .def("bind_indexed_shape", &MLIRFunctionEmitter::bindIndexedMemRefShape,
           "Same as bind_memref_shape but returns a list of `Indexed` that "
           "support load and store operations")
      .def("bind_indexed_view", &MLIRFunctionEmitter::bindIndexedMemRefView,
           "Same as bind_memref_view but returns lists of `Indexed` that "
           "support load and store operations")
      .def("emit", &MLIRFunctionEmitter::emit,
           "Emits the MLIR for the EDSC expressions and statements in the "
           "current function body.");

  py::class_<PythonExpr>(m, "Expr", "Wrapping class for mlir::edsc::Expr")
      .def(py::init<PythonBindable>())
      .def("__add__", [](PythonExpr e1,
                         PythonExpr e2) { return PythonExpr(::Add(e1, e2)); })
      .def("__sub__", [](PythonExpr e1,
                         PythonExpr e2) { return PythonExpr(::Sub(e1, e2)); })
      .def("__mul__", [](PythonExpr e1,
                         PythonExpr e2) { return PythonExpr(::Mul(e1, e2)); })
      // .def("__div__", [](PythonExpr e1, PythonExpr e2) { return
      // PythonExpr(::Div(e1, e2)); })
      .def("__lt__", [](PythonExpr e1,
                        PythonExpr e2) { return PythonExpr(::LT(e1, e2)); })
      .def("__le__", [](PythonExpr e1,
                        PythonExpr e2) { return PythonExpr(::LE(e1, e2)); })
      .def("__gt__", [](PythonExpr e1,
                        PythonExpr e2) { return PythonExpr(::GT(e1, e2)); })
      .def("__ge__", [](PythonExpr e1,
                        PythonExpr e2) { return PythonExpr(::GE(e1, e2)); })
      .def("__eq__", [](PythonExpr e1,
                        PythonExpr e2) { return PythonExpr(::EQ(e1, e2)); })
      .def("__ne__", [](PythonExpr e1,
                        PythonExpr e2) { return PythonExpr(::NE(e1, e2)); })
      .def("__and__", [](PythonExpr e1,
                         PythonExpr e2) { return PythonExpr(::And(e1, e2)); })
      .def("__or__", [](PythonExpr e1,
                        PythonExpr e2) { return PythonExpr(::Or(e1, e2)); })
      .def("__invert__", [](PythonExpr e) { return PythonExpr(::Negate(e)); })
      .def("__str__", &PythonExpr::str,
           R"DOC(Returns the string value for the Expr)DOC");

  py::class_<PythonBindable>(
      m, "Bindable",
      "Wrapping class for mlir::edsc::Bindable.\nA Bindable is a special Expr "
      "that can be bound manually to specific MLIR SSA Values.")
      .def(py::init<>())
      .def("__str__", &PythonBindable::str);

  py::class_<PythonStmt>(m, "Stmt", "Wrapping class for mlir::edsc::Stmt.")
      .def(py::init<PythonExpr>())
      .def("__str__", &PythonStmt::str,
           R"DOC(Returns the string value for the Expr)DOC");

  py::class_<PythonIndexed>(
      m, "Indexed",
      "Wrapping class for mlir::edsc::Indexed.\nAn Indexed is a wrapper class "
      "that support load and store operations.")
      .def(py::init<>(), R"DOC(Build from fresh Bindable)DOC")
      .def(py::init<PythonExpr>(), R"DOC(Build from existing Expr)DOC")
      .def(py::init<PythonBindable>(), R"DOC(Build from existing Bindable)DOC")
      .def(
          "load",
          [](PythonIndexed &instance, const py::list &indices) {
            SmallVector<edsc_expr_t, 8> owning;
            return PythonExpr(Load(instance, makeCExprs(owning, indices)));
          },
          R"DOC(Returns an Expr that loads from an Indexed)DOC")
      .def(
          "store",
          [](PythonIndexed &instance, const py::list &indices,
             PythonExpr value) {
            SmallVector<edsc_expr_t, 8> owning;
            return PythonStmt(
                Store(value, instance, makeCExprs(owning, indices)));
          },
          R"DOC(Returns the Stmt that stores into an Indexed)DOC");
}

} // namespace python
} // namespace edsc
} // namespace mlir
