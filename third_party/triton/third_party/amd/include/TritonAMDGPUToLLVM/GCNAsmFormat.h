/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_GCNASMFORMAT_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_GCNASMFORMAT_H_

#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {

class ConversionPatternRewriter;
class Location;

} // namespace mlir

namespace mlir::triton {
using llvm::StringRef;

class GCNInstr;
class GCNInstrCommon;
class GCNInstrExecution;

// GCNBuilder helps to manage a GCN asm program consists of one or multiple
// instructions.
//
// A helper for building an ASM program, the objective of GCNBuilder is to give
// a thin encapsulation and make the ASM code for MLIR LLVM Dialect more clear.
// Currently, several factors are introduced to reduce the need for mixing
// string and C++ if-else code.
//
// Usage:
// To create a multiplcation operation
//
//
// GCNBuilder gcnBuilder;
// unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
//
// const std::string readConstraint = "v";
// const std::string writeConstraint = "=v";
// auto res = gcnBuilder.newOperand(writeConstraint);
// auto lhs = gcnBuilder.newOperand(operands[0], readConstraint);
// auto rhs = gcnBuilder.newOperand(operands[1], readConstraint);
//
// create inst
// auto &mul_inst =
// gcnBuilder.create<GCNInstr>("v_mul")->float_op_type(bitwidth);
//
// launch insts
// mul_inst(res, lhs, rhs);
//
// return result
// Value ret = gcnBuilder.launch(rewriter, loc, elemTy, false);
// return ret;
// To get the asm code:
// builder.dump()
//
// To get all the mlir::Value used in the GCN code,
//
// builder.getAllMlirArgs() // get {pVal, iVal, jVal, kVal}
//
// To get the string containing all the constraints with "," separated,
// builder.getConstraints() // get "=v,v,v"
//
// GCNBuilder can build a GCN asm with multiple instructions, sample code:
//
// GCNBuilder builder;
// auto &rcp = gcnBuilder.create<GCNInstr>("v_rcp")->float_op_type(bitwidth);
// auto &mul_inst =
// gcnBuilder.create<GCNInstr>("v_mul")->float_op_type(bitwidth);
//
// rcp(...);
// mul_inst(...);
// This will get a GCN code with two instructions.
//
// Similar to a C function, a declared GCNInstr instance can be launched
// multiple times with different operands, e.g.
//
//   auto &mul_inst =
//   gcnBuilder.create<GCNInstr>("v_mul")->float_op_type(bitwidth); mul_inst(...
//   some operands ...); mul_inst(... some different operands ...);
//
// Finally, we will get a GCN code with two mov instructions.
//
// There are several derived instruction type for typical instructions, for
// example, the GCNIOInstr for ld and st instructions.
struct GCNBuilder {
  struct Operand {
    std::string constraint;
    Value value;
    int idx{-1};
    llvm::SmallVector<Operand *> list;
    std::function<std::string(int idx)> repr;

    // for list
    Operand() = default;
    Operand(const Operation &) = delete;
    Operand(Value value, StringRef constraint)
        : constraint(constraint), value(value) {}

    bool isList() const { return !value && constraint.empty(); }

    Operand *listAppend(Operand *arg) {
      list.push_back(arg);
      return this;
    }

    Operand *listGet(size_t nth) const {
      assert(nth < list.size());
      return list[nth];
    }

    std::string dump() const;
  };

  struct Modifier {
    Value value;
    std::string modifier;
    std::string arg;
    llvm::SmallVector<Modifier *> list;

    Modifier() = default;
    Modifier(const Operation &) = delete;
    Modifier(Value value, StringRef arg) : value(value), arg(arg) {}

    bool isList() const { return !value && modifier.empty(); }

    Modifier *listAppend(Modifier *arg) {
      list.push_back(arg);
      return this;
    }

    Modifier *listGet(size_t index) const {
      assert(index < list.size());
      return list[index];
    }

    std::string to_str() const {
      std::string str = modifier;
      if (!arg.empty()) {
        str += ":" + arg;
      }
      return str;
    }

    std::string dump() const;
  };

  template <typename INSTR = GCNInstr, typename... Args>
  INSTR *create(Args &&...args) {
    instrs.emplace_back(std::make_unique<INSTR>(this, args...));
    return static_cast<INSTR *>(instrs.back().get());
  }

  // Create a list of operands.
  Operand *newListOperand() { return newOperand(); }

  Operand *newListOperand(ArrayRef<std::pair<mlir::Value, std::string>> items) {
    auto *list = newOperand();
    for (auto &item : items) {
      list->listAppend(newOperand(item.first, item.second));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, mlir::Value val,
                          const std::string &constraint) {
    auto *list = newOperand();
    for (int i = 0; i < count; ++i) {
      list->listAppend(newOperand(val, constraint));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, const std::string &constraint) {
    auto *list = newOperand();
    for (int i = 0; i < count; ++i) {
      list->listAppend(newOperand(constraint));
    }
    return list;
  }

  // Create a new operand. It will not add to operand list.
  // @value: the MLIR value bind to this operand.
  // @constraint: ASM operand constraint, .e.g. "=r"
  // @formatter: extra format to represent this operand in ASM code, default is
  //             "%{0}".format(operand.idx).
  Operand *newOperand(mlir::Value value, StringRef constraint,
                      std::function<std::string(int idx)> formatter = nullptr);

  // Create a new operand which is written to, that is, the constraint starts
  // with "=", e.g. "=r".
  Operand *newOperand(StringRef constraint);

  // Create a constant integer operand.
  Operand *newConstantOperand(int v);
  // Create a constant operand with explicit code specified.
  Operand *newConstantOperand(const std::string &v);

  Operand *newAddrOperand(mlir::Value addr, StringRef constraint);

  Modifier *newModifier(StringRef modifier, StringRef arg);

  llvm::SmallVector<Operand *, 4> getAllArgs() const;

  llvm::SmallVector<Value, 4> getAllMLIRArgs() const;

  std::string getConstraints() const;

  std::string dump() const;

  mlir::Value launch(RewriterBase &rewriter, Location loc, Type resTy,
                     bool hasSideEffect = true, bool isAlignStack = false,
                     ArrayRef<Attribute> attrs = {}) const;

private:
  Operand *newOperand() {
    argArchive.emplace_back(std::make_unique<Operand>());
    return argArchive.back().get();
  }

  Modifier *newModifier() {
    modArchive.emplace_back(std::make_unique<Modifier>());
    return modArchive.back().get();
  }

  friend class GCNInstr;
  friend class GCNInstrCommon;

protected:
  llvm::SmallVector<std::unique_ptr<Operand>, 6> argArchive;
  llvm::SmallVector<std::unique_ptr<Modifier>, 2> modArchive;
  llvm::SmallVector<std::unique_ptr<GCNInstrCommon>, 2> instrs;
  llvm::SmallVector<std::unique_ptr<GCNInstrExecution>, 4> executions;
  int oprCounter{};
};

// GCN instruction common interface.
// Put the generic logic for all the instructions here.
struct GCNInstrCommon {
  explicit GCNInstrCommon(GCNBuilder *builder) : builder(builder) {}

  using Operand = GCNBuilder::Operand;
  using Modifier = GCNBuilder::Modifier;

  // clang-format off
  GCNInstrExecution& operator()() { return call({}, {}); }
  GCNInstrExecution& operator()(Operand* a) { return call({a}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b) { return call({a, b}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c) { return call({a, b, c}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d) { return call({a, b, c, d}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e) { return call({a, b, c, d, e}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f) { return call({a, b, c, d, e, f}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f, Operand* g) { return call({a, b, c, d, e, f, g}, {}); }
  // clang-format on

  // Set operands of this instruction.
  GCNInstrExecution &operator()(llvm::ArrayRef<Operand *> oprs,
                                llvm::ArrayRef<Modifier *> mods);

protected:
  GCNInstrExecution &call(llvm::ArrayRef<Operand *> oprs,
                          ArrayRef<Modifier *> mods);

  GCNBuilder *builder{};
  llvm::SmallVector<std::string, 4> instrParts;

  friend class GCNInstrExecution;
};

template <class ConcreteT> struct GCNInstrBase : public GCNInstrCommon {
  using Operand = GCNBuilder::Operand;
  using Modifier = GCNBuilder::Modifier;

  explicit GCNInstrBase(GCNBuilder *builder, const std::string &name)
      : GCNInstrCommon(builder) {
    o(name);
  }

  ConcreteT &o(const std::string &suffix, bool predicate = true) {
    if (predicate)
      instrParts.push_back(suffix);
    return *static_cast<ConcreteT *>(this);
  }
};

enum VectorWidth { Byte = 8, Short = 16, Dword = 32, Qword = 64 };

struct GCNInstr : public GCNInstrBase<GCNInstr> {
  using GCNInstrBase<GCNInstr>::GCNInstrBase;

  GCNInstr &float_op_type(int width) {
    switch (width) {
    case Byte:
      assert(Byte != width);
      break;
    case Short:
      o("f16");
      break;
    case Dword:
      o("f32");
      break;
    case Qword:
      o("f64");
      break;
    default:
      break;
    }
    return *this;
  }
};

struct GCNInstrExecution {
  using Operand = GCNBuilder::Operand;
  using Modifier = GCNBuilder::Modifier;

  llvm::SmallVector<Operand *> argsInOrder;
  llvm::SmallVector<Modifier *> mods;

  GCNInstrExecution() = default;
  explicit GCNInstrExecution(GCNInstrCommon *instr,
                             llvm::ArrayRef<Operand *> oprs,
                             llvm::ArrayRef<Modifier *> modifiers)
      : argsInOrder(oprs.begin(), oprs.end()), instr(instr),
        mods(modifiers.begin(), modifiers.end()) {}

  std::string dump() const;

  SmallVector<Operand *> getArgList() const;

  GCNInstrCommon *instr{};
};

struct GCNMemInstr : public GCNInstrBase<GCNMemInstr> {
  using GCNInstrBase<GCNMemInstr>::GCNInstrBase;
  // Add specific type suffix to instruction

  GCNMemInstr &load_type(int width) {
    switch (width) {
    case Byte:
      o("ubyte");
      break;
    case Short:
      o("ushort");
      break;
    case Dword:
      o("dword");
      break;
    case Qword:
      o("dwordx2");
      break;
    default:
      break;
    }
    return *this;
  }

  GCNMemInstr &store_type(int width) {
    switch (width) {
    case Byte:
      o("byte");
      break;
    case Short:
      o("short");
      break;
    case Dword:
      o("dword");
      break;
    case Qword:
      o("dwordx2");
      break;
    default:
      break;
    }
    return *this;
  }
};

} // namespace mlir::triton

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_GCNASMFORMAT_H_
