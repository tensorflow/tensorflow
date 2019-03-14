"""Python2 and 3 test for the MLIR EDSC C API and Python bindings"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import google_mlir.bindings.python.pybind as E

class EdscTest(unittest.TestCase):

  def setUp(self):
    self.module = E.MLIRModule()
    self.boolType = self.module.make_scalar_type("i", 1)
    self.i32Type = self.module.make_scalar_type("i", 32)
    self.f32Type = self.module.make_scalar_type("f32")
    self.indexType = self.module.make_index_type()

  def testFunctionContext(self):
    with self.module.function_context("foo", [], []):
      pass
    self.assertIsNotNone(self.module.get_function("foo"))

  def testMultipleFunctions(self):
    with self.module.function_context("foo", [], []):
      E.constant_index(0)
    code = str(self.module)
    self.assertIn("func @foo()", code)
    self.assertIn("  %c0 = constant 0 : index", code)

    with self.module.function_context("bar", [], []):
      E.constant_index(42)
    code = str(self.module)
    barPos = code.find("func @bar()")
    c42Pos = code.find("%c42 = constant 42 : index")
    self.assertNotEqual(barPos, -1)
    self.assertNotEqual(c42Pos, -1)
    self.assertGreater(c42Pos, barPos)

  def testFunctionArgs(self):
    with self.module.function_context("foo", [self.f32Type, self.f32Type],
                                      [self.indexType]) as fun:
      pass
    code = str(fun)
    self.assertIn("func @foo(%arg0: f32, %arg1: f32) -> index", code)

  def testLoopContext(self):
    with self.module.function_context("foo", [], []) as fun:
      lhs = E.constant_index(0)
      rhs = E.constant_index(42)
      with E.LoopContext(lhs, rhs, 1) as i:
        lhs + rhs + i
        with E.LoopContext(rhs, rhs + rhs, 2) as j:
          x = i + j
    code = str(fun)
    # TODO(zinenko,ntv): use FileCheck for these tests
    self.assertIn(
        '  "for"() {lower_bound: () -> (0), step: 1 : index, upper_bound: () -> (42)} : () -> () {\n',
        code)
    self.assertIn("  ^bb1(%i0: index):", code)
    self.assertIn(
        '    "for"(%c42, %2) {lower_bound: (d0) -> (d0), step: 2 : index, upper_bound: (d0) -> (d0)} : (index, index) -> () {\n',
        code)
    self.assertIn("    ^bb2(%i1: index):", code)
    self.assertIn(
        '      %3 = "affine.apply"(%i0, %i1) {map: (d0, d1) -> (d0 + d1)} : (index, index) -> index',
        code)

  def testLoopNestContext(self):
    with self.module.function_context("foo", [], []) as fun:
      lbs = [E.constant_index(i) for i in range(4)]
      ubs = [E.constant_index(10 * i + 5) for i in range(4)]
      with E.LoopNestContext(lbs, ubs, [1, 3, 5, 7]) as (i, j, k, l):
        i + j + k + l

    code = str(fun)
    self.assertIn(
        ' "for"() {lower_bound: () -> (0), step: 1 : index, upper_bound: () -> (5)} : () -> () {\n',
        code)
    self.assertIn("  ^bb1(%i0: index):", code)
    self.assertIn(
        '    "for"() {lower_bound: () -> (1), step: 3 : index, upper_bound: () -> (15)} : () -> () {\n',
        code)
    self.assertIn("    ^bb2(%i1: index):", code)
    self.assertIn(
        '      "for"() {lower_bound: () -> (2), step: 5 : index, upper_bound: () -> (25)} : () -> () {\n',
        code)
    self.assertIn("      ^bb3(%i2: index):", code)
    self.assertIn(
        '        "for"() {lower_bound: () -> (3), step: 7 : index, upper_bound: () -> (35)} : () -> () {\n',
        code)
    self.assertIn("        ^bb4(%i3: index):", code)
    self.assertIn(
        '          %2 = "affine.apply"(%i0, %i1, %i2, %i3) {map: (d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)} : (index, index, index, index) -> index',
        code)

  def testBlockContext(self):
    with self.module.function_context("foo", [], []) as fun:
      cst = E.constant_index(42)
      with E.BlockContext():
        cst + cst
    code = str(fun)
    # Find positions of instructions and make sure they are in the block we
    # put them by comparing those positions.
    # TODO(zinenko,ntv): this (and tests below) should use FileCheck instead.
    c42pos = code.find("%c42 = constant 42 : index")
    bb1pos = code.find("^bb1:")
    c84pos = code.find('%0 = "affine.apply"() {map: () -> (84)} : () -> index')
    self.assertNotEqual(c42pos, -1)
    self.assertNotEqual(bb1pos, -1)
    self.assertNotEqual(c84pos, -1)
    self.assertGreater(bb1pos, c42pos)
    self.assertLess(bb1pos, c84pos)

  def testBlockContextAppend(self):
    with self.module.function_context("foo", [], []) as fun:
      E.constant_index(41)
      with E.BlockContext() as b:
        blk = b  # save block handle for later
        E.constant_index(0)
      E.constant_index(42)
      with E.BlockContext(E.appendTo(blk)):
        E.constant_index(1)
    code = str(fun)
    # Find positions of instructions and make sure they are in the block we put
    # them by comparing those positions.
    c41pos = code.find("%c41 = constant 41 : index")
    c42pos = code.find("%c42 = constant 42 : index")
    bb1pos = code.find("^bb1:")
    c0pos = code.find("%c0 = constant 0 : index")
    c1pos = code.find("%c1 = constant 1 : index")
    self.assertNotEqual(c41pos, -1)
    self.assertNotEqual(c42pos, -1)
    self.assertNotEqual(bb1pos, -1)
    self.assertNotEqual(c0pos, -1)
    self.assertNotEqual(c1pos, -1)
    self.assertGreater(bb1pos, c41pos)
    self.assertGreater(bb1pos, c42pos)
    self.assertLess(bb1pos, c0pos)
    self.assertLess(bb1pos, c1pos)

  def testBlockContextStandalone(self):
    with self.module.function_context("foo", [], []) as fun:
      blk1 = E.BlockContext()
      blk2 = E.BlockContext()
      with blk1:
        E.constant_index(0)
      with blk2:
        E.constant_index(56)
        E.constant_index(57)
      E.constant_index(41)
      with blk1:
        E.constant_index(1)
      E.constant_index(42)
    code = str(fun)
    # Find positions of instructions and make sure they are in the block we put
    # them by comparing those positions.
    c41pos = code.find("  %c41 = constant 41 : index")
    c42pos = code.find("  %c42 = constant 42 : index")
    bb1pos = code.find("^bb1:")
    c0pos = code.find("  %c0 = constant 0 : index")
    c1pos = code.find("  %c1 = constant 1 : index")
    bb2pos = code.find("^bb2:")
    c56pos = code.find("  %c56 = constant 56 : index")
    c57pos = code.find("  %c57 = constant 57 : index")
    self.assertNotEqual(c41pos, -1)
    self.assertNotEqual(c42pos, -1)
    self.assertNotEqual(bb1pos, -1)
    self.assertNotEqual(c0pos, -1)
    self.assertNotEqual(c1pos, -1)
    self.assertNotEqual(bb2pos, -1)
    self.assertNotEqual(c56pos, -1)
    self.assertNotEqual(c57pos, -1)
    self.assertGreater(bb1pos, c41pos)
    self.assertGreater(bb1pos, c42pos)
    self.assertLess(bb1pos, c0pos)
    self.assertLess(bb1pos, c1pos)
    self.assertGreater(bb2pos, c0pos)
    self.assertGreater(bb2pos, c1pos)
    self.assertGreater(bb2pos, bb1pos)
    self.assertLess(bb2pos, c56pos)
    self.assertLess(bb2pos, c57pos)

  def testBlockArguments(self):
    with self.module.function_context("foo", [], []) as fun:
      E.constant_index(42)
      with E.BlockContext([self.f32Type, self.f32Type]) as b:
        b.arg(0) + b.arg(1)
    code = str(fun)
    self.assertIn("%c42 = constant 42 : index", code)
    self.assertIn("^bb1(%0: f32, %1: f32):", code)
    self.assertIn("  %2 = addf %0, %1 : f32", code)

  def testBr(self):
    with self.module.function_context("foo", [], []) as fun:
      with E.BlockContext() as b:
        blk = b
        E.ret()
      E.br(blk)
    code = str(fun)
    self.assertIn("  br ^bb1", code)
    self.assertIn("^bb1:", code)
    self.assertIn("  return", code)

  def testBrDeclaration(self):
    with self.module.function_context("foo", [], []) as fun:
      blk = E.BlockContext()
      E.br(blk.handle())
      with blk:
        E.ret()
    code = str(fun)
    self.assertIn("  br ^bb1", code)
    self.assertIn("^bb1:", code)
    self.assertIn("  return", code)

  def testBrArgs(self):
    with self.module.function_context("foo", [], []) as fun:
      # Create an infinite loop.
      with E.BlockContext([self.indexType, self.indexType]) as b:
        E.br(b, [b.arg(1), b.arg(0)])
      E.br(b, [E.constant_index(0), E.constant_index(1)])
    code = str(fun)
    self.assertIn("  %c0 = constant 0 : index", code)
    self.assertIn("  %c1 = constant 1 : index", code)
    self.assertIn("  br ^bb1(%c0, %c1 : index, index)", code)
    self.assertIn("^bb1(%0: index, %1: index):", code)
    self.assertIn("  br ^bb1(%1, %0 : index, index)", code)

  def testCondBr(self):
    with self.module.function_context("foo", [self.boolType], []) as fun:
      with E.BlockContext() as blk1:
        E.ret([])
      with E.BlockContext([self.indexType]) as blk2:
        E.ret([])
      cst = E.constant_index(0)
      E.cond_br(fun.arg(0), blk1, [], blk2, [cst])

    code = str(fun)
    self.assertIn("cond_br %arg0, ^bb1, ^bb2(%c0 : index)", code)

  def testRet(self):
    with self.module.function_context("foo", [],
                                      [self.indexType, self.indexType]) as fun:
      c42 = E.constant_index(42)
      c0 = E.constant_index(0)
      E.ret([c42, c0])
    code = str(fun)
    self.assertIn("  %c42 = constant 42 : index", code)
    self.assertIn("  %c0 = constant 0 : index", code)
    self.assertIn("  return %c42, %c0 : index, index", code)

  def testSelectOp(self):
    with self.module.function_context("foo", [self.boolType],
                                      [self.i32Type]) as fun:
      a = E.constant_int(42, 32)
      b = E.constant_int(0, 32)
      E.ret([E.select(fun.arg(0), a, b)])

    code = str(fun)
    self.assertIn("%0 = select %arg0, %c42_i32, %c0_i32 : i32", code)

  def testCallOp(self):
    callee = self.module.declare_function("sqrtf", [self.f32Type],
                                          [self.f32Type])
    with self.module.function_context("call", [self.f32Type], []) as fun:
      funCst = E.constant_function(callee)
      funCst([fun.arg(0)]) + E.constant_float(42., self.f32Type)

    code = str(self.module)
    self.assertIn("func @sqrtf(f32) -> f32", code)
    self.assertIn("%f = constant @sqrtf : (f32) -> f32", code)
    self.assertIn("%0 = call_indirect %f(%arg0) : (f32) -> f32", code)

  def testBooleanOps(self):
    with self.module.function_context(
        "booleans", [self.boolType for _ in range(4)], []) as fun:
      i, j, k, l = (fun.arg(x) for x in range(4))
      stmt1 = (i < j) & (j >= k)
      stmt2 = ~(stmt1 | (k == l))

    code = str(fun)
    self.assertIn('%0 = cmpi "slt", %arg0, %arg1 : i1', code)
    self.assertIn('%1 = cmpi "sge", %arg1, %arg2 : i1', code)
    self.assertIn("%2 = muli %0, %1 : i1", code)
    self.assertIn('%3 = cmpi "eq", %arg2, %arg3 : i1', code)
    self.assertIn("%true = constant 1 : i1", code)
    self.assertIn("%4 = subi %true, %2 : i1", code)
    self.assertIn("%true_0 = constant 1 : i1", code)
    self.assertIn("%5 = subi %true_0, %3 : i1", code)
    self.assertIn("%6 = muli %4, %5 : i1", code)
    self.assertIn("%true_1 = constant 1 : i1", code)
    self.assertIn("%7 = subi %true_1, %6 : i1", code)
    self.assertIn("%true_2 = constant 1 : i1", code)
    self.assertIn("%8 = subi %true_2, %7 : i1", code)

  def testCustom(self):
    with self.module.function_context("custom", [self.indexType, self.f32Type],
                                      []) as fun:
      E.op("foo", [fun.arg(0)], [self.f32Type]) + fun.arg(1)
    code = str(fun)
    self.assertIn('%0 = "foo"(%arg0) : (index) -> f32', code)
    self.assertIn("%1 = addf %0, %arg1 : f32", code)

  def testConstants(self):
    with self.module.function_context("constants", [], []) as fun:
      E.constant_float(1.23, self.module.make_scalar_type("bf16"))
      E.constant_float(1.23, self.module.make_scalar_type("f16"))
      E.constant_float(1.23, self.module.make_scalar_type("f32"))
      E.constant_float(1.23, self.module.make_scalar_type("f64"))
      E.constant_int(1, 1)
      E.constant_int(123, 8)
      E.constant_int(123, 16)
      E.constant_int(123, 32)
      E.constant_int(123, 64)
      E.constant_index(123)
      E.constant_function(fun)

    code = str(fun)
    self.assertIn("constant 1.230000e+00 : bf16", code)
    self.assertIn("constant 1.230470e+00 : f16", code)
    self.assertIn("constant 1.230000e+00 : f32", code)
    self.assertIn("constant 1.230000e+00 : f64", code)
    self.assertIn("constant 1 : i1", code)
    self.assertIn("constant 123 : i8", code)
    self.assertIn("constant 123 : i16", code)
    self.assertIn("constant 123 : i32", code)
    self.assertIn("constant 123 : index", code)
    self.assertIn("constant @constants : () -> ()", code)

  def testIndexedValue(self):
    memrefType = self.module.make_memref_type(self.f32Type, [10, 42])
    with self.module.function_context("indexed", [memrefType],
                                      [memrefType]) as fun:
      A = E.IndexedValue(fun.arg(0))
      cst = E.constant_float(1., self.f32Type)
      with E.LoopNestContext(
          [E.constant_index(0), E.constant_index(0)],
          [E.constant_index(10), E.constant_index(42)], [1, 1]) as (i, j):
        A.store([i, j], A.load([i, j]) + cst)
      E.ret([fun.arg(0)])

    code = str(fun)
    self.assertIn(
        '"for"() {lower_bound: () -> (0), step: 1 : index, upper_bound: () -> (10)}',
        code)
    self.assertIn(
        '"for"() {lower_bound: () -> (0), step: 1 : index, upper_bound: () -> (42)}',
        code)
    self.assertIn("%0 = load %arg0[%i0, %i1] : memref<10x42xf32>", code)
    self.assertIn("%1 = addf %0, %cst : f32", code)
    self.assertIn("store %1, %arg0[%i0, %i1] : memref<10x42xf32>", code)

  def testMatrixMultiply(self):
    memrefType = self.module.make_memref_type(self.f32Type, [32, 32])
    with self.module.function_context(
        "matmul", [memrefType, memrefType, memrefType], []) as fun:
      A = E.IndexedValue(fun.arg(0))
      B = E.IndexedValue(fun.arg(1))
      C = E.IndexedValue(fun.arg(2))
      c0 = E.constant_index(0)
      c32 = E.constant_index(32)
      with E.LoopNestContext([c0, c0, c0], [c32, c32, c32], [1, 1, 1]) as (i, j,
                                                                           k):
        C.store([i, j], A.load([i, k]) * B.load([k, j]))
      E.ret([])

    code = str(fun)
    self.assertIn(
        '"for"() {lower_bound: () -> (0), step: 1 : index, upper_bound: () -> (32)} : () -> ()',
        code)
    self.assertIn("%0 = load %arg0[%i0, %i2] : memref<32x32xf32>", code)
    self.assertIn("%1 = load %arg1[%i2, %i1] : memref<32x32xf32>", code)
    self.assertIn("%2 = mulf %0, %1 : f32", code)
    self.assertIn("store %2, %arg2[%i0, %i1] : memref<32x32xf32>", code)

  def testBindables(self):
    with E.ContextManager():
      i = E.Expr(E.Bindable(self.i32Type))
      self.assertIn("$1", i.__str__())

  def testOneExpr(self):
    with E.ContextManager():
      i, lb, ub = list(
          map(E.Expr, [E.Bindable(self.i32Type) for _ in range(3)]))
      expr = E.Mul(i, E.Add(lb, ub))
      str = expr.__str__()
      self.assertIn("($1 * ($2 + $3))", str)

  def testCustomOp(self):
    with E.ContextManager():
      a, b = (E.Expr(E.Bindable(self.i32Type)) for _ in range(2))
      c1 = self.module.op(
          "std.constant",
          self.i32Type, [],
          value=self.module.integerAttr(self.i32Type, 42))
      expr = self.module.op("std.addi", self.i32Type, [c1, b])
      str = expr.__str__()
      self.assertIn("addi(42, $2)", str)

  def testOneLoop(self):
    with E.ContextManager():
      i, lb, ub, step = list(
          map(E.Expr, [E.Bindable(self.indexType) for _ in range(4)]))
      loop = E.For(i, lb, ub, step, [E.Stmt(E.Add(lb, ub))])
      str = loop.__str__()
      self.assertIn("for($1 = $2 to $3 step $4) {", str)
      self.assertIn(" = ($2 + $3)", str)

  def testTwoLoops(self):
    with E.ContextManager():
      i, lb, ub, step = list(
          map(E.Expr, [E.Bindable(self.indexType) for _ in range(4)]))
      loop = E.For(i, lb, ub, step, [E.For(i, lb, ub, step, [E.Stmt(i)])])
      str = loop.__str__()
      self.assertIn("for($1 = $2 to $3 step $4) {", str)
      self.assertIn("for($1 = $2 to $3 step $4) {", str)
      self.assertIn("$5 = $1;", str)

  def testNestedLoops(self):
    with E.ContextManager():
      i, lb, ub = list(
          map(E.Expr, [E.Bindable(self.indexType) for _ in range(3)]))
      step = E.ConstantInteger(self.indexType, 42)
      ivs = list(map(E.Expr, [E.Bindable(self.indexType) for _ in range(4)]))
      lbs = list(map(E.Expr, [E.Bindable(self.indexType) for _ in range(4)]))
      ubs = list(map(E.Expr, [E.Bindable(self.indexType) for _ in range(4)]))
      steps = list(map(E.Expr, [E.Bindable(self.indexType) for _ in range(4)]))
      loop = E.For(ivs, lbs, ubs, steps, [
          E.For(i, lb, ub, step, [E.Stmt(ub * step - lb)]),
      ])
      str = loop.__str__()
      self.assertIn("for($5 = $9 to $13 step $17) {", str)
      self.assertIn("for($6 = $10 to $14 step $18) {", str)
      self.assertIn("for($7 = $11 to $15 step $19) {", str)
      self.assertIn("for($8 = $12 to $16 step $20) {", str)
      self.assertIn("for($1 = $2 to $3 step 42) {", str)
      self.assertIn("= (($3 * 42) + $2 * -1);", str)

  def testMaxMinLoop(self):
    with E.ContextManager():
      i = E.Expr(E.Bindable(self.indexType))
      step = E.Expr(E.Bindable(self.indexType))
      lbs = list(map(E.Expr, [E.Bindable(self.indexType) for _ in range(4)]))
      ubs = list(map(E.Expr, [E.Bindable(self.indexType) for _ in range(3)]))
      loop = E.For(i, E.Max(lbs), E.Min(ubs), step, [])
      s = str(loop)
      self.assertIn("for($1 = max($3, $4, $5, $6) to min($7, $8, $9) step $2)",
                    s)

  def testIndexed(self):
    with E.ContextManager():
      i, j, k = list(
          map(E.Expr, [E.Bindable(self.indexType) for _ in range(3)]))
      memrefType = self.module.make_memref_type(self.f32Type, [42, 42])
      A, B, C = list(map(E.Indexed, [E.Bindable(memrefType) for _ in range(3)]))
      stmt = C.store([i, j], A.load([i, k]) * B.load([k, j]))
      str = stmt.__str__()
      self.assertIn(" = std.store(", str)

  def testMatmul(self):
    with E.ContextManager():
      ivs = list(map(E.Expr, [E.Bindable(self.indexType) for _ in range(3)]))
      lbs = list(map(E.Expr, [E.Bindable(self.indexType) for _ in range(3)]))
      ubs = list(map(E.Expr, [E.Bindable(self.indexType) for _ in range(3)]))
      steps = list(map(E.Expr, [E.Bindable(self.indexType) for _ in range(3)]))
      i, j, k = ivs[0], ivs[1], ivs[2]
      memrefType = self.module.make_memref_type(self.f32Type, [42, 42])
      A, B, C = list(map(E.Indexed, [E.Bindable(memrefType) for _ in range(3)]))
      loop = E.For(
          ivs, lbs, ubs, steps,
          [C.store([i, j],
                   C.load([i, j]) + A.load([i, k]) * B.load([k, j]))])
      str = loop.__str__()
      self.assertIn("for($1 = $4 to $7 step $10) {", str)
      self.assertIn("for($2 = $5 to $8 step $11) {", str)
      self.assertIn("for($3 = $6 to $9 step $12) {", str)
      self.assertIn(" = std.store", str)

  def testArithmetic(self):
    with E.ContextManager():
      i, j, k, l = list(
          map(E.Expr, [E.Bindable(self.f32Type) for _ in range(4)]))
      stmt = i % j + j * k - l / k
      str = stmt.__str__()
      self.assertIn("((($1 % $2) + ($2 * $3)) - ($4 / $3))", str)

  def testBoolean(self):
    with E.ContextManager():
      i, j, k, l = list(
          map(E.Expr, [E.Bindable(self.i32Type) for _ in range(4)]))
      stmt1 = (i < j) & (j >= k)
      stmt2 = ~(stmt1 | (k == l))
      str = stmt2.__str__()
      # Note that "a | b" is currently implemented as ~(~a && ~b) and "~a" is
      # currently implemented as "constant 1 - a", which leads to this
      # expression.
      self.assertIn(
          "(1 - (1 - ((1 - (($1 < $2) && ($2 >= $3))) && (1 - ($3 == $4)))))",
          str)

  def testSelect(self):
    with E.ContextManager():
      i, j, k = list(map(E.Expr, [E.Bindable(self.i32Type) for _ in range(3)]))
      stmt = E.Select(i > j, i, j)
      str = stmt.__str__()
      self.assertIn("select(($1 > $2), $1, $2)", str)

  def testCall(self):
    with E.ContextManager():
      module = E.MLIRModule()
      f32 = module.make_scalar_type("f32")
      func, arg = [E.Expr(E.Bindable(f32)) for _ in range(2)]
      code = func(arg, result=f32)
      self.assertIn("@$1($2)", str(code))

  def testBlock(self):
    with E.ContextManager():
      i, j = list(map(E.Expr, [E.Bindable(self.f32Type) for _ in range(2)]))
      stmt = E.Block([E.Stmt(i + j), E.Stmt(i - j)])
      str = stmt.__str__()
      self.assertIn("^bb", str)
      self.assertIn(" = ($1 + $2)", str)
      self.assertIn(" = ($1 - $2)", str)

  def testBlockArgs(self):
    with E.ContextManager():
      module = E.MLIRModule()
      t = module.make_scalar_type("i", 32)
      i, j = list(map(E.Expr, [E.Bindable(t) for _ in range(2)]))
      stmt = E.Block([i, j], [E.Stmt(i + j)])
      str = stmt.__str__()
      self.assertIn("^bb", str)
      self.assertIn("($1, $2):", str)
      self.assertIn("($1 + $2)", str)

  def testBranch(self):
    with E.ContextManager():
      i, j = list(map(E.Expr, [E.Bindable(self.i32Type) for _ in range(2)]))
      b1 = E.Block([E.Stmt(i + j)])
      b2 = E.Block([E.Branch(b1)])
      str1 = b1.__str__()
      str2 = b2.__str__()
      self.assertIn("^bb1:\n" + "$4 = ($1 + $2)", str1)
      self.assertIn("^bb2:\n" + "$6 = br ^bb1", str2)

  def testBranchArgs(self):
    with E.ContextManager():
      b1arg, b2arg = (E.Expr(E.Bindable(self.i32Type)) for _ in range(2))
      # Declare empty blocks with arguments and bind those arguments.
      b1 = E.Block([b1arg], [])
      b2 = E.Block([b2arg], [])
      one = E.ConstantInteger(self.i32Type, 1)
      # Make blocks branch to each other in a sort of infinite loop.
      # This checks that the EDSC implementation does not fall into such loop.
      b1.set([E.Branch(b2, [b1arg + one])])
      b2.set([E.Branch(b1, [b2arg])])
      str1 = b1.__str__()
      str2 = b2.__str__()
      self.assertIn("^bb1($1):\n" + "$6 = br ^bb2(($1 + 1))", str1)
      self.assertIn("^bb2($2):\n" + "$8 = br ^bb1($2)", str2)

  def testCondBranch(self):
    with E.ContextManager():
      cond = E.Expr(E.Bindable(self.boolType))
      b1 = E.Block([])
      b2 = E.Block([])
      b3 = E.Block([E.CondBranch(cond, b1, b2)])
      str = b3.__str__()
      self.assertIn("cond_br($1, ^bb1, ^bb2)", str)

  def testCondBranchArgs(self):
    with E.ContextManager():
      arg1, arg2, arg3 = (E.Expr(E.Bindable(self.i32Type)) for _ in range(3))
      expr1, expr2, expr3 = (E.Expr(E.Bindable(self.i32Type)) for _ in range(3))
      cond = E.Expr(E.Bindable(self.boolType))
      b1 = E.Block([arg1], [])
      b2 = E.Block([arg2, arg3], [])
      b3 = E.Block([E.CondBranch(cond, b1, [expr1], b2, [expr2, expr3])])
      str = b3.__str__()
      self.assertIn("cond_br($7, ^bb1($4), ^bb2($5, $6))", str)

  def testMLIRScalarTypes(self):
    module = E.MLIRModule()
    t = module.make_scalar_type("bf16")
    self.assertIn("bf16", t.__str__())
    t = module.make_scalar_type("f16")
    self.assertIn("f16", t.__str__())
    t = module.make_scalar_type("f32")
    self.assertIn("f32", t.__str__())
    t = module.make_scalar_type("f64")
    self.assertIn("f64", t.__str__())
    t = module.make_scalar_type("i", 1)
    self.assertIn("i1", t.__str__())
    t = module.make_scalar_type("i", 8)
    self.assertIn("i8", t.__str__())
    t = module.make_scalar_type("i", 32)
    self.assertIn("i32", t.__str__())
    t = module.make_scalar_type("i", 123)
    self.assertIn("i123", t.__str__())
    t = module.make_scalar_type("index")
    self.assertIn("index", t.__str__())

  def testMLIRFunctionCreation(self):
    module = E.MLIRModule()
    t = module.make_scalar_type("f32")
    self.assertIn("f32", t.__str__())
    m = module.make_memref_type(t, [3, 4, -1, 5])
    self.assertIn("memref<3x4x?x5xf32>", m.__str__())
    f = module.make_function("copy", [m, m], [])
    self.assertIn(
        "func @copy(%arg0: memref<3x4x?x5xf32>, %arg1: memref<3x4x?x5xf32>) {",
        f.__str__())

    f = module.make_function("sqrtf", [t], [t])
    self.assertIn("func @sqrtf(%arg0: f32) -> f32", f.__str__())

  def testMLIRConstantEmission(self):
    module = E.MLIRModule()
    f = module.make_function("constants", [], [])
    with E.ContextManager():
      emitter = E.MLIRFunctionEmitter(f)
      emitter.bind_constant_bf16(1.23)
      emitter.bind_constant_f16(1.23)
      emitter.bind_constant_f32(1.23)
      emitter.bind_constant_f64(1.23)
      emitter.bind_constant_int(1, 1)
      emitter.bind_constant_int(123, 8)
      emitter.bind_constant_int(123, 16)
      emitter.bind_constant_int(123, 32)
      emitter.bind_constant_index(123)
      emitter.bind_constant_function(f)
      str = f.__str__()
      self.assertIn("constant 1.230000e+00 : bf16", str)
      self.assertIn("constant 1.230470e+00 : f16", str)
      self.assertIn("constant 1.230000e+00 : f32", str)
      self.assertIn("constant 1.230000e+00 : f64", str)
      self.assertIn("constant 1 : i1", str)
      self.assertIn("constant 123 : i8", str)
      self.assertIn("constant 123 : i16", str)
      self.assertIn("constant 123 : i32", str)
      self.assertIn("constant 123 : index", str)
      self.assertIn("constant @constants : () -> ()", str)

  def testMLIRBuiltinEmission(self):
    module = E.MLIRModule()
    m = module.make_memref_type(self.f32Type, [10])  # f32 tensor
    f = module.make_function("call_builtin", [m, m], [])
    with E.ContextManager():
      emitter = E.MLIRFunctionEmitter(f)
      input, output = list(map(E.Indexed, emitter.bind_function_arguments()))
      fn = module.declare_function("sqrtf", [self.f32Type], [self.f32Type])
      fn = emitter.bind_constant_function(fn)
      zero = emitter.bind_constant_index(0)
      emitter.emit_inplace(E.Block([
        output.store([zero], fn(input.load([zero]), result=self.f32Type))
      ]))
      str = f.__str__()
      self.assertIn("%f = constant @sqrtf : (f32) -> f32", str)
      self.assertIn("call_indirect %f(%0) : (f32) -> f32", str)

  def testFunctionDeclaration(self):
    module = E.MLIRModule()
    boolAttr = self.module.boolAttr(True)
    t = module.make_memref_type(self.f32Type, [10])
    t_llvm_noalias = t({"llvm.noalias": boolAttr})
    t_readonly = t({"readonly": boolAttr})
    f = module.declare_function("foo", [t, t_llvm_noalias, t_readonly], [])
    str = module.__str__()
    self.assertIn(
        "func @foo(memref<10xf32>, memref<10xf32> {llvm.noalias: true}, memref<10xf32> {readonly: true})",
        str)

  def testMLIRBooleanEmission(self):
    m = self.module.make_memref_type(self.boolType, [10])  # i1 tensor
    f = self.module.make_function("mkbooltensor", [m, m], [])
    with E.ContextManager():
      emitter = E.MLIRFunctionEmitter(f)
      input, output = list(map(E.Indexed, emitter.bind_function_arguments()))
      i = E.Expr(E.Bindable(self.indexType))
      j = E.Expr(E.Bindable(self.indexType))
      k = E.Expr(E.Bindable(self.indexType))
      idxs = [i, j, k]
      zero = emitter.bind_constant_index(0)
      one = emitter.bind_constant_index(1)
      ten = emitter.bind_constant_index(10)
      b1 = E.And(i < j, j < k)
      b2 = E.Negate(b1)
      b3 = E.Or(b2, k < j)
      loop = E.Block([
          E.For(idxs, [zero]*3, [ten]*3, [one]*3, [
              output.store([i], E.And(input.load([i]), b3))
          ]),
          E.Return()
      ])
      emitter.emit_inplace(loop)
      # str = f.__str__()
      # print(str)
      self.module.compile()
      self.assertNotEqual(self.module.get_engine_address(), 0)

  def testCustomOpEmission(self):
    f = self.module.make_function("fooer", [self.i32Type, self.i32Type], [])
    with E.ContextManager():
      emitter = E.MLIRFunctionEmitter(f)
      funcArg1, funcArg2 = emitter.bind_function_arguments()
      boolAttr = self.module.boolAttr(True)
      expr = self.module.op(
          "foo", self.i32Type, [funcArg1, funcArg2], attr=boolAttr)
      block = E.Block([E.Stmt(expr), E.Return()])
      emitter.emit_inplace(block)

      code = str(f)
      self.assertIn('%0 = "foo"(%arg0, %arg1) {attr: true} : (i32, i32) -> i32',
                    code)

  # Create 'addi' using the generic Op interface.  We need an operation known
  # to the execution engine so that the engine can compile it.
  def testCustomOpCompilation(self):
    f = self.module.make_function("adder", [self.i32Type], [])
    with E.ContextManager():
      emitter = E.MLIRFunctionEmitter(f)
      funcArg, = emitter.bind_function_arguments()
      c1 = self.module.op(
          "std.constant",
          self.i32Type, [],
          value=self.module.integerAttr(self.i32Type, 42))
      expr = self.module.op("std.addi", self.i32Type, [c1, funcArg])
      block = E.Block([E.Stmt(expr), E.Return()])
      emitter.emit_inplace(block)
      self.module.compile()
      self.assertNotEqual(self.module.get_engine_address(), 0)


  def testMLIREmission(self):
    shape = [3, 4, 5]
    m = self.module.make_memref_type(self.f32Type, shape)
    f = self.module.make_function("copy", [m, m], [])

    with E.ContextManager():
      emitter = E.MLIRFunctionEmitter(f)
      zero = emitter.bind_constant_index(0)
      one = emitter.bind_constant_index(1)
      input, output = list(map(E.Indexed, emitter.bind_function_arguments()))
      M, N, O = emitter.bind_indexed_shape(input)

      ivs = list(
          map(E.Expr, [E.Bindable(self.indexType) for _ in range(len(shape))]))
      lbs = [zero, zero, zero]
      ubs = [M, N, O]
      steps = [one, one, one]

      # TODO(ntv): emitter.assertEqual(M, oM) etc
      loop = E.Block([
          E.For(ivs, lbs, ubs, steps, [output.store(ivs, input.load(ivs))]),
          E.Return()
      ])
      emitter.emit_inplace(loop)

      # print(f) # uncomment to see the emitted IR
      str = f.__str__()
      self.assertIn("""store %0, %arg1[%i0, %i1, %i2] : memref<3x4x5xf32>""",
                    str)

      self.module.compile()
      self.assertNotEqual(self.module.get_engine_address(), 0)


if __name__ == "__main__":
  unittest.main()
