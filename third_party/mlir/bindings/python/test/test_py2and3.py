# Copyright 2019 The MLIR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# RUN: %p/test_edsc %s | FileCheck %s
"""Python2 and 3 test for the MLIR EDSC Python bindings"""

import google_mlir.bindings.python.pybind as E
import inspect

# Prints `str` prefixed by the current test function name so we can use it in
# Filecheck label directives.
# This is achieved by inspecting the stack and getting the parent name.
def printWithCurrentFunctionName(str):
  print(inspect.stack()[1][3])
  print(str)

class EdscTest:

  def setUp(self):
    self.module = E.MLIRModule()
    self.boolType = self.module.make_scalar_type("i", 1)
    self.i32Type = self.module.make_scalar_type("i", 32)
    self.f32Type = self.module.make_scalar_type("f32")
    self.indexType = self.module.make_index_type()

  def testBlockArguments(self):
    self.setUp()
    with self.module.function_context("foo", [], []) as fun:
      E.constant_index(42)
      with E.BlockContext([self.f32Type, self.f32Type]) as b:
        b.arg(0) + b.arg(1)
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testBlockArguments
    #       CHECK: %{{.*}} = constant 42 : index
    #       CHECK: ^bb{{.*}}(%{{.*}}: f32, %{{.*}}: f32):
    #       CHECK:   %{{.*}} = addf %{{.*}}, %{{.*}} : f32

  def testBlockContext(self):
    self.setUp()
    with self.module.function_context("foo", [], []) as fun:
      cst = E.constant_index(42)
      with E.BlockContext():
        cst + cst
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testBlockContext
    #       CHECK: %{{.*}} = constant 42 : index
    #       CHECK: ^bb
    #       CHECK: %{{.*}} = "affine.apply"() {map = () -> (84)} : () -> index

  def testBlockContextAppend(self):
    self.setUp()
    with self.module.function_context("foo", [], []) as fun:
      E.constant_index(41)
      with E.BlockContext() as b:
        blk = b  # save block handle for later
        E.constant_index(0)
      E.constant_index(42)
      with E.BlockContext(E.appendTo(blk)):
        E.constant_index(1)
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testBlockContextAppend
    #       CHECK: %{{.*}} = constant 41 : index
    #       CHECK: %{{.*}} = constant 42 : index
    #       CHECK: ^bb
    #       CHECK: %{{.*}} = constant 0 : index
    #       CHECK: %{{.*}} = constant 1 : index

  def testBlockContextStandalone(self):
    self.setUp()
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
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testBlockContextStandalone
    #       CHECK: %{{.*}} = constant 41 : index
    #       CHECK: %{{.*}} = constant 42 : index
    #       CHECK: ^bb
    #       CHECK: %{{.*}} = constant 0 : index
    #       CHECK: %{{.*}} = constant 1 : index
    #       CHECK: ^bb
    #       CHECK: %{{.*}} = constant 56 : index
    #       CHECK: %{{.*}} = constant 57 : index

  def testBooleanOps(self):
    self.setUp()
    with self.module.function_context(
        "booleans", [self.boolType for _ in range(4)], []) as fun:
      i, j, k, l = (fun.arg(x) for x in range(4))
      stmt1 = (i < j) & (j >= k)
      stmt2 = ~(stmt1 | (k == l))
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testBooleanOps
    #       CHECK: %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : i1
    #       CHECK: %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : i1
    #       CHECK: %{{.*}} = muli %{{.*}}, %{{.*}} : i1
    #       CHECK: %{{.*}} = cmpi "eq", %{{.*}}, %{{.*}} : i1
    #       CHECK: %{{.*}} = constant 1 : i1
    #       CHECK: %{{.*}} = subi %{{.*}}, %{{.*}} : i1
    #       CHECK: %{{.*}} = constant 1 : i1
    #       CHECK: %{{.*}} = subi %{{.*}}, %{{.*}} : i1
    #       CHECK: %{{.*}} = muli %{{.*}}, %{{.*}} : i1
    #       CHECK: %{{.*}} = constant 1 : i1
    #       CHECK: %{{.*}} = subi %{{.*}}, %{{.*}} : i1
    #       CHECK: %{{.*}} = constant 1 : i1
    #       CHECK: %{{.*}} = subi %{{.*}}, %{{.*}} : i1

  def testBr(self):
    self.setUp()
    with self.module.function_context("foo", [], []) as fun:
      with E.BlockContext() as b:
        blk = b
        E.ret()
      E.br(blk)
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testBr
    #       CHECK:   br ^bb
    #       CHECK: ^bb
    #       CHECK:   return

  def testBrArgs(self):
    self.setUp()
    with self.module.function_context("foo", [], []) as fun:
      # Create an infinite loop.
      with E.BlockContext([self.indexType, self.indexType]) as b:
        E.br(b, [b.arg(1), b.arg(0)])
      E.br(b, [E.constant_index(0), E.constant_index(1)])
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testBrArgs
    #       CHECK:   %{{.*}} = constant 0 : index
    #       CHECK:   %{{.*}} = constant 1 : index
    #       CHECK:   br ^bb{{.*}}(%{{.*}}, %{{.*}} : index, index)
    #       CHECK: ^bb{{.*}}(%{{.*}}: index, %{{.*}}: index):
    #       CHECK:   br ^bb{{.*}}(%{{.*}}, %{{.*}} : index, index)

  def testBrDeclaration(self):
    self.setUp()
    with self.module.function_context("foo", [], []) as fun:
      blk = E.BlockContext()
      E.br(blk.handle())
      with blk:
        E.ret()
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testBrDeclaration
    #       CHECK:   br ^bb
    #       CHECK: ^bb
    #       CHECK:   return

  def testCallOp(self):
    self.setUp()
    callee = self.module.declare_function("sqrtf", [self.f32Type],
                                          [self.f32Type])
    with self.module.function_context("call", [self.f32Type], []) as fun:
      funCst = E.constant_function(callee)
      funCst([fun.arg(0)]) + E.constant_float(42., self.f32Type)
      printWithCurrentFunctionName(str(self.module))
    # CHECK-LABEL: testCallOp
    #       CHECK: func @sqrtf(f32) -> f32
    #       CHECK:   %{{.*}} = constant @sqrtf : (f32) -> f32
    #       CHECK:   %{{.*}} = call_indirect %{{.*}}(%{{.*}}) : (f32) -> f32

  def testCondBr(self):
    self.setUp()
    with self.module.function_context("foo", [self.boolType], []) as fun:
      with E.BlockContext() as blk1:
        E.ret([])
      with E.BlockContext([self.indexType]) as blk2:
        E.ret([])
      cst = E.constant_index(0)
      E.cond_br(fun.arg(0), blk1, [], blk2, [cst])
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testCondBr
    #       CHECK:   cond_br %{{.*}}, ^bb{{.*}}, ^bb{{.*}}(%{{.*}} : index)

  def testConstants(self):
    self.setUp()
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
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testConstants
    #       CHECK:  constant 1.230000e+00 : bf16
    #       CHECK:  constant 1.230470e+00 : f16
    #       CHECK:  constant 1.230000e+00 : f32
    #       CHECK:  constant 1.230000e+00 : f64
    #       CHECK:  constant 1 : i1
    #       CHECK:  constant 123 : i8
    #       CHECK:  constant 123 : i16
    #       CHECK:  constant 123 : i32
    #       CHECK:  constant 123 : index
    #       CHECK:  constant @constants : () -> ()

  def testCustom(self):
    self.setUp()
    with self.module.function_context("custom", [self.indexType, self.f32Type],
                                      []) as fun:
      E.op("foo", [fun.arg(0)], [self.f32Type]) + fun.arg(1)
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testCustom
    #       CHECK: %{{.*}} = "foo"(%{{.*}}) : (index) -> f32
    #       CHECK:  %{{.*}} = addf %{{.*}}, %{{.*}} : f32

  # Create 'addi' using the generic Op interface.  We need an operation known
  # to the execution engine so that the engine can compile it.
  def testCustomOpCompilation(self):
    self.setUp()
    with self.module.function_context("adder", [self.i32Type], []) as f:
      c1 = E.op(
          "std.constant", [], [self.i32Type],
          value=self.module.integerAttr(self.i32Type, 42))
      E.op("std.addi", [c1, f.arg(0)], [self.i32Type])
      E.ret([])
    self.module.compile()
    printWithCurrentFunctionName(str(self.module.get_engine_address() == 0))
    # CHECK-LABEL: testCustomOpCompilation
    #       CHECK: False

  def testDivisions(self):
    self.setUp()
    with self.module.function_context(
        "division", [self.indexType, self.i32Type, self.i32Type], []) as fun:
      # indices only support floor division
      fun.arg(0) // E.constant_index(42)
      # regular values only support regular division
      fun.arg(1) / fun.arg(2)
      printWithCurrentFunctionName(str(self.module))
    # CHECK-LABEL: testDivisions
    #       CHECK:  floordiv 42
    #       CHECK:  divis %{{.*}}, %{{.*}} : i32

  def testFunctionArgs(self):
    self.setUp()
    with self.module.function_context("foo", [self.f32Type, self.f32Type],
                                      [self.indexType]) as fun:
      pass
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testFunctionArgs
    #       CHECK: func @foo(%{{.*}}: f32, %{{.*}}: f32) -> index

  def testFunctionContext(self):
    self.setUp()
    with self.module.function_context("foo", [], []):
      pass
      printWithCurrentFunctionName(self.module.get_function("foo"))
    # CHECK-LABEL: testFunctionContext
    #       CHECK: func @foo() {

  def testFunctionDeclaration(self):
    self.setUp()
    boolAttr = self.module.boolAttr(True)
    t = self.module.make_memref_type(self.f32Type, [10])
    t_llvm_noalias = t({"llvm.noalias": boolAttr})
    t_readonly = t({"readonly": boolAttr})
    f = self.module.declare_function("foo", [t, t_llvm_noalias, t_readonly], [])
    printWithCurrentFunctionName(str(self.module))
    # CHECK-LABEL: testFunctionDeclaration
    #       CHECK: func @foo(memref<10xf32>, memref<10xf32> {llvm.noalias = true}, memref<10xf32> {readonly = true})

  def testFunctionMultiple(self):
    self.setUp()
    with self.module.function_context("foo", [], []):
      pass
    with self.module.function_context("foo", [], []):
      E.constant_index(0)
    printWithCurrentFunctionName(str(self.module))
    # CHECK-LABEL: testFunctionMultiple
    #       CHECK: func @foo()
    #       CHECK: func @foo_0()
    #       CHECK: %{{.*}} = constant 0 : index

  def testIndexCast(self):
    self.setUp()
    with self.module.function_context("testIndexCast", [], []):
      index = E.constant_index(0)
      E.index_cast(index, self.module.make_scalar_type("i", 32))
    printWithCurrentFunctionName(str(self.module))
    # CHECK-LABEL: testIndexCast
    #       CHECK: index_cast %{{.*}} : index to i32

  def testIndexedValue(self):
    self.setUp()
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
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testIndexedValue
    #       CHECK: "affine.for"()
    #       CHECK: "affine.for"()
    #       CHECK: "affine.load"
    #  CHECK-SAME: memref<10x42xf32>
    #       CHECK:  %{{.*}} = addf %{{.*}}, %{{.*}} : f32
    #       CHECK:  "affine.store"
    #  CHECK-SAME:  memref<10x42xf32>
    #       CHECK: {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (42)}
    #       CHECK: {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (10)}

  def testLoopContext(self):
    self.setUp()
    with self.module.function_context("foo", [], []) as fun:
      lhs = E.constant_index(0)
      rhs = E.constant_index(42)
      with E.LoopContext(lhs, rhs, 1) as i:
        lhs + rhs + i
        with E.LoopContext(rhs, rhs + rhs, 2) as j:
          x = i + j
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testLoopContext
    #       CHECK: "affine.for"() (
    #       CHECK:   ^bb{{.*}}(%{{.*}}: index):
    #       CHECK: "affine.for"(%{{.*}}, %{{.*}}) (
    #       CHECK: ^bb{{.*}}(%{{.*}}: index):
    #       CHECK: "affine.apply"(%{{.*}}, %{{.*}}) {map = (d0, d1) -> (d0 + d1)} : (index, index) -> index
    #       CHECK: {lower_bound = (d0) -> (d0), step = 2 : index, upper_bound = (d0) -> (d0)} : (index, index) -> ()
    #       CHECK: {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (42)}

  def testLoopNestContext(self):
    self.setUp()
    with self.module.function_context("foo", [], []) as fun:
      lbs = [E.constant_index(i) for i in range(4)]
      ubs = [E.constant_index(10 * i + 5) for i in range(4)]
      with E.LoopNestContext(lbs, ubs, [1, 3, 5, 7]) as (i, j, k, l):
        i + j + k + l
    printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testLoopNestContext
    #       CHECK: "affine.for"() (
    #       CHECK: ^bb{{.*}}(%{{.*}}: index):
    #       CHECK: "affine.for"() (
    #       CHECK: ^bb{{.*}}(%{{.*}}: index):
    #       CHECK: "affine.for"() (
    #       CHECK: ^bb{{.*}}(%{{.*}}: index):
    #       CHECK: "affine.for"() (
    #       CHECK: ^bb{{.*}}(%{{.*}}: index):
    #       CHECK: %{{.*}} = "affine.apply"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {map = (d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)} : (index, index, index, index) -> index

  def testMLIRBooleanCompilation(self):
    self.setUp()
    m = self.module.make_memref_type(self.boolType, [10])  # i1 tensor
    with self.module.function_context("mkbooltensor", [m, m], []) as f:
      input = E.IndexedValue(f.arg(0))
      output = E.IndexedValue(f.arg(1))
      zero = E.constant_index(0)
      ten = E.constant_index(10)
      with E.LoopNestContext([zero] * 3, [ten] * 3, [1] * 3) as (i, j, k):
        b1 = (i < j) & (j < k)
        b2 = ~b1
        b3 = b2 | (k < j)
        output.store([i], input.load([i]) & b3)
      E.ret([])
    self.module.compile()
    printWithCurrentFunctionName(str(self.module.get_engine_address() == 0))
    # CHECK-LABEL: testMLIRBooleanCompilation
    #       CHECK: False

  def testMLIRFunctionCreation(self):
    self.setUp()
    module = E.MLIRModule()
    t = module.make_scalar_type("f32")
    m = module.make_memref_type(t, [3, 4, -1, 5])
    printWithCurrentFunctionName(str(t))
    print(str(m))
    print(str(module.make_function("copy", [m, m], [])))
    print(str(module.make_function("sqrtf", [t], [t])))
    # CHECK-LABEL: testMLIRFunctionCreation
    #       CHECK:  f32
    #       CHECK:  memref<3x4x?x5xf32>
    #       CHECK: func @copy(%{{.*}}: memref<3x4x?x5xf32>, %{{.*}}: memref<3x4x?x5xf32>) {
    #       CHECK:  func @sqrtf(%{{.*}}: f32) -> f32

  def testMLIRScalarTypes(self):
    self.setUp()
    module = E.MLIRModule()
    printWithCurrentFunctionName(str(module.make_scalar_type("bf16")))
    print(str(module.make_scalar_type("f16")))
    print(str(module.make_scalar_type("f32")))
    print(str(module.make_scalar_type("f64")))
    print(str(module.make_scalar_type("i", 1)))
    print(str(module.make_scalar_type("i", 8)))
    print(str(module.make_scalar_type("i", 32)))
    print(str(module.make_scalar_type("i", 123)))
    print(str(module.make_scalar_type("index")))
    # CHECK-LABEL: testMLIRScalarTypes
    #       CHECK:  bf16
    #       CHECK:  f16
    #       CHECK:  f32
    #       CHECK:  f64
    #       CHECK:  i1
    #       CHECK:  i8
    #       CHECK:  i32
    #       CHECK:  i123
    #       CHECK:  index

  def testMatrixMultiply(self):
    self.setUp()
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
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testMatrixMultiply
    #       CHECK: "affine.for"()
    #       CHECK: "affine.for"()
    #       CHECK: "affine.for"()
    #   CHECK-DAG:  %{{.*}} = "affine.load"
    #   CHECK-DAG:  %{{.*}} = "affine.load"
    #       CHECK:  %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
    #       CHECK:  "affine.store"
    #  CHECK-SAME:  memref<32x32xf32>
    #       CHECK: {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (32)} : () -> ()
    #       CHECK: {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (32)} : () -> ()
    #       CHECK: {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (32)} : () -> ()

  def testRet(self):
    self.setUp()
    with self.module.function_context("foo", [],
                                      [self.indexType, self.indexType]) as fun:
      c42 = E.constant_index(42)
      c0 = E.constant_index(0)
      E.ret([c42, c0])
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testRet
    #       CHECK:    %{{.*}} = constant 42 : index
    #       CHECK:    %{{.*}} = constant 0 : index
    #       CHECK:    return %{{.*}}, %{{.*}} : index, index

  def testSelectOp(self):
    self.setUp()
    with self.module.function_context("foo", [self.boolType],
                                      [self.i32Type]) as fun:
      a = E.constant_int(42, 32)
      b = E.constant_int(0, 32)
      E.ret([E.select(fun.arg(0), a, b)])
      printWithCurrentFunctionName(str(fun))
    # CHECK-LABEL: testSelectOp
    #       CHECK:  %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : i32


# Until python 3.6 this cannot be used because the order in the dict is not the
# order of method declaration.
def runTests():
  def isTest(attr):
    return inspect.ismethod(attr) and "EdscTest.setUp " not in str(attr)

  edscTest = EdscTest()
  tests = sorted(filter(isTest,
                        (getattr(edscTest, attr) for attr in dir(edscTest))),
                 key = lambda x : str(x))
  for test in tests:
    test()

if __name__ == '__main__':
  runTests()
