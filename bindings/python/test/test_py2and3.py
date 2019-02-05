"""Python2 and 3 test for the MLIR EDSC C API and Python bindings"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import google_mlir.bindings.python.pybind as E

class EdscTest(unittest.TestCase):

  def testBindables(self):
    with E.ContextManager():
      i = E.Expr(E.Bindable())
      self.assertIn("$1", i.__str__())

  def testOneExpr(self):
    with E.ContextManager():
      i, lb, ub = list(map(E.Expr, [E.Bindable() for _ in range(3)]))
      expr = E.Mul(i, E.Add(lb, ub))
      str = expr.__str__()
      self.assertIn("($1 * ($2 + $3))", str)

  def testOneLoop(self):
    with E.ContextManager():
      i, lb, ub, step = list(map(E.Expr, [E.Bindable() for _ in range(4)]))
      loop = E.For(i, lb, ub, step, [E.Stmt(E.Add(lb, ub))])
      str = loop.__str__()
      self.assertIn("for($1 = $2 to $3 step $4) {", str)
      self.assertIn(" = ($2 + $3)", str)

  def testTwoLoops(self):
    with E.ContextManager():
      i, lb, ub, step = list(map(E.Expr, [E.Bindable() for _ in range(4)]))
      loop = E.For(i, lb, ub, step, [E.For(i, lb, ub, step, [E.Stmt(i)])])
      str = loop.__str__()
      self.assertIn("for($1 = $2 to $3 step $4) {", str)
      self.assertIn("for($1 = $2 to $3 step $4) {", str)
      self.assertIn("$5 = $1;", str)

  def testNestedLoops(self):
    with E.ContextManager():
      i, lb, ub, step = list(map(E.Expr, [E.Bindable() for _ in range(4)]))
      ivs = list(map(E.Expr, [E.Bindable() for _ in range(4)]))
      lbs = list(map(E.Expr, [E.Bindable() for _ in range(4)]))
      ubs = list(map(E.Expr, [E.Bindable() for _ in range(4)]))
      steps = list(map(E.Expr, [E.Bindable() for _ in range(4)]))
      loop = E.For(ivs, lbs, ubs, steps, [
          E.For(i, lb, ub, step, [E.Stmt(ub * step - lb)]),
      ])
      str = loop.__str__()
      self.assertIn("for($5 = $9 to $13 step $17) {", str)
      self.assertIn("for($6 = $10 to $14 step $18) {", str)
      self.assertIn("for($7 = $11 to $15 step $19) {", str)
      self.assertIn("for($8 = $12 to $16 step $20) {", str)
      self.assertIn("for($1 = $2 to $3 step $4) {", str)
      self.assertIn("= (($3 * $4) - $2);", str)

  def testIndexed(self):
    with E.ContextManager():
      i, j, k = list(map(E.Expr, [E.Bindable() for _ in range(3)]))
      A, B, C = list(map(E.Indexed, [E.Bindable() for _ in range(3)]))
      stmt = C.store([i, j], A.load([i, k]) * B.load([k, j]))
      str = stmt.__str__()
      self.assertIn(" = store(", str)

  def testMatmul(self):
    with E.ContextManager():
      ivs = list(map(E.Expr, [E.Bindable() for _ in range(3)]))
      lbs = list(map(E.Expr, [E.Bindable() for _ in range(3)]))
      ubs = list(map(E.Expr, [E.Bindable() for _ in range(3)]))
      steps = list(map(E.Expr, [E.Bindable() for _ in range(3)]))
      i, j, k = ivs[0], ivs[1], ivs[2]
      A, B, C = list(map(E.Indexed, [E.Bindable() for _ in range(3)]))
      loop = E.For(
          ivs, lbs, ubs, steps,
          [C.store([i, j],
                   C.load([i, j]) + A.load([i, k]) * B.load([k, j]))])
      str = loop.__str__()
      self.assertIn("for($1 = $4 to $7 step $10) {", str)
      self.assertIn("for($2 = $5 to $8 step $11) {", str)
      self.assertIn("for($3 = $6 to $9 step $12) {", str)
      self.assertIn(" = store", str)

  def testArithmetic(self):
    with E.ContextManager():
      i, j, k, l = list(map(E.Expr, [E.Bindable() for _ in range(4)]))
      stmt = i + j * k - l
      str = stmt.__str__()
      self.assertIn("(($1 + ($2 * $3)) - $4)", str)

  def testBoolean(self):
    with E.ContextManager():
      i, j, k, l = list(map(E.Expr, [E.Bindable() for _ in range(4)]))
      stmt1 = (i < j) & (j >= k)
      stmt2 = ~(stmt1 | (k == l))
      str = stmt2.__str__()
      self.assertIn("~((($1 < $2) && ($2 >= $3)) || ($3 == $4))", str)

  def testSelect(self):
    with E.ContextManager():
      i, j, k = list(map(E.Expr, [E.Bindable() for _ in range(3)]))
      stmt = E.Select(i > j, i, j)
      str = stmt.__str__()
      self.assertIn("select(($1 > $2), $1, $2)", str)

  def testBlock(self):
    with E.ContextManager():
      i, j = list(map(E.Expr, [E.Bindable() for _ in range(2)]))
      stmt = E.Block([E.Stmt(i + j), E.Stmt(i - j)])
      str = stmt.__str__()
      self.assertIn("stmt_list {", str)
      self.assertIn(" = ($1 + $2)", str)
      self.assertIn(" = ($1 - $2)", str)
      self.assertIn("}", str)

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

  def testMLIRBooleanEmission(self):
    module = E.MLIRModule()
    t = module.make_scalar_type("i", 1)
    m = module.make_memref_type(t, [10]) # i1 tensor
    f = module.make_function("mkbooltensor", [m, m], [])
    with E.ContextManager():
      emitter = E.MLIRFunctionEmitter(f)
      input, output = list(map(E.Indexed, emitter.bind_function_arguments()))
      i = E.Expr(E.Bindable())
      j = E.Expr(E.Bindable())
      k = E.Expr(E.Bindable())
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
      emitter.emit(loop)
      # str = f.__str__()
      # print(str)
      module.compile()
      self.assertNotEqual(module.get_engine_address(), 0)

  # TODO(ntv): support symbolic For bounds with EDSCs
  def testMLIREmission(self):
    shape = [3, 4, 5]
    module = E.MLIRModule()
    index = module.make_scalar_type("index")
    t = module.make_scalar_type("f32")
    m = module.make_memref_type(t, shape)
    f = module.make_function("copy", [m, m], [])

    with E.ContextManager():
      emitter = E.MLIRFunctionEmitter(f)
      zero = emitter.bind_constant_index(0)
      one = emitter.bind_constant_index(1)
      input, output = list(map(E.Indexed, emitter.bind_function_arguments()))
      M, N, O = emitter.bind_indexed_shape(input)

      ivs = list(map(E.Expr, [E.Bindable() for _ in range(len(shape))]))
      lbs = [zero, zero, zero]
      ubs = [M, N, O]
      steps = [one, one, one]

      # TODO(ntv): emitter.assertEqual(M, oM) etc
      loop = E.Block([
          E.For(ivs, lbs, ubs, steps, [output.store(ivs, input.load(ivs))]),
          E.Return()
      ])
      emitter.emit(loop)

      # print(f) # uncomment to see the emitted IR
      str = f.__str__()
      self.assertIn("""store %0, %arg1[%i0, %i1, %i2] : memref<3x4x5xf32>""",
                    str)

      module.compile()
      self.assertNotEqual(module.get_engine_address(), 0)

if __name__ == "__main__":
  unittest.main()
