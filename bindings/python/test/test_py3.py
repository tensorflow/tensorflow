"""Python3 test for the MLIR EDSC C API and Python bindings"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import google_mlir.bindings.python.pybind as E


class EdscTest(unittest.TestCase):

  def testSugaredMLIREmission(self):
    shape = [3, 4, 5, 6, 7]
    shape_t = [7, 4, 5, 6, 3]
    module = E.MLIRModule()
    t = module.make_scalar_type("f32")
    m = module.make_memref_type(t, shape)
    m_t = module.make_memref_type(t, shape_t)
    f = module.make_function("copy", [m, m_t], [])

    with E.ContextManager():
      emitter = E.MLIRFunctionEmitter(f)
      input, output = list(map(E.Indexed, emitter.bind_function_arguments()))
      lbs, ubs, steps = emitter.bind_indexed_view(input)
      i, *ivs, j = list(map(E.Expr, [E.Bindable() for _ in range(len(shape))]))

      # n-D type and rank agnostic copy-transpose-first-last (where n >= 2).
      loop = E.Block([
          E.For([i, *ivs, j], lbs, ubs, steps,
                [output.store([i, *ivs, j], input.load([j, *ivs, i]))]),
          E.Return()
      ])
      emitter.emit(loop)

      # print(f) # uncomment to see the emitted IR
      str = f.__str__()
      self.assertIn("load %arg0[%i4, %i1, %i2, %i3, %i0]", str)
      self.assertIn("store %0, %arg1[%i0, %i1, %i2, %i3, %i4]", str)

      module.compile()
      self.assertNotEqual(module.get_engine_address(), 0)


if __name__ == "__main__":
  unittest.main()
