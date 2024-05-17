# Copyright 2018 The OpenXLA Authors.
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
"""Proof that transforming (A*C)+(B*C) <=> (A+B)*C is "safe" if C=2^k.

Specifically, for all floating-point values A, B, and C, if

 - C is equal to +/- 2^k for some (possibly negative) integer k, and
 - A, B, C, A*C, B*C, and A+B are not subnormal, zero, or inf,

then there exists a rounding mode rm in [RTZ, RNE] such that

 (A*C) + (B*C) == (A+B) * C  (computed with rounding mode rm).

Informally, this means that the equivalence holds for powers of 2 C, modulo
flushing to zero or inf, and modulo rounding of intermediate results.

Requires z3 python bindings; try `pip install z3-solver`.
"""

import z3

# We do float16 because it lets the solver run much faster.  These results
# should generalize to fp32 and fp64, and you can verify this by changing the
# value of FLOAT_TY (and then waiting a while).
FLOAT_TY = z3.Float16

a = z3.FP("a", FLOAT_TY())
b = z3.FP("b", FLOAT_TY())
c = z3.FP("c", FLOAT_TY())

s = z3.Solver()

# C must be a power of 2, i.e. significand bits must all be 0.
s.add(z3.Extract(FLOAT_TY().sbits() - 1, 0, z3.fpToIEEEBV(c)) == 0)

for rm in [z3.RTZ(), z3.RNE()]:
  z3.set_default_rounding_mode(rm)
  before = a * c + b * c
  after = (a + b) * c

  # Check that before == after, allowing that 0 == -0.
  s.add(
      z3.Not(
          z3.Or(
              before == after,  #
              z3.And(z3.fpIsZero(before), z3.fpIsZero(after)))))

  for x in [
      (a * c),
      (b * c),
      (a + b),
  ]:
    s.add(z3.Not(z3.fpIsSubnormal(x)))
    s.add(z3.Not(z3.fpIsZero(x)))
    s.add(z3.Not(z3.fpIsInf(x)))

if s.check() == z3.sat:
  m = s.model()
  print("Counterexample found!")
  print(m)
  print("a*c:       ", z3.simplify(m[a] * m[c]))
  print("b*c:       ", z3.simplify(m[b] * m[c]))
  print("a+b:       ", z3.simplify(m[a] + m[b]))
  print("a*c + b*c: ", z3.simplify(m[a] * m[c] + m[b] * m[c]))
  print("(a+b) * c: ", z3.simplify((m[a] + m[b]) * m[c]))
else:
  print("Proved!")
