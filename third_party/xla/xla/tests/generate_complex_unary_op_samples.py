"""A script to generate the complex_unary_op_samples.h file.

The generated file contains samples and reference values of complex unary
functions used by the complex_unary_op_test program.

Prerequisites:
  jax version 0.4.26 or newer
  mpmath 1.3
  numpy

Usage:
  Running
    python /path/to/generate_complex_unary_op_samples.py [xla | tensorflow]
  will create
    /path/to/generate_complex_unary_op_samples.h

Constraints:
  Running
    clang-format -i -style=Google /path/to/complex_unary_op_samples.h
  should not change the generated complex_unary_op_samples.h
"""

import os
import platform
import re
import sys
import jax._src.test_util as jtu
import mpmath
import numpy as np


def disable(op, real, imag):
  # Return True to disable samples (real, imag) that are know to be
  # problematic for the given op.
  del op, real, imag
  return False


def main():
  machine = platform.machine()
  is_arm_cpu = machine.startswith('aarch') or machine.startswith('arm')
  if is_arm_cpu and platform.system() == 'Darwin':
    # jtu.complex_plane_sample on Darwin ARM generates samples that
    # are specific to the given platform (tiny is mapped to
    # nextafter(tiny, inf) to avoid unexpected result when DAZ is
    # enabled). Here we handle the Mac specific DAZ difference at C++
    # level (see the __aarch64__-dependent min value mapping below).
    sys.stdout.write("Don't run this script under Darwin ARM\n")
    return
  target = (sys.argv[1] if len(sys.argv) > 1 else 'xla').lower()
  assert target in {'xla', 'tensorflow'}, target
  header_file_define = dict(
      xla='XLA_TESTS_COMPLEX_UNARY_OP_SAMPLES_H_',
      tensorflow='TENSORFLOW_COMPILER_XLA_TESTS_COMPLEX_UNARY_OP_SAMPLES_H_',
  )[target]
  default_size = 7
  default_extra_prec_multiplier = 1

  blocks = []
  for opname in ['Log1p', 'Tan', 'Asin', 'Asinh']:
    mpmath_op = opname.lower()
    mpmath_op = dict(asin='arcsin', asinh='arcsinh').get(mpmath_op, mpmath_op)
    size_re, size_im = dict(Log1p=(7, 7), Tan=(7, 7)).get(
        opname, (default_size, default_size)
    )
    extra_prec_multiplier = dict(
        Log1p=1,
        Tan=1,
        # TODO(pearu): reduce to 1 after a fix to mpmath/mpmath#787 becomes
        # available
        Asin=20,
        Asinh=20,
    ).get(opname, default_extra_prec_multiplier)
    nmp = jtu.numpy_with_mpmath(
        mpmath, extra_prec_multiplier=extra_prec_multiplier
    )

    ifblocks = []
    input_ttype = 'std::complex<T>'
    output_ttype = 'TBD'
    for dtype in [np.complex64, np.complex128]:
      float_dtype = {np.complex64: np.float32, np.complex128: np.float64}[dtype]
      ctype = {np.float32: 'float', np.float64: 'double'}[float_dtype]
      cnan = {np.float32: 'std::nanf("")', np.float64: 'std::nan("")'}[
          float_dtype
      ]
      pi = float_dtype(np.pi)
      h_pi = float_dtype(np.pi / 2)
      q_pi = float_dtype(np.pi / 4)
      tq_pi = float_dtype(3 * np.pi / 4)
      cfloat_suffix = 'f' if float_dtype == np.float32 else ''
      cpi = str(pi) + cfloat_suffix
      cpi_2 = str(h_pi) + cfloat_suffix
      cpi_4 = str(q_pi) + cfloat_suffix
      cpi3_4 = str(tq_pi) + cfloat_suffix
      czero = str(float_dtype(0)) + cfloat_suffix
      finfo = np.finfo(float_dtype)

      # pylint: disable=cell-var-from-loop
      def _tostr(v):
        if v == pi:
          return 'pi'
        if v == -pi:
          return '-pi'
        if v == h_pi:
          return 'pi_2'
        if v == -h_pi:
          return '-pi_2'
        if v == q_pi:
          return 'pi_4'
        if v == -q_pi:
          return '-pi_4'
        if v == tq_pi:
          return 'pi3_4'
        if v == -tq_pi:
          return '-pi3_4'
        if v == finfo.max:
          return 'max'
        if v == -finfo.max:
          return '-max'
        if v == finfo.tiny:
          return 'min'
        if v == -finfo.tiny:
          return '-min'
        if np.isnan(v):
          return 'nan'
        if np.isneginf(v):
          return '-inf'
        if np.isposinf(v):
          return 'inf'
        if v == 0.0:
          return 'zero'
        if float_dtype == np.float32:
          s = f'{v:.7e}f'
        elif float_dtype == np.float64:
          s = f'{v:.16e}'
        else:
          assert 0  # unreachable
        return re.sub(r'0+e', 'e', s)

      used_constants = set()

      def tostr(v):
        r = _tostr(v)
        used_constants.add(r.removeprefix('-'))
        return r

      rows = []
      counter = 0

      sample = jtu.complex_plane_sample(
          dtype, size_re=size_re, size_im=size_im
      ).flatten()
      values = getattr(nmp, mpmath_op)(sample)
      for x, y in zip(sample, values):
        prev_used_constants = used_constants.copy()
        re_x, im_x = tostr(x.real), tostr(x.imag)
        skip = disable(opname, re_x, im_x)
        if skip:
          prefix = '// '
        else:
          # to ease tracking mismatching cases:
          prefix = f'/* {counter} */ '
          counter += 1
        if values.dtype.kind == 'c':
          output_ttype = 'std::complex<T>'
          re_y, im_y = tostr(y.real), tostr(y.imag)
          scale = tostr(np.ldexp(1.0, -np.frexp(abs(y))[1]))
          rows.append(
              f'{prefix}{{ {{ {re_x}, {im_x} }}, {{ {re_y}, {im_y} }},'
              f' {scale} }}'
          )
        else:
          assert values.dtype.kind == 'f'
          output_ttype = 'T'
          # Scale is power of 2 so that multiplication with
          # it has minimal effect to the binary mantissa
          # part of other operand.
          scale = tostr(np.ldexp(1.0, -np.frexp(abs(y))[1]))
          rows.append(
              f'{prefix}{{ {{ {re_x}, {im_x} }}, {tostr(y)}, {scale} }}'
          )
        if skip:
          # restore used_constants
          used_constants.difference_update(
              used_constants.difference(prev_used_constants)
          )

      rows = ',\n        '.join(rows)

      constants = []
      for name, value in dict(
          nan=cnan,
          pi=cpi,
          pi_4=cpi_4,
          pi_2=cpi_2,
          pi3_4=cpi3_4,
          zero=czero,
          inf='std::numeric_limits<T>::infinity()',
          min='std::numeric_limits<T>::min()',
          max='std::numeric_limits<T>::max()',
      ).items():
        if name in used_constants:
          if name == 'min':
            constants.append('#ifdef __aarch64__')
            constants.append(f'const T {name} = std::nextafter({value}, T(1));')
            constants.append('#else')
            constants.append(f'const T {name} = {value};')
            constants.append('#endif')
          else:
            constants.append(f'const T {name} = {value};')
      nl = '\n      '
      constants = nl.join(constants)
      constants = constants.replace(nl + '#', '\n#')

      ifblocks.append(f"""\
if constexpr (std::is_same_v<T, {ctype}>) {{
      {constants}
      const TableType table{{
          // clang-format off
          // Ignore max 80 character line width style requirement for
          // (i) the readability
          // (ii) the consistency with the local conventions
        {rows}
          // clang-format on
      }};
      return table;
    }}""")
    ifblocks.append(
        '{\n      static_assert(dependent_false<T>); /* unreachable */\n    }'
    )
    ifblocks = ' else '.join(ifblocks)
    blocks.append(f"""\
template <typename T, int default_dps_deficiency = 0>
struct {opname} {{
  typedef {input_ttype} InputType;
  typedef {output_ttype} OutputType;
  typedef T FloatType;
  using TableType = std::vector<std::tuple<InputType, OutputType, FloatType>>;
  static constexpr int dps_deficiency = default_dps_deficiency;
  const TableType get() {{
    {ifblocks}
  }}
}};
""")
  blocks = '\n'.join(blocks)

  output_filename = os.path.join(
      os.path.dirname(__file__), 'complex_unary_op_samples.h'
  )
  output = open(output_filename, 'w')

  output.write(f"""\
/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*
  This file is generated using xla/tests/{os.path.basename(__file__)}.
  Do not edit!
 */

#include <cmath>
#include <complex>
#include <limits>
#include <tuple>
#include <vector>

#ifndef {header_file_define}
#define {header_file_define}

namespace complex_unary_op_samples {{
// NOLINTBEGIN(whitespace/line_length)

template <class>
constexpr bool dependent_false = false;

{blocks}
// NOLINTEND(whitespace/line_length)
}}  // namespace complex_unary_op_samples

#endif  // {header_file_define}
""")
  output.close()
  sys.stdout.write(f'Created {output_filename}\n')


if __name__ == '__main__':
  main()
