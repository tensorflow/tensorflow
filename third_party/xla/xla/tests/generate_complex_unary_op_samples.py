"""A script to generate the complex_unary_op_samples.h file.

The generated file contains samples and reference values of complex unary
functions used by the complex_unary_op_test program.

Prerequisites:
  jax version 0.4.26 or newer
  mpmath 1.3
  numpy

Usage:
  Running
    python /path/to/generate_complex_unary_op_samples.py
  will create
    /path/to/generate_complex_unary_op_samples.h
"""

import os
import re
import sys
import jax._src.test_util as jtu
import mpmath
import numpy as np


def disable(op, real, imag):
  del op, real, imag
  # Return True to disable samples (real, imag) that are know to be
  # problematic for the op.
  return False


def main():
  default_size = 7
  nmp = jtu.numpy_with_mpmath(mpmath, extra_prec_multiplier=1)
  blocks = []
  for opname in ['Log1p']:
    mpmath_op = opname.lower()
    size_re, size_im = dict(Log1p=(7, 7)).get(
        opname, (default_size, default_size)
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

      sample = jtu.complex_plane_sample(dtype, size_re=size_re, size_im=size_im)
      values = getattr(nmp, mpmath_op)(sample)
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
          s = f'{v:.6e}f'
        elif float_dtype == np.float64:
          s = f'{v:.15e}'
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
      for x, y in zip(sample.flatten(), values.flatten()):
        re_x, im_x = tostr(x.real), tostr(x.imag)
        if disable(opname, re_x, im_x):
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
      rows = ',\n          '.join(rows)

      constants = []
      for name, value in dict(
          nan=cnan,
          pi=cpi,
          pi_4=cpi_4,
          pi_2=cpi_2,
          pi3_4=cpi3_4,
          zero=czero,
      ).items():
        if name in used_constants:
          constants.append(f'const T {name} = {value};')
      constants = '\n        '.join(constants)

      ifblocks.append(f"""\
if constexpr (std::is_same_v<T, {ctype}>) {{
        {constants}
        const TableType table{{
          {rows}
        }};
        return table;
      }}""")
    ifblocks.append('{ static_assert(false); /* unreachable */ }')
    ifblocks = ' else '.join(ifblocks)
    blocks.append(f"""
  template <typename T, int default_dps_deficiency=0>
  struct {opname} {{
    typedef {input_ttype} InputType;
    typedef {output_ttype} OutputType;
    typedef T FloatType;
    using TableType = std::vector<std::tuple<InputType, OutputType, FloatType>>;
    static constexpr int dps_deficiency = default_dps_deficiency;
    const TableType get() {{
      const T inf = std::numeric_limits<T>::infinity();
      const T min = std::numeric_limits<T>::min();
      const T max = std::numeric_limits<T>::max();
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
  This file is generated using xla/tests/{os.path.basename(__file__)}. Do not edit!
 */

#include <cmath>
#include <complex>
#include <limits>
#include <tuple>
#include <vector>

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_COMPLEX_UNARY_OP_SAMPLES_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_COMPLEX_UNARY_OP_SAMPLES_H_

namespace complex_unary_op_samples {{
{blocks}
}}

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_COMPLEX_UNARY_OP_SAMPLES_H_
""")
  output.close()
  sys.stdout.write(f'Created {output_filename}\n')


if __name__ == '__main__':
  main()
