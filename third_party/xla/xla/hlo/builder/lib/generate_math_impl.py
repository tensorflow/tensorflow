"""A script to generate math_impl.h.

Prerequisites:
  python 3.11 or newer
  functional_algorithms 0.3.1 or newer

Usage:
  Running
    python /path/to/generate_math_impl.py [xla | tensorflow]
  will create
    /path/to/math_impl.cc
"""

import os
import sys
import warnings

try:
  import functional_algorithms as fa  # pylint: disable=g-import-not-at-top
except ImportError as msg:
  warnings.warn(f"Skipping: {msg}")
  fa = None


def main():
  if fa is None:
    return
  target = (sys.argv[1] if len(sys.argv) > 1 else "xla").lower()
  assert target in {"xla", "tensorflow"}, target
  header_file_define = dict(
      xla="XLA_CLIENT_MATH_IMPL_H_",
      tensorflow="TENSORFLOW_COMPILER_XLA_CLIENT_MATH_IMPL_H_",
  )[target]

  fa_version = tuple(map(int, fa.__version__.split(".", 4)[:3]))
  if fa_version < (0, 3, 1):
    warnings.warn(
        "functional_algorithm version 0.3.1 or newer is required,"
        f" got {fa.__version__}"
    )
    return

  output_file = os.path.join(os.path.dirname(__file__), "math_impl.h")

  sources = []
  target = fa.targets.xla_client
  for xlaname, fname, args in [
      ("AsinComplex", "complex_asin", ("z:complex",)),
      ("AsinReal", "real_asin", ("x:float",)),
  ]:
    func = getattr(fa.algorithms, fname, None)
    if func is None:
      warnings.warn(
          f"{fa.algorithms.__name__} does not define {fname}. Skipping."
      )
      continue
    ctx = fa.Context(
        paths=[fa.algorithms],
        enable_alt=True,
        default_constant_type="FloatType",
    )
    graph = ctx.trace(func, *args).implement_missing(target).simplify()
    graph.props.update(name=xlaname)
    src = graph.tostring(target)
    if func.__doc__:
      sources.append(target.make_comment(func.__doc__))
    sources[-1] += src
  source = "\n\n".join(sources) + "\n"

  if os.path.isfile(output_file):
    f = open(output_file, "r", encoding="UTF-8")
    content = f.read()
    f.close()
    if content.endswith(source) and 0:
      warnings.warn(f"{output_file} is up-to-date.")
      return

  f = open(output_file, "w", encoding="UTF-8")
  f.write("""/* Copyright 2024 The OpenXLA Authors.

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

""")
  f.write(target.make_comment(f"""\
This file is generated using functional_algorithms tool ({fa.__version__}), see
  https://github.com/pearu/functional_algorithms
for more information.""") + "\n")
  f.write(f"""\
#ifndef {header_file_define}
#define {header_file_define}

#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"

namespace xla {{
namespace math_impl {{
// NOLINTBEGIN(whitespace/line_length)
// clang-format off

""")
  f.write(source)
  f.write(f"""
// clang-format on
// NOLINTEND(whitespace/line_length)
}}  // namespace math_impl
}}  // namespace xla

#endif  // {header_file_define}
""")
  f.close()
  warnings.warn(f"Created {output_file}")


if __name__ == "__main__":
  main()
