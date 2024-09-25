"""Given a .so file, lists symbols that should be included in a stub.

Example usage:
$ bazel run -c opt @local_tsl//third_party/implib_so:get_symbols
/usr/local/cuda/lib64/libcudart.so > third_party/tsl/tsl/cuda/cudart.symbols
"""

import argparse
import importlib

# We can't import implib-gen directly because it has a dash in its name.
implib = importlib.import_module('implib-gen')


def _is_exported_function(s):
  return (
      s['Bind'] != 'LOCAL'
      and s['Type'] == 'FUNC'
      and s['Ndx'] != 'UND'
      and s['Name'] not in ['', '_init', '_fini']
      and s['Default']
  )


def main():
  parser = argparse.ArgumentParser(
      description='Extracts a list of symbols from a shared library'
  )
  parser.add_argument('library', help='Path to the .so file.')
  args = parser.parse_args()
  syms = implib.collect_syms(args.library)
  funs = [s['Name'] for s in syms if _is_exported_function(s)]
  for f in sorted(funs):
    print(f)


if __name__ == '__main__':
  main()
