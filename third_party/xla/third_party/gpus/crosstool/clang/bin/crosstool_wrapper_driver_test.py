#!/usr/bin/env python3
"""Unit tests for crosstool wrapper driver parsing helpers.

The .tpl source files are not directly importable (they contain unexpanded
template variables like %{rocm_root}), so the functions under test are
reproduced verbatim here for isolated testing.
"""

import unittest
from argparse import ArgumentParser


# -- Functions under test (copied from crosstool_wrapper_driver_rocm.tpl) -----

def _parse_hipcc_env(hipcc_env_str):
  env_list = []
  if not hipcc_env_str or not hipcc_env_str.strip():
    return env_list
  for assignment in hipcc_env_str.split(';'):
    assignment = assignment.strip()
    if assignment and '=' in assignment:
      key, _, value = assignment.partition('=')
      value = value.strip().strip('"')
      env_list.append('{0}={1}'.format(key.strip(), value))
  return env_list


def GetHostCompilerOptions(argv):
  parser = ArgumentParser()
  parser.add_argument('-isystem', nargs='*', action='append')
  parser.add_argument('-iquote', nargs='*', action='append')
  parser.add_argument('--sysroot', nargs=1)
  parser.add_argument('-g', nargs='*', action='append')
  parser.add_argument('-no-canonical-prefixes', action='store_true')
  parser.add_argument('--genco', action='store_true')

  args, _ = parser.parse_known_args(argv)

  opts = []
  if args.isystem:
    for p in sum(args.isystem, []):
      opts.extend(['-isystem', p])
  if args.iquote:
    for p in sum(args.iquote, []):
      opts.extend(['-iquote', p])
  if args.g:
    for g in sum(args.g, []):
      opts.append('-g' + g)
  if args.no_canonical_prefixes:
    opts.append('-no-canonical-prefixes')
  if args.sysroot:
    opts.extend(['--sysroot', args.sysroot[0]])
  if args.genco:
    opts.append('--genco')
  return opts


# -- Tests --------------------------------------------------------------------

class ParseHipccEnvTest(unittest.TestCase):

  def test_empty_string(self):
    self.assertEqual(_parse_hipcc_env(''), [])

  def test_whitespace_only(self):
    self.assertEqual(_parse_hipcc_env('   '), [])

  def test_none(self):
    self.assertEqual(_parse_hipcc_env(None), [])

  def test_single_assignment(self):
    self.assertEqual(_parse_hipcc_env('FOO=bar'), ['FOO=bar'])

  def test_quoted_value(self):
    self.assertEqual(_parse_hipcc_env('FOO="bar"'), ['FOO=bar'])

  def test_multiple_assignments(self):
    result = _parse_hipcc_env('FOO="bar"; BAZ="qux"')
    self.assertEqual(result, ['FOO=bar', 'BAZ=qux'])

  def test_whitespace_around_key(self):
    self.assertEqual(_parse_hipcc_env(' KEY = value '), ['KEY=value'])

  def test_no_equals_sign_skipped(self):
    self.assertEqual(_parse_hipcc_env('NOEQUALS'), [])

  def test_value_with_equals(self):
    # partition only splits on the first '=', value may contain '='
    result = _parse_hipcc_env('FLAG=a=b')
    self.assertEqual(result, ['FLAG=a=b'])


class GetHostCompilerOptionsTest(unittest.TestCase):

  def test_empty(self):
    self.assertEqual(GetHostCompilerOptions([]), [])

  def test_isystem(self):
    result = GetHostCompilerOptions(['-isystem', '/usr/include'])
    self.assertEqual(result, ['-isystem', '/usr/include'])

  def test_multiple_isystem(self):
    result = GetHostCompilerOptions(
        ['-isystem', '/usr/include', '-isystem', '/opt/include'])
    self.assertEqual(result, ['-isystem', '/usr/include', '-isystem', '/opt/include'])

  def test_iquote(self):
    result = GetHostCompilerOptions(['-iquote', 'src/'])
    self.assertEqual(result, ['-iquote', 'src/'])

  def test_sysroot(self):
    result = GetHostCompilerOptions(['--sysroot', '/sysroot'])
    self.assertEqual(result, ['--sysroot', '/sysroot'])

  def test_no_canonical_prefixes(self):
    result = GetHostCompilerOptions(['-no-canonical-prefixes'])
    self.assertEqual(result, ['-no-canonical-prefixes'])

  def test_genco(self):
    result = GetHostCompilerOptions(['--genco'])
    self.assertEqual(result, ['--genco'])

  def test_unknown_flags_ignored(self):
    result = GetHostCompilerOptions(['-DFOO', '-O2', '-c', 'main.cc'])
    self.assertEqual(result, [])

  def test_mixed_flags(self):
    result = GetHostCompilerOptions(
        ['-isystem', '/inc', '-DFOO=1', '-no-canonical-prefixes'])
    self.assertEqual(result, ['-isystem', '/inc', '-no-canonical-prefixes'])


if __name__ == '__main__':
  unittest.main()
