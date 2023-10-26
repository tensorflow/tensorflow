# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
import collections
import os

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow.python.tools.api.generator2.generator import generator
from tensorflow.python.tools.api.generator2.shared import exported_api

tensor_es = exported_api.ExportedSymbol(
    file_name='tf/python/framework/tensor.py',
    line_no=1,
    symbol_name='Tensor',
    v1_apis=('tf.Tensor',),
    v2_apis=(
        'tf.Tensor',
        'tf.experimental.numpy.ndarray',
    ),
)

test_data = {
    'tf/python/framework/tensor_mapping.json': exported_api.ExportedApi(
        docs=[],
        symbols=[tensor_es],
    ),
    'tf/python/framework/test_ops.json': exported_api.ExportedApi(
        docs=[],
        symbols=[
            exported_api.ExportedSymbol(
                file_name='tf/python/framework/test_ops.py',
                line_no=2,
                symbol_name='a',
                v1_apis=(),
                v2_apis=('a',),
            )
        ],
    ),
}


def write_test_data(tmp_dir: str):
  for f in test_data:
    file_name = os.path.join(tmp_dir, f)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    test_data[f].write(file_name)


class GeneratorTest(parameterized.TestCase):

  def test_get_public_api(self):
    tmp_dir = self.create_tempdir()
    write_test_data(tmp_dir.full_path)

    expected_tensor_top_level = generator._Entrypoint(
        module='tf',
        name='Tensor',
        exported_symbol=tensor_es,
    )

    expected = generator.PublicAPI(
        v1_entrypoints_by_module=collections.defaultdict(set),
        v2_entrypoints_by_module=collections.defaultdict(set),
        v1_generated_imports_by_module=collections.defaultdict(set),
        v2_generated_imports_by_module=collections.defaultdict(set),
        docs_by_module={},
    )
    expected.v1_entrypoints_by_module['tf'].add(expected_tensor_top_level)
    expected.v2_entrypoints_by_module['tf'].add(expected_tensor_top_level)
    expected.v2_entrypoints_by_module['tf.experimental.numpy'].add(
        generator._Entrypoint(
            module='tf.experimental.numpy',
            name='ndarray',
            exported_symbol=tensor_es,
        )
    )
    expected.v2_generated_imports_by_module['tf'].add('tf.experimental')
    expected.v2_generated_imports_by_module['tf.experimental'].add(
        'tf.experimental.numpy'
    )

    got = generator.get_public_api(
        [os.path.join(tmp_dir, f) for f in test_data],
        file_prefixes_to_strip=[tmp_dir.full_path],
        packages_to_ignore=['tf.python.framework.test_ops'],
        output_package='tf',
        module_prefix=''
    )

    self.assertEqual(
        expected,
        got,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='normal_file',
          entrypoint=generator._Entrypoint(
              module='tf.io',
              name='decode_csv',
              exported_symbol=exported_api.ExportedSymbol(
                  file_name='tf/python/ops/parsing_ops.py',
                  line_no=10,
                  symbol_name='decode_csv_v2',
                  v1_apis=[],
                  v2_apis=['tf.io.decode_csv'],
              ),
          ),
          prefixes_to_strip=[],
          expected='tf.python.ops.parsing_ops',
      ),
      dict(
          testcase_name='genfile',
          entrypoint=generator._Entrypoint(
              module='tf.io',
              name='decode_proto_v2',
              exported_symbol=exported_api.ExportedSymbol(
                  file_name=(
                      'bazel-out/genfiles/tf/python/ops/gen_decode_proto_ops.py'
                  ),
                  line_no=20,
                  symbol_name='decode_proto_v2',
                  v1_apis=[],
                  v2_apis=['tf.io.decode_proto_v2'],
              ),
          ),
          prefixes_to_strip=['bazel-out/genfiles'],
          expected='tf.python.ops.gen_decode_proto_ops',
      ),
  )
  def test_get_import_path(self, entrypoint, prefixes_to_strip, expected):
    self.assertEqual(
        expected,
        generator._get_import_path(
            entrypoint.exported_symbol.file_name, prefixes_to_strip, ''
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='direct',
          entrypoint=generator._Entrypoint(
              module='tf',
              name='Tensor',
              exported_symbol=tensor_es,
          ),
          use_lazy_loading=False,
          expected='from tf.python.framework.tensor import Tensor # line: 1',
      ),
      dict(
          testcase_name='alias',
          entrypoint=generator._Entrypoint(
              module='tf.io',
              name='decode_csv',
              exported_symbol=exported_api.ExportedSymbol(
                  file_name='tf/python/ops/parsing_ops.py',
                  line_no=10,
                  symbol_name='decode_csv_v2',
                  v1_apis=[],
                  v2_apis=['tf.io.decode_csv'],
              ),
          ),
          use_lazy_loading=False,
          expected=(
              'from tf.python.ops.parsing_ops import decode_csv_v2 as'
              ' decode_csv # line: 10'
          ),
      ),
      dict(
          testcase_name='direct_lazy',
          entrypoint=generator._Entrypoint(
              module='tf',
              name='Tensor',
              exported_symbol=tensor_es,
          ),
          use_lazy_loading=True,
          expected=(
              "  'Tensor': ('tf.python.framework.tensor', 'Tensor'), # line: 1"
          ),
      ),
      dict(
          testcase_name='alias_lazy',
          entrypoint=generator._Entrypoint(
              module='tf.io',
              name='decode_csv',
              exported_symbol=exported_api.ExportedSymbol(
                  file_name='tf/python/ops/parsing_ops.py',
                  line_no=10,
                  symbol_name='decode_csv_v2',
                  v1_apis=[],
                  v2_apis=['tf.io.decode_csv'],
              ),
          ),
          use_lazy_loading=True,
          expected=(
              "  'decode_csv': ('tf.python.ops.parsing_ops',"
              " 'decode_csv_v2'), # line: 10"
          ),
      ),
  )
  def test_entrypoint_get_import(self, entrypoint, use_lazy_loading, expected):
    self.assertEqual(expected, entrypoint.get_import([], '', use_lazy_loading))

  def test_get_module(self):
    self.assertEqual(
        'keras.losses',
        generator.get_module(
            'bazel/tensorflow/keras/losses/', 'bazel/tensorflow'
        ),
    )

  def test_generate_proxy_api_files(self):
    tmp_dir = self.create_tempdir()
    proxy_file = os.path.join(tmp_dir, 'tensorflow/keras/losses/__init__.py')
    generator.generate_proxy_api_files(
        [proxy_file], 'keras', os.path.join(tmp_dir, 'tensorflow/keras')
    )
    self.assertTrue(os.path.isfile(proxy_file))
    with open(proxy_file, 'r') as f:
      self.assertEqual('from keras.losses import *', f.read())

  def test_get_module_docstring(self):
    docs_by_module = {
        'io': 'io docs',
    }
    self.assertEqual(
        'io docs', generator._get_module_docstring(docs_by_module, 'io')
    )
    self.assertEqual(
        'Public API for math namespace',
        generator._get_module_docstring(docs_by_module, 'math'),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='static_imports',
          use_lazy_loading=False,
          subpackage_rewrite=None,
          expected="""from tf import io
from tf.python.framework.tensor import Tensor # line: 1
""",
      ),
      dict(
          testcase_name='lazy_imports',
          use_lazy_loading=True,
          subpackage_rewrite=None,
          expected="""  'io': ('', 'tf.io'),
  'Tensor': ('tf.python.framework.tensor', 'Tensor'), # line: 1
""",
      ),
      dict(
          testcase_name='subpackage_rewrite',
          use_lazy_loading=False,
          subpackage_rewrite='tf.compat.v1',
          expected="""from tf.compat.v1 import io
from tf.python.framework.tensor import Tensor # line: 1
""",
      ),
  )
  def test_get_imports_for_module(
      self, use_lazy_loading, subpackage_rewrite, expected
  ):
    symbols_by_module = {
        'tf': {
            generator._Entrypoint(
                module='tf', name='Tensor', exported_symbol=tensor_es
            )
        }
    }
    generated_imports_by_module = {'tf': {'tf.io'}}
    self.assertEqual(
        expected,
        generator._get_imports_for_module(
            'tf',
            'tf',
            symbols_by_module,
            generated_imports_by_module,
            [],
            '',
            use_lazy_loading,
            subpackage_rewrite,
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_prefixes_and_packages',
          file='tf/python/framework/test_ops.py',
          file_prefixes_to_strip=[],
          packages_to_ignore=[],
          should_skip=False,
      ),
      dict(
          testcase_name='empty_prefix_nonempty_package',
          file='tf/python/framework/test_ops.py',
          file_prefixes_to_strip=[],
          packages_to_ignore=['tf.python.framework.test_ops'],
          should_skip=True,
      ),
      dict(
          testcase_name='nonempty_prefix_empty_package',
          file='gen/tf/python/framework/test_ops.py',
          file_prefixes_to_strip=['gen/'],
          packages_to_ignore=[],
          should_skip=False,
      ),
      dict(
          testcase_name='nonempty_prefix_nonempty_package',
          file='gen/tf/python/framework/test_ops.py',
          file_prefixes_to_strip=['gen/'],
          packages_to_ignore=['tf.python.framework.test_ops'],
          should_skip=True,
      ),
      dict(
          testcase_name='non_matching_prefix_and_package',
          file='tf/python/ops/test_ops.py',
          file_prefixes_to_strip=['gen/'],
          packages_to_ignore=['tf.python.framework.test_ops'],
          should_skip=False,
      ),
  )
  def test_should_skip_file(
      self, file, file_prefixes_to_strip, packages_to_ignore, should_skip
  ):
    self.assertEqual(
        should_skip,
        generator._should_skip_file(
            file, file_prefixes_to_strip, packages_to_ignore, '',
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='default',
          root_file_name=None,
      ),
      dict(
          testcase_name='renamed_root',
          root_file_name='v2.py',
      ),
  )
  def test_gen_init_files(self, root_file_name):
    output_dir = self.create_tempdir()
    mapping_dir = self.create_tempdir()
    write_test_data(mapping_dir.full_path)

    file_prefixes_to_strip = [mapping_dir.full_path]

    public_api = generator.get_public_api(
        [os.path.join(mapping_dir, f) for f in test_data],
        file_prefixes_to_strip=file_prefixes_to_strip,
        packages_to_ignore=['tf.python.framework.test_ops'],
        output_package='tf',
        module_prefix='',
    )

    paths_expected = [
        root_file_name if root_file_name else '__init__.py',
        'experimental/__init__.py',
        'experimental/numpy/__init__.py',
    ]
    paths_expected = set(
        [
            os.path.normpath(os.path.join(output_dir, path))
            for path in paths_expected
        ]
    )
    if root_file_name is None:
      generator._gen_init_files(
          output_dir,
          'tf',
          2,
          public_api.v2_entrypoints_by_module,
          public_api.v2_generated_imports_by_module,
          public_api.docs_by_module,
          '',
          file_prefixes_to_strip,
          False,
          '',
          paths_expected,
      )
    else:
      generator._gen_init_files(
          output_dir,
          'tf',
          2,
          public_api.v2_entrypoints_by_module,
          public_api.v2_generated_imports_by_module,
          public_api.docs_by_module,
          '',
          file_prefixes_to_strip,
          False,
          '',
          paths_expected,
          root_file_name=root_file_name,
      )
    expected_init_path = os.path.join(
        output_dir.full_path,
        root_file_name if root_file_name else '__init__.py',
    )
    self.assertTrue(os.path.exists(expected_init_path))
    with open(expected_init_path, 'r') as f:
      self.assertEqual(
          f.read(),
          """# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator2/generator/generator.py script.
\"""Public API for tf namespace
\"""

import sys as _sys

from tf import experimental
from tf.python.framework.tensor import Tensor # line: 1
""",
      )
    expected_numpy_path = os.path.join(
        output_dir.full_path, 'experimental/numpy/__init__.py'
    )
    self.assertTrue(os.path.exists(expected_numpy_path))
    with open(expected_numpy_path, 'r') as f:
      self.assertEqual(
          f.read(),
          """# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator2/generator/generator.py script.
\"""Public API for tf.experimental.numpy namespace
\"""

import sys as _sys

from tf.python.framework.tensor import Tensor as ndarray # line: 1
""",
      )

  def testRaisesOnNotExpectedFile(self):
    output_dir = self.create_tempdir()
    mapping_dir = self.create_tempdir()
    write_test_data(mapping_dir.full_path)

    file_prefixes_to_strip = [mapping_dir.full_path]

    public_api = generator.get_public_api(
        [os.path.normpath(os.path.join(mapping_dir, f)) for f in test_data],
        file_prefixes_to_strip=file_prefixes_to_strip,
        packages_to_ignore=['tf.python.framework.test_ops'],
        output_package='tf',
        module_prefix='',
    )

    with self.assertRaisesRegex(
        AssertionError, 'Exported api attempted to write to'
    ):
      generator._gen_init_files(
          output_dir,
          'tf',
          2,
          public_api.v2_entrypoints_by_module,
          public_api.v2_generated_imports_by_module,
          public_api.docs_by_module,
          '',
          file_prefixes_to_strip,
          False,
          '',
          [],
      )


if __name__ == '__main__':
  absltest.main()
