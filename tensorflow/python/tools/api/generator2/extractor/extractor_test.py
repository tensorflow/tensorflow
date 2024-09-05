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
# =============================================================================
from absl.testing import absltest

from tensorflow.python.tools.api.generator2.extractor import extractor
from tensorflow.python.tools.api.generator2.shared import exported_api


class ParserTest(absltest.TestCase):

  def test_exported_docstring(self):
    exporter = exported_api.ExportedApi()
    p = extractor.Parser(
        exporter,
        decorator='tf.tf_export',
        api_name='tf',
    )
    p.process(
        'test.py',
        '''# 1
"""this is an exported docstring.
API docstring: tf.test
"""  # 4
        ''',
    )
    self.assertEqual(
        exporter,
        exported_api.ExportedApi(
            docs=[
                exported_api.ExportedDoc(
                    file_name='test.py',
                    line_no=2,
                    docstring='this is an exported docstring.',
                    modules=('tf.test',),
                )
            ],
        ),
    )

  def test_exported_docstring_not_at_top_level(self):
    exporter = exported_api.ExportedApi()
    p = extractor.Parser(
        exporter,
        decorator='tf.tf_export',
        api_name='tf',
    )
    self.assertRaisesRegex(
        extractor.BadExportError,
        'test.py:3',
        lambda: p.process(  # pylint: disable=g-long-lambda
            'test.py',
            '''# 1
def a():  # 2
  """a docstring
  API docstring: tf.test
  """  # 5
          ''',
        ),
    )

  def test_exported_symbol(self):
    exporter = exported_api.ExportedApi()
    p = extractor.Parser(
        exporter,
        decorator='extractor.api_export.tf_export',
        api_name='tf',
    )
    p.process(
        'test.py',
        """# 1
from extractor import api_export  # 2
from extractor import api_export as ae  # 3
try:  # 4
  from extractor.api_export import tf_export  # 5
except ImportError:  # 6
  pass  # 7
from extractor.api_export import tf_export as tfe # 8
from extractor.api_export import other_export  # 9
_a = api_export.tf_export("a")(foo)  # 10
api_export.tf_export("b", v1=["v1_b"])(_b)  # 11
tfe("c")(_c)  # 12
@ae.tf_export("d")  # 13
class _D():  # 14
  pass  # 15
@api_export.tf_export("e", "e_v2", v1=[])  # 16
def _e():  # 17
  pass  # 18
tf_export(v1=["f", "f_alias"])(  # 19
    dispatch.dispatch(deprecation(_f))  # 20
)  # 21
@other_export("not-exported")  # 22
def _not_exported():  # 23
  pass  # 24
        """,
    )
    self.assertEqual(
        exporter,
        exported_api.ExportedApi(
            symbols=[
                exported_api.ExportedSymbol(
                    file_name='test.py',
                    line_no=10,
                    symbol_name='_a',
                    v1_apis=('tf.a',),
                    v2_apis=('tf.a',),
                ),
                exported_api.ExportedSymbol(
                    file_name='test.py',
                    line_no=11,
                    symbol_name='_b',
                    v1_apis=('tf.v1_b',),
                    v2_apis=('tf.b',),
                ),
                exported_api.ExportedSymbol(
                    file_name='test.py',
                    line_no=12,
                    symbol_name='_c',
                    v1_apis=('tf.c',),
                    v2_apis=('tf.c',),
                ),
                exported_api.ExportedSymbol(
                    file_name='test.py',
                    line_no=13,
                    symbol_name='_D',
                    v1_apis=('tf.d',),
                    v2_apis=('tf.d',),
                ),
                exported_api.ExportedSymbol(
                    file_name='test.py',
                    line_no=16,
                    symbol_name='_e',
                    v1_apis=(),
                    v2_apis=('tf.e', 'tf.e_v2'),
                ),
                exported_api.ExportedSymbol(
                    file_name='test.py',
                    line_no=19,
                    symbol_name='_f',
                    v1_apis=('tf.f', 'tf.f_alias'),
                    v2_apis=(),
                ),
            ],
        ),
    )

  def test_exported_symbol_not_at_top_level(self):
    exporter = exported_api.ExportedApi()
    p = extractor.Parser(
        exporter,
        decorator='tf.tf_export',
        api_name='tf',
    )
    self.assertRaisesRegex(
        extractor.BadExportError,
        'test.py:4',
        lambda: p.process(  # pylint: disable=g-long-lambda
            'test.py',
            """# 1
from tf import tf_export  # 2
def method():  # 3
  tf_export("a")(a)  # 4
            """,
        ),
    )

  def test_exported_symbol_not_applied(self):
    exporter = exported_api.ExportedApi()
    p = extractor.Parser(
        exporter,
        decorator='tf.tf_export',
        api_name='tf',
    )
    self.assertRaisesRegex(
        extractor.BadExportError,
        'test.py:3',
        lambda: p.process(  # pylint: disable=g-long-lambda
            'test.py',
            """# 1
from tf import tf_export  # 2
tf_export("a")  # 3
            """,
        ),
    )

  def test_exported_symbol_non_literal_args(self):
    exporter = exported_api.ExportedApi()
    p = extractor.Parser(
        exporter,
        decorator='tf.tf_export',
        api_name='tf',
    )
    self.assertRaisesRegex(
        extractor.BadExportError,
        'test.py:3',
        lambda: p.process(  # pylint: disable=g-long-lambda
            'test.py',
            """# 1
from tf import tf_export  # 2
tf_export(a)(b)  # 3
            """,
        ),
    )

  def test_exported_symbol_unknown_args(self):
    exporter = exported_api.ExportedApi()
    p = extractor.Parser(
        exporter,
        decorator='tf.tf_export',
        api_name='tf',
    )
    self.assertRaisesRegex(
        extractor.BadExportError,
        'test.py:3',
        lambda: p.process(  # pylint: disable=g-long-lambda
            'test.py',
            """# 1
from tf import tf_export  # 2
tf_export(a)(b)  # 3
            """,
        ),
    )

  def test_exported_symbol_includes_module(self):
    exporter = exported_api.ExportedApi()
    p = extractor.Parser(
        exporter,
        decorator='tf.tf_export',
        api_name='tf',
    )
    self.assertRaisesRegex(
        extractor.BadExportError,
        'test.py:3',
        lambda: p.process(  # pylint: disable=g-long-lambda
            'test.py',
            """# 1
from tf import tf_export  # 2
tf_export(a)(x.b)  # 3
            """,
        ),
    )


if __name__ == '__main__':
  absltest.main()
