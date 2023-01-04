# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for fenced_doctest."""
from typing import List, Optional, Tuple

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow.tools.docs import fenced_doctest_lib

EXAMPLES = [
    # pyformat: disable
    ('simple', [('code', None)], """
     Hello

     ``` python
     code
     ```

     Goodbye
     """),
    ('output', [('code', 'result')], """
     Hello

     ``` python
     code
     ```

     ```
     result
     ```

     Goodbye
     """),
    ('not-output', [('code', None)], """
     Hello

     ``` python
     code
     ```

     ``` bash
     result
     ```

     Goodbye
     """),
    ('first', [('code', None)], """
     ``` python
     code
     ```

     Goodbye
     """[1:]),
    ('last', [('code', None)], """
     Hello

     ``` python
     code
     ```"""),
    ('last_output', [('code', 'result')], """
     Hello

     ``` python
     code
     ```

     ```
     result
     ```"""),
    ('skip-unlabeled', [], """
     Hello

     ```
     skip
     ```

     Goodbye
     """),
    ('skip-wrong-label', [], """
     Hello

     ``` sdkfjgsd
     skip
     ```

     Goodbye
     """),
    ('doctest_skip', [], """
     Hello

     ``` python
     doctest: +SKIP
     ```

     Goodbye
     """),
    ('skip_all', [], """
     <!-- doctest: skip-all -->

     Hello

     ``` python
     a
     ```

     ``` python
     b
     ```

     Goodbye
     """),
    ('two', [('a', None), ('b', None)], """
     Hello

     ``` python
     a
     ```

     ``` python
     b
     ```

     Goodbye
     """),
    ('two-outputs', [('a', 'A'), ('b', 'B')], """
     Hello

     ``` python
     a
     ```

     ```
     A
     ```

     ``` python
     b
     ```

     ```
     B
     ```

     Goodbye
     """),
    ('list', [('a', None), ('b', 'B'), ('c', 'C'), ('d', None)], """
     Hello

     ``` python
     a
     ```

     ``` python
     b
     ```

     ```
     B
     ```

     List:
     * first

       ``` python
       c
       ```

       ```
       C
       ```

       ``` python
       d
       ```
     * second


     Goodbye
     """),
    ('multiline', [('a\nb', 'A\nB')], """
     Hello

     ``` python
     a
     b
     ```

     ```
     A
     B
     ```

     Goodbye
     """)
]

ExampleTuples = List[Tuple[str, Optional[str]]]


class G3DoctestTest(parameterized.TestCase):

  def _do_test(self, expected_example_tuples, string):
    parser = fenced_doctest_lib.FencedCellParser(fence_label='python')

    example_tuples = []
    for example in parser.get_examples(string, name=self._testMethodName):
      source = example.source.rstrip('\n')
      want = example.want
      if want is not None:
        want = want.rstrip('\n')
      example_tuples.append((source, want))

    self.assertEqual(expected_example_tuples, example_tuples)

  @parameterized.named_parameters(*EXAMPLES)
  def test_parser(self, expected_example_tuples: ExampleTuples, string: str):
    self._do_test(expected_example_tuples, string)

  @parameterized.named_parameters(*EXAMPLES)
  def test_parser_no_blanks(self, expected_example_tuples: ExampleTuples,
                            string: str):
    string = string.replace('\n\n', '\n')
    self._do_test(expected_example_tuples, string)


if __name__ == '__main__':
  absltest.main()
