#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.CsvDataset`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import zlib

from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class CsvDatasetTest(test_base.DatasetTestBase):

  def _setup_files(self, inputs, linebreak='\n', compression_type=None):
    filenames = []
    for i, ip in enumerate(inputs):
      fn = os.path.join(self.get_temp_dir(), 'temp_%d.csv' % i)
      contents = linebreak.join(ip).encode('utf-8')
      if compression_type is None:
        with open(fn, 'wb') as f:
          f.write(contents)
      elif compression_type == 'GZIP':
        with gzip.GzipFile(fn, 'wb') as f:
          f.write(contents)
      elif compression_type == 'ZLIB':
        contents = zlib.compress(contents)
        with open(fn, 'wb') as f:
          f.write(contents)
      else:
        raise ValueError('Unsupported compression_type', compression_type)
      filenames.append(fn)
    return filenames

  def _make_test_datasets(self, inputs, **kwargs):
    # Test by comparing its output to what we could get with map->decode_csv
    filenames = self._setup_files(inputs)
    dataset_expected = core_readers.TextLineDataset(filenames)
    dataset_expected = dataset_expected.map(
        lambda l: parsing_ops.decode_csv(l, **kwargs))
    dataset_actual = readers.CsvDataset(filenames, **kwargs)
    return (dataset_actual, dataset_expected)

  def _test_by_comparison(self, inputs, **kwargs):
    """Checks that CsvDataset is equiv to TextLineDataset->map(decode_csv)."""
    dataset_actual, dataset_expected = self._make_test_datasets(
        inputs, **kwargs)
    self.assertDatasetsEqual(dataset_actual, dataset_expected)

  def _verify_output_or_err(self,
                            dataset,
                            expected_output=None,
                            expected_err_re=None):
    if expected_err_re is None:
      # Verify that output is expected, without errors
      nxt = self.getNext(dataset)
      expected_output = [[
          v.encode('utf-8') if isinstance(v, str) else v for v in op
      ] for op in expected_output]
      for value in expected_output:
        op = self.evaluate(nxt())
        self.assertAllEqual(op, value)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(nxt())
    else:
      # Verify that OpError is produced as expected
      with self.assertRaisesOpError(expected_err_re):
        nxt = self.getNext(dataset)
        while True:
          try:
            self.evaluate(nxt())
          except errors.OutOfRangeError:
            break

  def _test_dataset(
      self,
      inputs,
      expected_output=None,
      expected_err_re=None,
      linebreak='\n',
      compression_type=None,  # Used for both setup and parsing
      **kwargs):
    """Checks that elements produced by CsvDataset match expected output."""
    # Convert str type because py3 tf strings are bytestrings
    filenames = self._setup_files(inputs, linebreak, compression_type)
    kwargs['compression_type'] = compression_type
    dataset = readers.CsvDataset(filenames, **kwargs)
    self._verify_output_or_err(dataset, expected_output, expected_err_re)

  def testCsvDataset_requiredFields(self):
    record_defaults = [[]] * 4
    inputs = [['1,2,3,4']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  def testCsvDataset_int(self):
    record_defaults = [[0]] * 4
    inputs = [['1,2,3,4', '5,6,7,8']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  def testCsvDataset_float(self):
    record_defaults = [[0.0]] * 4
    inputs = [['1.0,2.1,3.2,4.3', '5.4,6.5,7.6,8.7']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  def testCsvDataset_string(self):
    record_defaults = [['']] * 4
    inputs = [['1.0,2.1,hello,4.3', '5.4,6.5,goodbye,8.7']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  def testCsvDataset_withEmptyFields(self):
    record_defaults = [[0]] * 4
    inputs = [[',,,', '1,1,1,', ',2,2,2']]
    self._test_dataset(
        inputs, [[0, 0, 0, 0], [1, 1, 1, 0], [0, 2, 2, 2]],
        record_defaults=record_defaults)

  def testCsvDataset_errWithUnquotedQuotes(self):
    record_defaults = [['']] * 3
    inputs = [['1,2"3,4']]
    self._test_dataset(
        inputs,
        expected_err_re='Unquoted fields cannot have quotes inside',
        record_defaults=record_defaults)

  def testCsvDataset_errWithUnescapedQuotes(self):
    record_defaults = [['']] * 3
    inputs = [['"a"b","c","d"']]
    self._test_dataset(
        inputs,
        expected_err_re=
        'Quote inside a string has to be escaped by another quote',
        record_defaults=record_defaults)

  def testCsvDataset_ignoreErrWithUnescapedQuotes(self):
    record_defaults = [['']] * 3
    inputs = [['1,"2"3",4', '1,"2"3",4",5,5', 'a,b,"c"d"', 'e,f,g']]
    filenames = self._setup_files(inputs)
    dataset = readers.CsvDataset(filenames, record_defaults=record_defaults)
    dataset = dataset.apply(error_ops.ignore_errors())
    self._verify_output_or_err(dataset, [['e', 'f', 'g']])

  def testCsvDataset_ignoreErrWithUnquotedQuotes(self):
    record_defaults = [['']] * 3
    inputs = [['1,2"3,4', 'a,b,c"d', '9,8"7,6,5', 'e,f,g']]
    filenames = self._setup_files(inputs)
    dataset = readers.CsvDataset(filenames, record_defaults=record_defaults)
    dataset = dataset.apply(error_ops.ignore_errors())
    self._verify_output_or_err(dataset, [['e', 'f', 'g']])

  def testCsvDataset_withNoQuoteDelimAndUnquotedQuotes(self):
    record_defaults = [['']] * 3
    inputs = [['1,2"3,4']]
    self._test_by_comparison(
        inputs, record_defaults=record_defaults, use_quote_delim=False)

  def testCsvDataset_mixedTypes(self):
    record_defaults = [
        constant_op.constant([], dtype=dtypes.int32),
        constant_op.constant([], dtype=dtypes.float32),
        constant_op.constant([], dtype=dtypes.string),
        constant_op.constant([], dtype=dtypes.float64)
    ]
    inputs = [['1,2.1,3.2,4.3', '5,6.5,7.6,8.7']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  def testCsvDataset_withUseQuoteDelimFalse(self):
    record_defaults = [['']] * 4
    inputs = [['1,2,"3,4"', '"5,6",7,8']]
    self._test_by_comparison(
        inputs, record_defaults=record_defaults, use_quote_delim=False)

  def testCsvDataset_withFieldDelim(self):
    record_defaults = [[0]] * 4
    inputs = [['1:2:3:4', '5:6:7:8']]
    self._test_by_comparison(
        inputs, record_defaults=record_defaults, field_delim=':')

  def testCsvDataset_withNaValue(self):
    record_defaults = [[0]] * 4
    inputs = [['1,NA,3,4', 'NA,6,7,8']]
    self._test_by_comparison(
        inputs, record_defaults=record_defaults, na_value='NA')

  def testCsvDataset_withSelectCols(self):
    record_defaults = [['']] * 2
    inputs = [['1,2,3,4', '"5","6","7","8"']]
    self._test_by_comparison(
        inputs, record_defaults=record_defaults, select_cols=[1, 2])

  def testCsvDataset_withSelectColsTooHigh(self):
    record_defaults = [[0]] * 2
    inputs = [['1,2,3,4', '5,6,7,8']]
    self._test_dataset(
        inputs,
        expected_err_re='Expect 2 fields but have 1 in record',
        record_defaults=record_defaults,
        select_cols=[3, 4])

  def testCsvDataset_withOneCol(self):
    record_defaults = [['NA']]
    inputs = [['0', '', '2']]
    self._test_dataset(
        inputs, [['0'], ['NA'], ['2']], record_defaults=record_defaults)

  def testCsvDataset_withMultipleFiles(self):
    record_defaults = [[0]] * 4
    inputs = [['1,2,3,4', '5,6,7,8'], ['5,6,7,8']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  def testCsvDataset_withLeadingAndTrailingSpaces(self):
    record_defaults = [[0.0]] * 4
    inputs = [['0, 1, 2, 3']]
    expected = [[0.0, 1.0, 2.0, 3.0]]
    self._test_dataset(inputs, expected, record_defaults=record_defaults)

  def testCsvDataset_errorWithMissingDefault(self):
    record_defaults = [[]] * 2
    inputs = [['0,']]
    self._test_dataset(
        inputs,
        expected_err_re='Field 1 is required but missing in record!',
        record_defaults=record_defaults)

  def testCsvDataset_errorWithFewerDefaultsThanFields(self):
    record_defaults = [[0.0]] * 2
    inputs = [['0,1,2,3']]
    self._test_dataset(
        inputs,
        expected_err_re='Expect 2 fields but have more in record',
        record_defaults=record_defaults)

  def testCsvDataset_errorWithMoreDefaultsThanFields(self):
    record_defaults = [[0.0]] * 5
    inputs = [['0,1,2,3']]
    self._test_dataset(
        inputs,
        expected_err_re='Expect 5 fields but have 4 in record',
        record_defaults=record_defaults)

  def testCsvDataset_withHeader(self):
    record_defaults = [[0]] * 2
    inputs = [['col1,col2', '1,2']]
    expected = [[1, 2]]
    self._test_dataset(
        inputs,
        expected,
        record_defaults=record_defaults,
        header=True,
    )

  def testCsvDataset_withHeaderAndNoRecords(self):
    record_defaults = [[0]] * 2
    inputs = [['col1,col2']]
    expected = []
    self._test_dataset(
        inputs,
        expected,
        record_defaults=record_defaults,
        header=True,
    )

  def testCsvDataset_errorWithHeaderEmptyFile(self):
    record_defaults = [[0]] * 2
    inputs = [[]]
    expected_err_re = "Can't read header of file"
    self._test_dataset(
        inputs,
        expected_err_re=expected_err_re,
        record_defaults=record_defaults,
        header=True,
    )

  def testCsvDataset_withEmptyFile(self):
    record_defaults = [['']] * 2
    inputs = [['']]  # Empty file
    self._test_dataset(
        inputs, expected_output=[], record_defaults=record_defaults)

  def testCsvDataset_errorWithEmptyRecord(self):
    record_defaults = [['']] * 2
    inputs = [['', '1,2']]  # First record is empty
    self._test_dataset(
        inputs,
        expected_err_re='Expect 2 fields but have 1 in record',
        record_defaults=record_defaults)

  def testCsvDataset_withChainedOps(self):
    # Testing that one dataset can create multiple iterators fine.
    # `repeat` creates multiple iterators from the same C++ Dataset.
    record_defaults = [[0]] * 4
    inputs = [['1,,3,4', '5,6,,8']]
    ds_actual, ds_expected = self._make_test_datasets(
        inputs, record_defaults=record_defaults)
    self.assertDatasetsEqual(
        ds_actual.repeat(5).prefetch(1),
        ds_expected.repeat(5).prefetch(1))

  def testCsvDataset_withTypeDefaults(self):
    # Testing using dtypes as record_defaults for required fields
    record_defaults = [dtypes.float32, [0.0]]
    inputs = [['1.0,2.0', '3.0,4.0']]
    self._test_dataset(
        inputs,
        [[1.0, 2.0], [3.0, 4.0]],
        record_defaults=record_defaults,
    )

  def testMakeCsvDataset_fieldOrder(self):
    data = [[
        '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19',
        '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19'
    ]]
    file_path = self._setup_files(data)

    ds = readers.make_csv_dataset(
        file_path, batch_size=1, shuffle=False, num_epochs=1)
    nxt = self.getNext(ds)

    result = list(self.evaluate(nxt()).values())

    self.assertEqual(result, sorted(result))

## The following tests exercise parsing logic for quoted fields

  def testCsvDataset_withQuoted(self):
    record_defaults = [['']] * 4
    inputs = [['"a","b","c :)","d"', '"e","f","g :(","h"']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  def testCsvDataset_withOneColAndQuotes(self):
    record_defaults = [['']]
    inputs = [['"0"', '"1"', '"2"']]
    self._test_dataset(
        inputs, [['0'], ['1'], ['2']], record_defaults=record_defaults)

  def testCsvDataset_withNewLine(self):
    # In this case, we expect it to behave differently from
    # TextLineDataset->map(decode_csv) since that flow has bugs
    record_defaults = [['']] * 4
    inputs = [['a,b,"""c""\n0","d\ne"', 'f,g,h,i']]
    expected = [['a', 'b', '"c"\n0', 'd\ne'], ['f', 'g', 'h', 'i']]
    self._test_dataset(inputs, expected, record_defaults=record_defaults)

  def testCsvDataset_withNewLineInUnselectedCol(self):
    record_defaults = [['']]
    inputs = [['1,"2\n3",4', '5,6,7']]
    self._test_dataset(
        inputs,
        expected_output=[['1'], ['5']],
        record_defaults=record_defaults,
        select_cols=[0])

  def testCsvDataset_withMultipleNewLines(self):
    # In this case, we expect it to behave differently from
    # TextLineDataset->map(decode_csv) since that flow has bugs
    record_defaults = [['']] * 4
    inputs = [['a,"b\n\nx","""c""\n \n0","d\ne"', 'f,g,h,i']]
    expected = [['a', 'b\n\nx', '"c"\n \n0', 'd\ne'], ['f', 'g', 'h', 'i']]
    self._test_dataset(inputs, expected, record_defaults=record_defaults)

  def testCsvDataset_errorWithTerminateMidRecord(self):
    record_defaults = [['']] * 4
    inputs = [['a,b,c,"a']]
    self._test_dataset(
        inputs,
        expected_err_re=
        'Reached end of file without closing quoted field in record',
        record_defaults=record_defaults)

  def testCsvDataset_withEscapedQuotes(self):
    record_defaults = [['']] * 4
    inputs = [['1.0,2.1,"she said: ""hello""",4.3', '5.4,6.5,goodbye,8.7']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)


## Testing that parsing works with all buffer sizes, quoted/unquoted fields,
## and different types of line breaks

  def testCsvDataset_withInvalidBufferSize(self):
    record_defaults = [['']] * 4
    inputs = [['a,b,c,d']]
    self._test_dataset(
        inputs,
        expected_err_re='buffer_size should be positive',
        record_defaults=record_defaults,
        buffer_size=0)

  def _test_dataset_on_buffer_sizes(self,
                                    inputs,
                                    expected,
                                    linebreak,
                                    record_defaults,
                                    compression_type=None,
                                    num_sizes_to_test=20):
    # Testing reading with a range of buffer sizes that should all work.
    for i in list(range(1, 1 + num_sizes_to_test)) + [None]:
      self._test_dataset(
          inputs,
          expected,
          linebreak=linebreak,
          compression_type=compression_type,
          record_defaults=record_defaults,
          buffer_size=i)

  def testCsvDataset_withLF(self):
    record_defaults = [['NA']] * 3
    inputs = [['abc,def,ghi', '0,1,2', ',,']]
    expected = [['abc', 'def', 'ghi'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\n', record_defaults=record_defaults)

  def testCsvDataset_withCR(self):
    # Test that when the line separator is '\r', parsing works with all buffer
    # sizes
    record_defaults = [['NA']] * 3
    inputs = [['abc,def,ghi', '0,1,2', ',,']]
    expected = [['abc', 'def', 'ghi'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\r', record_defaults=record_defaults)

  def testCsvDataset_withCRLF(self):
    # Test that when the line separator is '\r\n', parsing works with all buffer
    # sizes
    record_defaults = [['NA']] * 3
    inputs = [['abc,def,ghi', '0,1,2', ',,']]
    expected = [['abc', 'def', 'ghi'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\r\n', record_defaults=record_defaults)

  def testCsvDataset_withBufferSizeAndQuoted(self):
    record_defaults = [['NA']] * 3
    inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
    expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'],
                ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\n', record_defaults=record_defaults)

  def testCsvDataset_withCRAndQuoted(self):
    # Test that when the line separator is '\r', parsing works with all buffer
    # sizes
    record_defaults = [['NA']] * 3
    inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
    expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'],
                ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\r', record_defaults=record_defaults)

  def testCsvDataset_withCRLFAndQuoted(self):
    # Test that when the line separator is '\r\n', parsing works with all buffer
    # sizes
    record_defaults = [['NA']] * 3
    inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
    expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'],
                ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\r\n', record_defaults=record_defaults)

  def testCsvDataset_withGzipCompressionType(self):
    record_defaults = [['NA']] * 3
    inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
    expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'],
                ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs,
        expected,
        linebreak='\r\n',
        compression_type='GZIP',
        record_defaults=record_defaults)

  def testCsvDataset_withZlibCompressionType(self):
    record_defaults = [['NA']] * 3
    inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
    expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'],
                ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs,
        expected,
        linebreak='\r\n',
        compression_type='ZLIB',
        record_defaults=record_defaults)

  def testCsvDataset_withScalarDefaults(self):
    record_defaults = [constant_op.constant(0, dtype=dtypes.int64)] * 4
    inputs = [[',,,', '1,1,1,', ',2,2,2']]
    self._test_dataset(
        inputs, [[0, 0, 0, 0], [1, 1, 1, 0], [0, 2, 2, 2]],
        record_defaults=record_defaults)

  def testCsvDataset_with2DDefaults(self):
    record_defaults = [constant_op.constant([[0]], dtype=dtypes.int64)] * 4
    inputs = [[',,,', '1,1,1,', ',2,2,2']]

    if context.executing_eagerly():
      err_spec = errors.InvalidArgumentError, (
          'Each record default should be at '
          'most rank 1.')
    else:
      err_spec = ValueError, 'Shape must be at most rank 1 but is rank 2'

    with self.assertRaisesWithPredicateMatch(*err_spec):
      self._test_dataset(
          inputs, [[0, 0, 0, 0], [1, 1, 1, 0], [0, 2, 2, 2]],
          record_defaults=record_defaults)


if __name__ == '__main__':
  test.main()
