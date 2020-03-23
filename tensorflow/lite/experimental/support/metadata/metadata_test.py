# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.lite.experimental.support.metadata.metadata."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six

from flatbuffers.python import flatbuffers
from tensorflow.lite.experimental.support.metadata import metadata as _metadata
from tensorflow.lite.experimental.support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow.lite.experimental.support.metadata import schema_py_generated as _schema_fb
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class MetadataTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(MetadataTest, self).setUp()
    self._invalid_model_buf = None
    self._invalid_file = "not_existed_file"
    self._empty_model_buf = self._create_empty_model_buf()
    self._empty_model_file = self.create_tempfile().full_path
    with open(self._empty_model_file, "wb") as f:
      f.write(self._empty_model_buf)
    self._model_file = self._create_model_file_with_metadata_and_buf_fields()
    self._metadata_file = self._create_metadata_file()
    self._file1 = self.create_tempfile("file1").full_path
    self._file2 = self.create_tempfile("file2").full_path
    self._file3 = self.create_tempfile("file3").full_path

  def _create_empty_model_buf(self):
    model = _schema_fb.ModelT()
    model_builder = flatbuffers.Builder(0)
    model_builder.Finish(
        model.Pack(model_builder),
        _metadata.MetadataPopulator.TFLITE_FILE_IDENTIFIER)
    return model_builder.Output()

  def _create_model_file_with_metadata_and_buf_fields(self):
    metadata_field = _schema_fb.MetadataT()
    metadata_field.name = "meta"
    buffer_field = _schema_fb.BufferT()
    model = _schema_fb.ModelT()
    model.metadata = [metadata_field, metadata_field]
    model.buffers = [buffer_field, buffer_field, buffer_field]
    model_builder = flatbuffers.Builder(0)
    model_builder.Finish(
        model.Pack(model_builder),
        _metadata.MetadataPopulator.TFLITE_FILE_IDENTIFIER)

    mnodel_file = self.create_tempfile().full_path
    with open(mnodel_file, "wb") as f:
      f.write(model_builder.Output())

    return mnodel_file

  def _create_metadata_file(self):
    associated_file1 = _metadata_fb.AssociatedFileT()
    associated_file1.name = b"file1"
    associated_file2 = _metadata_fb.AssociatedFileT()
    associated_file2.name = b"file2"
    self.expected_recorded_files = [
        six.ensure_str(associated_file1.name),
        six.ensure_str(associated_file2.name)
    ]

    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.associatedFiles = [associated_file2]
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.outputTensorMetadata = [output_meta]

    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "Mobilenet_quantized"
    model_meta.associatedFiles = [associated_file1]
    model_meta.subgraphMetadata = [subgraph]
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)

    metadata_file = self.create_tempfile().full_path
    with open(metadata_file, "wb") as f:
      f.write(b.Output())
    return metadata_file


class MetadataPopulatorTest(MetadataTest):

  def testToValidModelFile(self):
    populator = _metadata.MetadataPopulator.with_model_file(
        self._empty_model_file)
    self.assertIsInstance(populator, _metadata.MetadataPopulator)

  def testToInvalidModelFile(self):
    with self.assertRaises(IOError) as error:
      _metadata.MetadataPopulator.with_model_file(self._invalid_file)
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def testToValidModelBuffer(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(
        self._empty_model_buf)
    self.assertIsInstance(populator, _metadata.MetadataPopulator)

  def testToInvalidModelBuffer(self):
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataPopulator.with_model_buffer(self._invalid_model_buf)
    self.assertEqual("model_buf cannot be empty.", str(error.exception))

  def testSinglePopulateAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(
        self._empty_model_buf)
    populator.load_associated_files([self._file1])
    populator.populate()

    packed_files = populator.get_packed_associated_file_list()
    expected_packed_files = [os.path.basename(self._file1)]
    self.assertEqual(set(packed_files), set(expected_packed_files))

  def testRepeatedPopulateAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_file(
        self._empty_model_file)
    populator.load_associated_files([self._file1, self._file2])
    # Loads file2 multiple times.
    populator.load_associated_files([self._file2])
    populator.populate()

    packed_files = populator.get_packed_associated_file_list()
    expected_packed_files = [
        os.path.basename(self._file1),
        os.path.basename(self._file2)
    ]
    self.assertEqual(len(packed_files), 2)
    self.assertEqual(set(packed_files), set(expected_packed_files))

    # Check if the model buffer read from file is the same as that read from
    # get_model_buffer().
    with open(self._empty_model_file, "rb") as f:
      model_buf_from_file = f.read()
    model_buf_from_getter = populator.get_model_buffer()
    self.assertEqual(model_buf_from_file, model_buf_from_getter)

  def testPopulateInvalidAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(
        self._empty_model_buf)
    with self.assertRaises(IOError) as error:
      populator.load_associated_files([self._invalid_file])
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def testPopulatePackedAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(
        self._empty_model_buf)
    populator.load_associated_files([self._file1])
    populator.populate()
    with self.assertRaises(ValueError) as error:
      populator.load_associated_files([self._file1])
      populator.populate()
    self.assertEqual(
        "File, '{0}', has already been packed.".format(
            os.path.basename(self._file1)), str(error.exception))

  def testGetPackedAssociatedFileList(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(
        self._empty_model_buf)
    packed_files = populator.get_packed_associated_file_list()
    self.assertEqual(packed_files, [])

  def testPopulateMetadataFileToEmptyModelFile(self):
    populator = _metadata.MetadataPopulator.with_model_file(
        self._empty_model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1, self._file2])
    populator.populate()

    with open(self._empty_model_file, "rb") as f:
      model_buf_from_file = f.read()
    model = _schema_fb.Model.GetRootAsModel(model_buf_from_file, 0)
    metadata_field = model.Metadata(0)
    self.assertEqual(
        six.ensure_str(metadata_field.Name()),
        six.ensure_str(_metadata.MetadataPopulator.METADATA_FIELD_NAME))

    buffer_index = metadata_field.Buffer()
    buffer_data = model.Buffers(buffer_index)
    metadata_buf_np = buffer_data.DataAsNumpy()
    metadata_buf = metadata_buf_np.tobytes()
    with open(self._metadata_file, "rb") as f:
      expected_metadata_buf = bytearray(f.read())
    self.assertEqual(metadata_buf, expected_metadata_buf)

    recorded_files = populator.get_recorded_associated_file_list()
    self.assertEqual(set(recorded_files), set(self.expected_recorded_files))

    # Up to now, we've proved the correctness of the model buffer that read from
    # file. Then we'll test if get_model_buffer() gives the same model buffer.
    model_buf_from_getter = populator.get_model_buffer()
    self.assertEqual(model_buf_from_file, model_buf_from_getter)

  def testPopulateMetadataFileWithoutAssociatedFiles(self):
    populator = _metadata.MetadataPopulator.with_model_file(
        self._empty_model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1])
    # Suppose to populate self._file2, because it is recorded in the metadta.
    with self.assertRaises(ValueError) as error:
      populator.populate()
    self.assertEqual(("File, '{0}', is recorded in the metadata, but has "
                      "not been loaded into the populator.").format(
                          os.path.basename(self._file2)), str(error.exception))

  def _assert_golden_metadata(self, model_file):
    with open(model_file, "rb") as f:
      model_buf_from_file = f.read()
    model = _schema_fb.Model.GetRootAsModel(model_buf_from_file, 0)
    # There are two elements in model.Metadata array before the population.
    # Metadata should be packed to the third element in the array.
    metadata_field = model.Metadata(2)
    self.assertEqual(
        six.ensure_str(metadata_field.Name()),
        six.ensure_str(_metadata.MetadataPopulator.METADATA_FIELD_NAME))

    buffer_index = metadata_field.Buffer()
    buffer_data = model.Buffers(buffer_index)
    metadata_buf_np = buffer_data.DataAsNumpy()
    metadata_buf = metadata_buf_np.tobytes()
    with open(self._metadata_file, "rb") as f:
      expected_metadata_buf = bytearray(f.read())
    self.assertEqual(metadata_buf, expected_metadata_buf)

  def testPopulateMetadataFileToModelWithMetadataAndAssociatedFiles(self):
    # First, creates a dummy metadata. Populates it and the associated files
    # into the model.
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "Mobilenet_quantized"
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    populator1 = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator1.load_metadata_buffer(metadata_buf)
    populator1.load_associated_files([self._file1, self._file2])
    populator1.populate()

    # Then, populates the metadata again.
    populator2 = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator2.load_metadata_file(self._metadata_file)
    populator2.populate()

    # Tests if the metadata is populated correctly.
    self._assert_golden_metadata(self._model_file)

  def testPopulateMetadataFileToModelFileWithMetadataAndBufFields(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1, self._file2])
    populator.populate()

    # Tests if the metadata is populated correctly.
    self._assert_golden_metadata(self._model_file)

    recorded_files = populator.get_recorded_associated_file_list()
    self.assertEqual(set(recorded_files), set(self.expected_recorded_files))

    # Up to now, we've proved the correctness of the model buffer that read from
    # file. Then we'll test if get_model_buffer() gives the same model buffer.
    with open(self._model_file, "rb") as f:
      model_buf_from_file = f.read()
    model_buf_from_getter = populator.get_model_buffer()
    self.assertEqual(model_buf_from_file, model_buf_from_getter)

  def testPopulateInvalidMetadataFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(
        self._empty_model_buf)
    with self.assertRaises(IOError) as error:
      populator.load_metadata_file(self._invalid_file)
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def testPopulateInvalidMetadataBuffer(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(
        self._empty_model_buf)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer([])
    self.assertEqual("The metadata to be populated is empty.",
                     str(error.exception))

  def testGetModelBufferBeforePopulatingData(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(
        self._empty_model_buf)
    model_buf = populator.get_model_buffer()
    expected_model_buf = self._empty_model_buf
    self.assertEqual(model_buf, expected_model_buf)


class MetadataDisplayerTest(MetadataTest):

  def setUp(self):
    super(MetadataDisplayerTest, self).setUp()
    self._model_file = self._create_model_with_metadata_and_associated_files()

  def _create_model_with_metadata_and_associated_files(self):
    model_buf = self._create_empty_model_buf()
    model_file = self.create_tempfile().full_path
    with open(model_file, "wb") as f:
      f.write(model_buf)

    populator = _metadata.MetadataPopulator.with_model_file(model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1, self._file2])
    populator.populate()
    return model_file

  def test_load_model_file_invalidModelFile_throwsException(self):
    with self.assertRaises(IOError) as error:
      _metadata.MetadataDisplayer.with_model_file(self._invalid_file)
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def test_load_model_file_modelWithoutMetadata_throwsException(self):
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_file(self._empty_model_file)
    self.assertEqual("The model does not have metadata.", str(error.exception))

  def test_load_model_file_modelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_file(self._model_file)
    self.assertIsInstance(displayer, _metadata.MetadataDisplayer)

  def test_load_model_buffer_modelWithOutMetadata_throwsException(self):
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_buffer(
          self._create_empty_model_buf())
    self.assertEqual("The model does not have metadata.", str(error.exception))

  def test_load_model_buffer_modelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_buffer(
        open(self._model_file, "rb").read())
    self.assertIsInstance(displayer, _metadata.MetadataDisplayer)

  def test_get_metadata_json_modelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_file(self._model_file)
    actual = displayer.get_metadata_json()

    # Verifies the generated json file.
    golden_json_file_path = resource_loader.get_path_to_datafile(
        "testdata/golden_json.json")
    with open(golden_json_file_path, "r") as f:
      expected = f.read()
    self.assertEqual(actual, expected)

  def test_get_packed_associated_file_list_modelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_file(self._model_file)
    packed_files = displayer.get_packed_associated_file_list()

    expected_packed_files = [
        os.path.basename(self._file1),
        os.path.basename(self._file2)
    ]
    self.assertEqual(len(packed_files), 2)
    self.assertEqual(set(packed_files), set(expected_packed_files))


if __name__ == "__main__":
  test.main()
