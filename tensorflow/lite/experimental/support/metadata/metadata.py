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
"""TensorFlow Lite metadata tools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import shutil
import tempfile
import warnings
import zipfile

from flatbuffers.python import flatbuffers
from tensorflow.lite.experimental.support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow.lite.experimental.support.metadata import schema_py_generated as _schema_fb
from tensorflow.lite.experimental.support.metadata.flatbuffers_lib import _pywrap_flatbuffers
from tensorflow.python.platform import resource_loader

_FLATC_TFLITE_METADATA_SCHEMA_FILE = resource_loader.get_path_to_datafile(
    "metadata_schema.fbs")


# TODO(b/141467403): add delete method for associated files.
class MetadataPopulator(object):
  """Packs metadata and associated files into TensorFlow Lite model file.

  MetadataPopulator can be used to populate metadata and model associated files
  into a model file or a model buffer (in bytearray). It can also help to
  inspect list of files that have been packed into the model or are supposed to
  be packed into the model.

  The metadata file (or buffer) should be generated based on the metadata
  schema:
  third_party/tensorflow/lite/schema/metadata_schema.fbs

  Example usage:
  Populate matadata and label file into an image classifier model.

  First, based on metadata_schema.fbs, generate the metadata for this image
  classifer model using Flatbuffers API. Attach the label file onto the ouput
  tensor (the tensor of probabilities) in the metadata.

  Then, pack the metadata and lable file into the model as follows.

    ```python
    # Populating a metadata file (or a metadta buffer) and associated files to
    a model file:
    populator = MetadataPopulator.with_model_file(model_file)
    # For metadata buffer (bytearray read from the metadata file), use:
    # populator.load_metadata_buffer(metadata_buf)
    populator.load_metadata_file(metadata_file)
    populator.load_associated_files([label.txt])
    populator.populate()

    # Populating a metadata file (or a metadta buffer) and associated files to
    a model buffer:
    populator = MetadataPopulator.with_model_buffer(model_buf)
    populator.load_metadata_file(metadata_file)
    populator.load_associated_files([label.txt])
    populator.populate()
    # Writing the updated model buffer into a file.
    updated_model_buf = populator.get_model_buffer()
    with open("updated_model.tflite", "wb") as f:
      f.write(updated_model_buf)
    ```
  """
  # As Zip API is used to concatenate associated files after tflite model file,
  # the populating operation is developed based on a model file. For in-memory
  # model buffer, we create a tempfile to serve the populating operation.
  # Creating the deleting such a tempfile is handled by the class,
  # _MetadataPopulatorWithBuffer.

  METADATA_FIELD_NAME = "TFLITE_METADATA"
  TFLITE_FILE_IDENTIFIER = b"TFL3"
  METADATA_FILE_IDENTIFIER = b"M001"

  def __init__(self, model_file):
    """Constructor for MetadataPopulator.

    Args:
      model_file: valid path to a TensorFlow Lite model file.

    Raises:
      IOError: File not found.
      ValueError: the model does not have the expected flatbuffer identifer.
    """
    _assert_model_file_identifier(model_file)
    self._model_file = model_file
    self._metadata_buf = None
    self._associated_files = set()

  @classmethod
  def with_model_file(cls, model_file):
    """Creates a MetadataPopulator object that populates data to a model file.

    Args:
      model_file: valid path to a TensorFlow Lite model file.

    Returns:
      MetadataPopulator object.

    Raises:
      IOError: File not found.
      ValueError: the model does not have the expected flatbuffer identifer.
    """
    return cls(model_file)

  # TODO(b/141468993): investigate if type check can be applied to model_buf for
  # FB.
  @classmethod
  def with_model_buffer(cls, model_buf):
    """Creates a MetadataPopulator object that populates data to a model buffer.

    Args:
      model_buf: TensorFlow Lite model buffer in bytearray.

    Returns:
      A MetadataPopulator(_MetadataPopulatorWithBuffer) object.

    Raises:
      ValueError: the model does not have the expected flatbuffer identifer.
    """
    return _MetadataPopulatorWithBuffer(model_buf)

  def get_model_buffer(self):
    """Gets the buffer of the model with packed metadata and associated files.

    Returns:
      Model buffer (in bytearray).
    """
    with open(self._model_file, "rb") as f:
      return f.read()

  def get_packed_associated_file_list(self):
    """Gets a list of associated files packed to the model file.

    Returns:
      List of packed associated files.
    """
    if not zipfile.is_zipfile(self._model_file):
      return []

    with zipfile.ZipFile(self._model_file, "r") as zf:
      return zf.namelist()

  def get_recorded_associated_file_list(self):
    """Gets a list of associated files recorded in metadata of the model file.

    Associated files may be attached to a model, a subgraph, or an input/output
    tensor.

    Returns:
      List of recorded associated files.
    """
    recorded_files = []

    if not self._metadata_buf:
      return recorded_files

    metadata = _metadata_fb.ModelMetadata.GetRootAsModelMetadata(
        self._metadata_buf, 0)

    # Add associated files attached to ModelMetadata
    self._get_associated_files_from_metadata_struct(metadata, recorded_files)

    # Add associated files attached to each SubgraphMetadata
    for j in range(metadata.SubgraphMetadataLength()):
      subgraph = metadata.SubgraphMetadata(j)
      self._get_associated_files_from_metadata_struct(subgraph, recorded_files)

      # Add associated files attached to each input tensor
      for k in range(subgraph.InputTensorMetadataLength()):
        tensor = subgraph.InputTensorMetadata(k)
        self._get_associated_files_from_metadata_struct(tensor, recorded_files)

      # Add associated files attached to each output tensor
      for k in range(subgraph.OutputTensorMetadataLength()):
        tensor = subgraph.OutputTensorMetadata(k)
        self._get_associated_files_from_metadata_struct(tensor, recorded_files)

    return recorded_files

  def load_associated_files(self, associated_files):
    """Loads associated files that to be concatenated after the model file.

    Args:
      associated_files: list of file paths.

    Raises:
      IOError:
        File not found.
    """
    for af in associated_files:
      _assert_exist(af)
      self._associated_files.add(af)

  def load_metadata_buffer(self, metadata_buf):
    """Loads the metadata buffer (in bytearray) to be populated.

    Args:
      metadata_buf: metadata buffer (in bytearray) to be populated.

    Raises:
      ValueError: The metadata to be populated is empty.
      ValueError: The metadata does not have the expected flatbuffer identifer.
    """
    if not metadata_buf:
      raise ValueError("The metadata to be populated is empty.")

    _assert_metadata_buffer_identifier(metadata_buf)
    self._metadata_buf = metadata_buf

  def load_metadata_file(self, metadata_file):
    """Loads the metadata file to be populated.

    Args:
      metadata_file: path to the metadata file to be populated.

    Raises:
      IOError: File not found.
      ValueError: The metadata does not have the expected flatbuffer identifer.
    """
    _assert_exist(metadata_file)
    with open(metadata_file, "rb") as f:
      metadata_buf = f.read()
    self.load_metadata_buffer(bytearray(metadata_buf))

  def populate(self):
    """Populates loaded metadata and associated files into the model file."""
    self._assert_validate()
    self._populate_metadata_buffer()
    self._populate_associated_files()

  def _assert_validate(self):
    """Validates the metadata and associated files to be populated.

    Raises:
      ValueError:
        File is recorded in the metadata, but is not going to be populated.
        File has already been packed.
    """
    # Gets files that are recorded in metadata.
    recorded_files = self.get_recorded_associated_file_list()

    # Gets files that have been packed to self._model_file.
    packed_files = self.get_packed_associated_file_list()

    # Gets the file name of those associated files to be populated.
    to_be_populated_files = []
    for af in self._associated_files:
      to_be_populated_files.append(os.path.basename(af))

    # Checks all files recorded in the metadata will be populated.
    for rf in recorded_files:
      if rf not in to_be_populated_files and rf not in packed_files:
        raise ValueError("File, '{0}', is recorded in the metadata, but has "
                         "not been loaded into the populator.".format(rf))

    for f in to_be_populated_files:
      if f in packed_files:
        raise ValueError("File, '{0}', has already been packed.".format(f))

      if f not in recorded_files:
        warnings.warn(
            "File, '{0}', does not exsit in the metadata. But packing it to "
            "tflite model is still allowed.".format(f))

  def _copy_archived_files(self, src_zip, dst_zip, file_list):
    """Copy archieved files in file_list from src_zip ro dst_zip."""

    if not zipfile.is_zipfile(src_zip):
      raise ValueError("File, '{0}', is not a zipfile.".format(src_zip))

    with zipfile.ZipFile(src_zip,
                         "r") as src_zf, zipfile.ZipFile(dst_zip,
                                                         "a") as dst_zf:
      src_list = src_zf.namelist()
      for f in file_list:
        if f not in src_list:
          raise ValueError(
              "File, '{0}', does not exist in the zipfile, {1}.".format(
                  f, src_zip))
        file_buffer = src_zf.read(f)
        dst_zf.writestr(f, file_buffer)

  def _get_associated_files_from_metadata_struct(self, file_holder, file_list):
    for j in range(file_holder.AssociatedFilesLength()):
      file_list.append(file_holder.AssociatedFiles(j).Name().decode("utf-8"))

  def _populate_associated_files(self):
    """Concatenates associated files after TensorFlow Lite model file.

    If the MetadataPopulator object is created using the method,
    with_model_file(model_file), the model file will be updated.
    """
    # Opens up the model file in "appending" mode.
    # If self._model_file already has pack files, zipfile will concatenate
    # addition files after self._model_file. For example, suppose we have
    # self._model_file = old_tflite_file | label1.txt | label2.txt
    # Then after trigger populate() to add label3.txt, self._model_file becomes
    # self._model_file = old_tflite_file | label1.txt | label2.txt | label3.txt
    with zipfile.ZipFile(self._model_file, "a") as zf:
      for af in self._associated_files:
        filename = os.path.basename(af)
        zf.write(af, filename)

  def _populate_metadata_buffer(self):
    """Populates the metadata buffer (in bytearray) into the model file.

    Inserts metadata_buf into the metadata field of schema.Model. If the
    MetadataPopulator object is created using the method,
    with_model_file(model_file), the model file will be updated.
    """

    with open(self._model_file, "rb") as f:
      model_buf = f.read()

    model = _schema_fb.ModelT.InitFromObj(
        _schema_fb.Model.GetRootAsModel(model_buf, 0))
    buffer_field = _schema_fb.BufferT()
    buffer_field.data = self._metadata_buf

    is_populated = False
    if not model.metadata:
      model.metadata = []
    else:
      # Check if metadata has already been populated.
      for meta in model.metadata:
        if meta.name.decode("utf-8") == self.METADATA_FIELD_NAME:
          is_populated = True
          model.buffers[meta.buffer] = buffer_field

    if not is_populated:
      if not model.buffers:
        model.buffers = []
      model.buffers.append(buffer_field)
      # Creates a new metadata field.
      metadata_field = _schema_fb.MetadataT()
      metadata_field.name = self.METADATA_FIELD_NAME
      metadata_field.buffer = len(model.buffers) - 1
      model.metadata.append(metadata_field)

    # Packs model back to a flatbuffer binaray file.
    b = flatbuffers.Builder(0)
    b.Finish(model.Pack(b), self.TFLITE_FILE_IDENTIFIER)
    model_buf = b.Output()

    # Saves the updated model buffer to model file.
    # Gets files that have been packed to self._model_file.
    packed_files = self.get_packed_associated_file_list()
    if packed_files:
      # Writes the updated model buffer and associated files into a new model
      # file. Then overwrites the original model file.
      with tempfile.NamedTemporaryFile() as temp:
        new_file = temp.name
      with open(new_file, "wb") as f:
        f.write(model_buf)
      self._copy_archived_files(self._model_file, new_file, packed_files)
      shutil.copy(new_file, self._model_file)
      os.remove(new_file)
    else:
      with open(self._model_file, "wb") as f:
        f.write(model_buf)


class _MetadataPopulatorWithBuffer(MetadataPopulator):
  """Subclass of MetadtaPopulator that populates metadata to a model buffer.

  This class is used to populate metadata into a in-memory model buffer. As we
  use Zip API to concatenate associated files after tflite model file, the
  populating operation is developed based on a model file. For in-memory model
  buffer, we create a tempfile to serve the populating operation. This class is
  then used to generate this tempfile, and delete the file when the
  MetadataPopulator object is deleted.
  """

  def __init__(self, model_buf):
    """Constructor for _MetadataPopulatorWithBuffer.

    Args:
      model_buf: TensorFlow Lite model buffer in bytearray.

    Raises:
      ValueError: model_buf is empty.
      ValueError: model_buf does not have the expected flatbuffer identifer.
    """
    if not model_buf:
      raise ValueError("model_buf cannot be empty.")

    with tempfile.NamedTemporaryFile() as temp:
      model_file = temp.name

    with open(model_file, "wb") as f:
      f.write(model_buf)

    MetadataPopulator.__init__(self, model_file)

  def __del__(self):
    """Destructor of _MetadataPopulatorWithBuffer.

    Deletes the tempfile.
    """
    if os.path.exists(self._model_file):
      os.remove(self._model_file)


class MetadataDisplayer(object):
  """Displays metadata and associated file info in human-readable format."""

  def __init__(self, model_file, metadata_file, associated_file_list):
    """Constructor for MetadataDisplayer.

    Args:
      model_file: valid path to the model file.
      metadata_file: valid path to the metadata file.
      associated_file_list: list of associate files in the model file.
    """
    _assert_model_file_identifier(model_file)
    _assert_metadata_file_identifier(metadata_file)
    self._model_file = model_file
    self._metadata_file = metadata_file
    self._associated_file_list = associated_file_list

  @classmethod
  def with_model_file(cls, model_file):
    """Creates a MetadataDisplayer object for the model file.

    Args:
      model_file: valid path to a TensorFlow Lite model file.

    Returns:
      MetadataDisplayer object.

    Raises:
      IOError: File not found.
      ValueError: The model does not have metadata.
    """
    _assert_exist(model_file)
    metadata_file = cls._save_temporary_metadata_file(model_file)
    associated_file_list = cls._parse_packed_associted_file_list(model_file)
    return cls(model_file, metadata_file, associated_file_list)

  @classmethod
  def with_model_buffer(cls, model_buffer):
    """Creates a MetadataDisplayer object for a file buffer.

    Args:
      model_buffer: TensorFlow Lite model buffer in bytearray.

    Returns:
      MetadataDisplayer object.
    """
    if not model_buffer:
      raise ValueError("model_buffer cannot be empty.")

    with tempfile.NamedTemporaryFile() as temp:
      model_file = temp.name

    with open(model_file, "wb") as f:
      f.write(model_buffer)
    return cls.with_model_file(model_file)

  def get_metadata_json(self):
    """Converts the metadata into a json string."""
    opt = _pywrap_flatbuffers.IDLOptions()
    opt.strict_json = True
    parser = _pywrap_flatbuffers.Parser(opt)
    with open(_FLATC_TFLITE_METADATA_SCHEMA_FILE) as f:
      metadata_schema_content = f.read()
    with open(self._metadata_file, "rb") as f:
      metadata_file_content = f.read()
    if not parser.parse(metadata_schema_content):
      raise ValueError("Cannot parse metadata schema. Reason: " + parser.error)
    with open(self._metadata_file, "rb") as f:
      metadata_file_content = f.read()
    return _pywrap_flatbuffers.generate_text(parser, metadata_file_content)

  def get_packed_associated_file_list(self):
    """Returns a list of associated files that are packed in the model.

    Returns:
      A name list of associated files.
    """
    return copy.deepcopy(self._associated_file_list)

  @staticmethod
  def _save_temporary_metadata_file(model_file):
    """Saves the metadata in the model file to a temporary file.

    Args:
      model_file: valid path to the model file.

    Returns:
      Path to the metadata temporary file.

    Raises:
      ValueError: The model does not have metadata.
    """
    with open(model_file, "rb") as f:
      model_buf = f.read()

    tflite_model = _schema_fb.Model.GetRootAsModel(model_buf, 0)

    # Gets metadata from the model file.
    for i in range(tflite_model.MetadataLength()):
      meta = tflite_model.Metadata(i)
      if meta.Name().decode("utf-8") == MetadataPopulator.METADATA_FIELD_NAME:
        buffer_index = meta.Buffer()
        metadata = tflite_model.Buffers(buffer_index)
        metadata_buf = metadata.DataAsNumpy().tobytes()
        # Creates a temporary file to store the metadata.
        with tempfile.NamedTemporaryFile() as temp:
          metadata_file = temp.name
        # Saves the metadata into the temporary file.
        with open(metadata_file, "wb") as f:
          f.write(metadata_buf)
          return metadata_file

    raise ValueError("The model does not have metadata.")

  @staticmethod
  def _parse_packed_associted_file_list(model_file):
    """Gets a list of associated files packed to the model file.

    Args:
      model_file: valid path to the model file.

    Returns:
      List of packed associated files.
    """
    if not zipfile.is_zipfile(model_file):
      return []

    with zipfile.ZipFile(model_file, "r") as zf:
      return zf.namelist()

  def __del__(self):
    """Destructor of MetadataDisplayer.

    Deletes the tempfile.
    """
    if os.path.exists(self._metadata_file):
      os.remove(self._metadata_file)


def _assert_exist(filename):
  """Checks if a file exists."""
  if not os.path.exists(filename):
    raise IOError("File, '{0}', does not exist.".format(filename))


def _assert_model_file_identifier(model_file):
  """Checks if a model file has the expected TFLite schema identifier."""
  _assert_exist(model_file)
  with open(model_file, "rb") as f:
    model_buf = f.read()

  if not _schema_fb.Model.ModelBufferHasIdentifier(model_buf, 0):
    raise ValueError(
        "The model provided does not have the expected identifier, and "
        "may not be a valid TFLite model.")


def _assert_metadata_file_identifier(metadata_file):
  """Checks if a metadata file has the expected Metadata schema identifier."""
  _assert_exist(metadata_file)
  with open(metadata_file, "rb") as f:
    metadata_buf = f.read()
  _assert_metadata_buffer_identifier(metadata_buf)


def _assert_metadata_buffer_identifier(metadata_buf):
  """Checks if a metadata buffer has the expected Metadata schema identifier."""
  if not _metadata_fb.ModelMetadata.ModelMetadataBufferHasIdentifier(
      metadata_buf, 0):
    raise ValueError(
        "The metadata buffer does not have the expected identifier, and may not"
        " be a valid TFLite Metadata.")
