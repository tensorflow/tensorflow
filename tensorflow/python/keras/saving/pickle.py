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
"""Support for Python's Pickle protocol by implementing
packing/unpacking of Keras Models into objects which are themselves
picklable.
"""

import os
import tarfile

from io import BytesIO
from uuid import uuid4

import numpy as np

from tensorflow.python.platform import gfile
from tensorflow.python.keras.saving.save import load_model



def unpack_model(packed_keras_model):
  """Reconstruct a Model from the result of pack_model.

  Args:
      packed_keras_model (np.array): return value from pack_model.

  Returns:
      tf.keras.Model: a Keras Model instance.
  """
  temp_dir = f"ram://{uuid4()}"
  b = BytesIO(packed_keras_model)
  with tarfile.open(fileobj=b, mode="r") as archive:
    for fname in archive.getnames():
      dest_path = os.path.join(temp_dir, fname)
      gfile.makedirs(os.path.dirname(dest_path))
      with gfile.GFile(dest_path, "wb") as f:
        f.write(archive.extractfile(fname).read())
  model = load_model(temp_dir)
  gfile.rmtree(temp_dir)
  return model


def pack_model(model):
  """Pack a Keras Model into a numpy array of bytes.

  Args:
      model (tf.keras.Model): a Keras Model instance.

  Returns:
      tuple: an unpacking function (unpack_model) and it's arguments,
      as per Python's pickle protocol.
  """
  temp_dir = f"ram://{uuid4()}"
  model.save(temp_dir)
  b = BytesIO()
  with tarfile.open(fileobj=b, mode="w") as archive:
    for root, _, filenames in gfile.walk(temp_dir):
      for filename in filenames:
        dest_path = os.path.join(root, filename)
        with gfile.GFile(dest_path, "rb") as f:
          info = tarfile.TarInfo(name=os.path.relpath(dest_path, temp_dir))
          info.size = f.size()
          archive.addfile(tarinfo=info, fileobj=f)
  gfile.rmtree(temp_dir)
  b.seek(0)
  return unpack_model, (np.asarray(memoryview(b.read())), )
