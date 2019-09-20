# Copyright 2014 Google Inc. All rights reserved.
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

from . import number_types as N
from . import packer
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature

np = import_numpy()

FILE_IDENTIFIER_LENGTH=4

def Get(packer_type, buf, head):
    """ Get decodes a value at buf[head] using `packer_type`. """
    return packer_type.unpack_from(memoryview_type(buf), head)[0]


def GetVectorAsNumpy(numpy_type, buf, count, offset):
    """ GetVecAsNumpy decodes values starting at buf[head] as
    `numpy_type`, where `numpy_type` is a numpy dtype. """
    if np is not None:
        # TODO: could set .flags.writeable = False to make users jump through
        #       hoops before modifying...
        return np.frombuffer(buf, dtype=numpy_type, count=count, offset=offset)
    else:
        raise NumpyRequiredForThisFeature('Numpy was not found.')


def Write(packer_type, buf, head, n):
    """ Write encodes `n` at buf[head] using `packer_type`. """
    packer_type.pack_into(buf, head, n)
