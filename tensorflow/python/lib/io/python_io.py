# Copyright 2015 Google Inc. All Rights Reserved.
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

"""## Data IO (Python Functions)

A TFRecords file represents a sequence of (binary) strings.  The format is not
random access, so it is suitable for streaming large amounts of data but not
suitable if fast sharding or other non-sequential access is desired.

@@TFRecordWriter
@@tf_record_iterator

- - -

### TFRecords Format Details

A TFRecords file contains a sequence of strings with CRC hashes.  Each record
has the format

    uint64 length
    uint32 masked_crc32_of_length
    byte   data[length]
    uint32 masked_crc32_of_data

and the records are concatenated together to produce the file.  The CRC32s
are [described here](https://en.wikipedia.org/wiki/Cyclic_redundancy_check),
and the mask of a CRC is

    masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.lib.io.tf_record import *
