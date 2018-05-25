"""Methods for loading avro data

## This package provides a dataset class and parser method for avro data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long
from tensorflow.contrib.avro.python.avro_record_dataset import *
from tensorflow.contrib.avro.python.parse_avro_record import *
# pylint: enable=unused-import,line-too-long

from tensorflow.python.util.all_util import remove_undocumented


_allowed_symbols = [
    'AvroRecordDataset',
    'parse_avro_record',
]

remove_undocumented(__name__, _allowed_symbols)
