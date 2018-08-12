# Data IO (Python functions)
[TOC]

A TFRecords file represents a sequence of (binary) strings.  The format is not
random access, so it is suitable for streaming large amounts of data but not
suitable if fast sharding or other non-sequential access is desired.

*   `tf.python_io.TFRecordWriter`
*   `tf.python_io.tf_record_iterator`
*   `tf.python_io.TFRecordCompressionType`
*   `tf.python_io.TFRecordOptions`

- - -

## TFRecords Format Details

A TFRecords file contains a sequence of strings with CRC32C (32-bit CRC using
the Castagnoli polynomial) hashes.  Each record has the format

    uint64 length
    uint32 masked_crc32_of_length
    byte   data[length]
    uint32 masked_crc32_of_data

and the records are concatenated together to produce the file. CRCs are
[described here](https://en.wikipedia.org/wiki/Cyclic_redundancy_check), and
the mask of a CRC is

    masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul
