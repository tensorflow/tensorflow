from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from StringIO import StringIO
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter, BinaryDecoder, BinaryEncoder
from avro.schema import parse
from tensorflow.python.framework import dtypes as tf_types

# Contains a collection of utility functions to compare avro records using the avro package and the avro record dataset


def list_avro_files(local_dir):
    """
    Lists all avro files that end in *.avro from the local directory in lexicographically sorted order.

    :param local_dir: The local diretory.

    :return: A listing of files with absolute path names.
    """
    return sorted([os.path.join(os.path.abspath(local_dir), filename) for filename in os.listdir(local_dir)
                   if filename.endswith('.avro')])


def deserialize(serialized_str, schema_object):
    """
    Deserialize an avro record from a string. This method is capable of schema resolution.

    :param serialized_str: The serialized input string.
    :param schema_object: The schema when this record was written also used for reading.

    :return: The de-serialized record structure in python as map-list object.
    """
    reader = StringIO(serialized_str)
    decoder = BinaryDecoder(reader)
    datum_reader = DatumReader(writers_schema=schema_object, readers_schema=schema_object)
    return datum_reader.read(decoder)


def serialize(record, writers_schema_object):
    """
    Serializes the record according to the schema and returns a string.

    :param record: The record.
    :param writers_schema_object: The writers schema.

    :return: Serialized string for the record.
    """
    string_writer = StringIO()
    encoder = BinaryEncoder(string_writer)
    datum_writer = DatumWriter(writers_schema_object)
    datum_writer.write(record, encoder)
    return string_writer.getvalue()


def read_records_resolved(filenames, schema_resolved):
    """
    Reads records as strings where each row is serialized separately.

    :param filenames: File names for the records.

    :return: An array of serialized string with one string per record.
    """
    records = []
    schema = parse(schema_resolved)
    for filename in filenames:
        with open(filename, 'rb') as file_handle:
            reader = DataFileReader(file_handle, DatumReader(readers_schema=schema))
            records += [record for record in reader]
    return records


def read_schema(filename):
    """
    Reads the schema from a file into json string.

    :param filename: The filename of an avro file.

    :return: json string of the schema.
    """
    with open(filename, 'rb') as file_handle:
        reader = DataFileReader(file_handle, DatumReader())
        return str(reader.datum_reader.writers_schema)


def parse_schema(schema):
    """
    Parse schema into schema object.

    :param schema: The json schema string.

    :return: The avro schema object.
    """
    return parse(schema)


def _get_record_value(record, keys, i_start, dtype):
    """
    This method goes through the keys and pulls fields, array items, or map items. It can also handle the asterisk *
    notation as a wildcard for deferred selection of a field within a record that is contained in an array.

    :param record: A nested map/list structure.
    :param keys: A list of keys.
    :param i_start: The start index into the list of keys.
    :param dtype: TensorFlow's data type.
    :return: The value/values for this list of keys.
    """

    def _is_array(key):
        return len(key) > 2 and key[0] == "[" and key[-1] == "]"

    def _is_array_wildcard(key):
        return key == "[*]"

    def _is_array_index(key):
        # Assumes is array
        return key[1:-1].isdigit() and int(key[1:-1]) >= 0

    def _get_array_index(key):
        return int(key[1:-1])

    def _is_array_filter(key):
        # Assumes is array
        return key.count("=") == 1

    def _get_array_filter(key):
        # Assumes is array
        key_value = key[1:-1].split("=")
        return key_value[0], key_value[1]

    def _is_map(key):
        return len(key) > 4 and key.startswith("['") and key.endswith("']")

    def _is_map_wildcard(key):
        return key == "['*']"

    def _is_map_key(key):
        return len(key) > 4 and key.startswith("['") and key.endswith("']")

    def _get_map_key(key):
        return key[2:-2]

    def _is_record(key):
        return len(key) > 0

    # Go over all keys starting from i_start
    for i_key in range(i_start, len(keys)):
        key = keys[i_key]

        # Check for map first
        if _is_map(key):
            if not isinstance(record, dict):
                raise RuntimeError("For '{}' expected type '{}' but found type '{}'.".format(record, dict,
                                                                                             type(record)))
            if _is_map_wildcard(key):
                return [_get_record_value(r, keys, i_key + 1, dtype) for r in record.values()]
            if _is_map_key(key):
                record = record[_get_map_key(key)]
            else:
                raise KeyError("Unable to parse '{}' into a key or wildcard for a map.".format(key))

        # Check for array second
        elif _is_array(key):
            if not isinstance(record, list):
                raise RuntimeError("For '{}' expected type '{}' but found type '{}'.".format(record, list,
                                                                                             type(record)))
            if _is_array_wildcard(key):
                return [_get_record_value(r, keys, i_key + 1, dtype) for r in record]
            elif _is_array_index(key):
                record = record[_get_array_index(key)]
            elif _is_array_filter(key):
                key, value = _get_array_filter(key)
                return [_get_record_value(r, keys, i_key + 1, dtype) for r in record if r[key] == value]
            else:
                raise IndexError("Unable to parse '{}' into index or wildcard for an array.".format(key))

        # Check for record
        elif _is_record(key):
            # Records are represented as dict
            if not isinstance(record, dict):
                raise RuntimeError("For '{}' expected type '{}' but found type '{}'.".format(record, dict,
                                                                                             type(record)))
            record = record[key]

        else:
            raise KeyError("Could not parse key '{}' in '{}'.".format(key, keys))

    if record is None:
        # These are the default mappings for null types. Notice that this is a dangerous assumption but not all
        # TensorFlow python classes for arrays support default values. So, we are stuck with this for now.
        if dtype == tf_types.string:
            record = 'null'
        elif dtype == tf_types.int64:
            record = 0L
        elif dtype == tf_types.bool:
            record = False
        elif dtype == tf_types.float32 or dtype == tf_types.float64 or dtype == tf_types.double \
                or dtype == tf_types.half or dtype == tf_types.float16:
            record = 0.0
        else:
            record = 0
    return record


def get_record_value(record, key_str, dtype):
    """
    Parses the key_str and pulls records according to the selection in the key string.

    :param record: A nested map/list structure.
    :param key_str: A key_str with nesting indicated by the separator '/'.
    :param dtype: TensorFlow's data type.

    :return: The nested sub-record or value for the key_str.
    """
    # Split the key_str at '/' and start parsing the keys from index 0
    return _get_record_value(record, key_str.split('/'), 0, dtype)


def write_records_to_file(records, filename, writers_schema, codec='deflate'):
    """
    Writes the string data into an avro encoded file.

    :param records: Records to write.
    :param filename: Filename for the file to be written.
    :param writers_schema: The schema used when writing the records.
    :param codec: Compression codec used to write the avro file.
    """
    schema = parse(writers_schema)
    with open(filename, 'wb') as out:
        writer = DataFileWriter(out, DatumWriter(), writers_schema=schema, codec=codec)
        for record in records:
            writer.append(record)
        writer.close()
