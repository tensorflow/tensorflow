### `tf.decode_csv(records, record_defaults, field_delim=None, name=None)` {#decode_csv}

Convert CSV records to tensors. Each column maps to one tensor.

RFC 4180 format is expected for the CSV records.
(https://tools.ietf.org/html/rfc4180)
Note that we allow leading and trailing spaces with int or float field.

##### Args:


*  <b>`records`</b>: A `Tensor` of type `string`.
    Each string is a record/row in the csv and all records should have
    the same format.
*  <b>`record_defaults`</b>: A list of `Tensor` objects with types from: `float32`, `int32`, `int64`, `string`.
    One tensor per column of the input record, with either a
    scalar default value for that column or empty if the column is required.
*  <b>`field_delim`</b>: An optional `string`. Defaults to `","`.
    delimiter to separate fields in a record.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A list of `Tensor` objects. Has the same type as `record_defaults`.
  Each tensor will have the same shape as records.

