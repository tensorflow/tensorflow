#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("DecodeRaw")
    .Input("bytes: string")
    .Output("output: out_type")
    .Attr("out_type: {float,double,int32,uint8,int16,int8,int64}")
    .Attr("little_endian: bool = true")
    .Doc(R"doc(
Reinterpret the bytes of a string as a vector of numbers.

bytes: All the elements must have the same length.
little_endian: Whether the input bytes are in little-endian order.
  Ignored for out_types that are stored in a single byte like uint8.
output: A Tensor with one more dimension than the input bytes.  The
  added dimension will have size equal to the length of the elements
  of bytes divided by the number of bytes to represent out_type.
)doc");

REGISTER_OP("ParseExample")
    .Input("serialized: string")
    .Input("names: string")
    .Input("sparse_keys: Nsparse * string")
    .Input("dense_keys: Ndense * string")
    .Input("dense_defaults: Tdense")
    .Output("sparse_indices: Nsparse * int64")
    .Output("sparse_values: sparse_types")
    .Output("sparse_shapes: Nsparse * int64")
    .Output("dense_values: Tdense")
    .Attr("Nsparse: int >= 0")  // Inferred from sparse_keys
    .Attr("Ndense: int >= 0")   // Inferred from dense_keys
    .Attr("sparse_types: list({float,int64,string}) >= 0")
    .Attr("Tdense: list({float,int64,string}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .Doc(R"doc(
Transforms a vector of brain.Example protos (as strings) into typed tensors.

serialized: A vector containing a batch of binary serialized Example protos.
names: A vector containing the names of the serialized protos.
  May contain, for example, table key (descriptive) names for the
  corresponding serialized protos.  These are purely useful for debugging
  purposes, and the presence of values here has no effect on the output.
  May also be an empty vector if no names are available.
  If non-empty, this vector must be the same length as "serialized".
dense_keys: A list of Ndense string Tensors (scalars).
  The keys expected in the Examples' features associated with dense values.
dense_defaults: A list of Ndense Tensors (some may be empty).
  dense_defaults[j] provides default values
  when the example's feature_map lacks dense_key[j].  If an empty Tensor is
  provided for dense_defaults[j], then the Feature dense_keys[j] is required.
  The input type is inferred from dense_defaults[j], even when it's empty.
  If dense_defaults[j] is not empty, its shape must match dense_shapes[j].
dense_shapes: A list of Ndense shapes; the shapes of data in each Feature
  given in dense_keys.
  The number of elements in the Feature corresponding to dense_key[j]
  must always equal dense_shapes[j].NumEntries().
  If dense_shapes[j] == (D0, D1, ..., DN) then the the shape of output
  Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
  The dense outputs are just the inputs row-stacked by batch.
sparse_keys: A list of Nsparse string Tensors (scalars).
  The keys expected in the Examples' features associated with sparse values.
sparse_types: A list of Nsparse types; the data types of data in each Feature
  given in sparse_keys.
  Currently the ParseExample supports DT_FLOAT (FloatList),
  DT_INT64 (Int64List), and DT_STRING (BytesList).
)doc");

REGISTER_OP("DecodeCSV")
    .Input("records: string")
    .Input("record_defaults: OUT_TYPE")
    .Output("output: OUT_TYPE")
    .Attr("OUT_TYPE: list({float,int32,int64,string})")
    .Attr("field_delim: string = ','")
    .Doc(R"doc(
Convert CSV records to tensors. Each column maps to one tensor.

RFC 4180 format is expected for the CSV records.
(https://tools.ietf.org/html/rfc4180)
Note that we allow leading and trailing spaces with int or float field.

records: Each string is a record/row in the csv and all records should have
  the same format.
record_defaults: One tensor per column of the input record, with either a
  scalar default value for that column or empty if the column is required.
field_delim: delimiter to separate fields in a record.
output: Each tensor will have the same shape as records.
)doc");

REGISTER_OP("StringToNumber")
    .Input("string_tensor: string")
    .Output("output: out_type")
    .Attr("out_type: {float, int32} = DT_FLOAT")
    .Doc(R"doc(
Converts each string in the input Tensor to the specified numeric type.

(Note that int32 overflow results in an error while float overflow
results in a rounded value.)

out_type: The numeric type to interpret each string in string_tensor as.
output: A Tensor of the same shape as the input string_tensor.
)doc");

}  // namespace tensorflow
