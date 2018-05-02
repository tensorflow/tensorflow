#include <avro.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;
using ::tensorflow::shape_inference::ShapeHandle;

// As boiler plate for the class I used tensorflow/core/util/example_proto_helper.h and therein
// "ParseSingleExampleAttrs".

// Checks for valid type for the avro attributes; currently we support bool, int, long, float, double, string.
//
// 'dtype' The data type.
//
// returns OK if any of the supported types; otherwise false.
//
tensorflow::Status CheckValidType(const tensorflow::DataType& dtype);

// Check that all dense shapes are defined. Here, 'defined' means that:
// * All shapes have at least one dimension.
// * A shape can have an undefined dimension -1, as first dimension.
//
// 'dense_shape' The dense shapes.
//
// returns OK if the shapes are defined; otherwise false.
//
tensorflow::Status CheckDenseShapeToBeDefined(const std::vector<tensorflow::PartialTensorShape>& dense_shapes);


// Struct that holds information about dense tensors that is used during parsing.
struct DenseInformation {
  tensorflow::DataType type; // Type
  tensorflow::PartialTensorShape shape; // Shape
  bool variable_length; // This dense tensor has a variable length in the 2nd dimension
  std::size_t elements_per_stride; // Number of elements per stride
};

// This class holds the attributes passed into the parse avro record function.
// In addition, it builds up information about the 'elements per stride', 'variable length' for dense tensors, and
// 'dense shape' information.
class ParseAvroAttrs {
 public:
  // Initializes the attribute information
  template <typename ContextType>
  tensorflow::Status Init(ContextType* ctx) {
    std::vector<tensorflow::DataType> dense_types;
    std::vector<tensorflow::PartialTensorShape> dense_shapes;

    TF_RETURN_IF_ERROR(ctx->GetAttr("Nsparse", &num_sparse));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Ndense", &num_dense));
    TF_RETURN_IF_ERROR(ctx->GetAttr("sparse_types", &sparse_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Tdense", &dense_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("dense_shapes", &dense_shapes));

    // Check that all dense shapes are defined
    TF_RETURN_IF_ERROR(CheckDenseShapeToBeDefined(dense_shapes));

    for (int i_dense = 0; i_dense < dense_shapes.size(); ++i_dense) {
      DenseInformation dense_info;
      tensorflow::TensorShape dense_shape;
      // This is the case where we have a fixed len sequence feature, and the 1st dimension is undefined.
      if (dense_shapes[i_dense].dims() > 0 && dense_shapes[i_dense].dim_size(0) == -1) {
        dense_info.variable_length = true;
        for (int d = 1; d < dense_shapes[i_dense].dims(); ++d) {
          dense_shape.AddDim(dense_shapes[i_dense].dim_size(d));
        }
      // This is the case where all dimensions are defined.
      } else {
        dense_info.variable_length = false;
        dense_shapes[i_dense].AsTensorShape(&dense_shape);
      }
      // Fill in the remaining information into the dense info and add it to to the vector
      dense_info.elements_per_stride = dense_shape.num_elements();
      dense_info.shape = dense_shapes[i_dense];
      dense_info.type = dense_types[i_dense];
      dense_infos.push_back(dense_info);
    }
    return FinishInit();
  }

  // All these attributes are publicly accessible, hence we did not suffix them with '_'.
  tensorflow::int64 num_sparse; // Number of sparse features
  tensorflow::int64 num_dense; // Number of dense features (fixed and variable length)
  std::vector<tensorflow::DataType> sparse_types; // Types for sparse features
  std::vector<DenseInformation> dense_infos; // Information about each dense tensor
 private:
  tensorflow::Status FinishInit();  // for context-independent parts of Init.
};

// As boiler plate I used tensorflow/core/util/example_proto_helper.cc and therein "ParseSingleExampleAttrs" and
Status CheckValidType(const DataType& dtype) {
  switch (dtype) {
    case DT_BOOL:
    case DT_INT32:
    case DT_INT64:
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_STRING:
      return Status::OK();
    default:
      return errors::InvalidArgument("Received input dtype: ", DataTypeString(dtype));
  }
}

Status CheckDenseShapeToBeDefined(const std::vector<PartialTensorShape>& dense_shapes) {
  for (int i = 0; i < dense_shapes.size(); ++i) {
    const PartialTensorShape& dense_shape = dense_shapes[i];
    bool shape_ok = true;
    if (dense_shape.dims() == -1) {
      shape_ok = false;
    } else {
      for (int d = 1; d < dense_shape.dims() && shape_ok; ++d) {
        if (dense_shape.dim_size(d) == -1) {
          shape_ok = false;
        }
      }
    }
    if (!shape_ok) {
      return errors::InvalidArgument("dense_shapes[", i, "] has unknown rank or unknown inner dimensions: ",
                                     dense_shape.DebugString());
    }
  }
  return Status::OK();
}

// Finishes the initialization for the attributes, which essentially checks that the attributes have the correct values.
//
// returns OK if all attributes are valid; otherwise false.
Status ParseAvroAttrs::FinishInit() {
  if (static_cast<size_t>(num_sparse) != sparse_types.size()) {
    return errors::InvalidArgument("len(sparse_keys) != len(sparse_types)");
  }
  if (static_cast<size_t>(num_dense) != dense_infos.size()) {
    return errors::InvalidArgument("len(dense_keys) != len(dense_infos)");
  }
  if (num_dense > std::numeric_limits<int32>::max()) {
    return errors::InvalidArgument("num_dense_ too large");
  }
  for (const DenseInformation& dense_info : dense_infos) {
    TF_RETURN_IF_ERROR(CheckValidType(dense_info.type));
  }
  for (const DataType& type : sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  return Status::OK();
}


// Register the parse function when building the shared library
// For the op I used as boiler plate: tensorflow/core/ops/parsing_ops.cc and there 'ParseExample'
// For the op kernel I used as boiler plate: tensorflow/core/kernels/example_parsing_ops.cc and there 'ExampleParserOp'
// For the compute method I used as boiler plate: tensorflow/core/util/example_proto_fast_parsing.cc and there the
//   method 'FastParseExample'

REGISTER_OP("ParseAvroRecord")
    .Input("serialized: string")
    .Input("sparse_keys: Nsparse * string")
    .Input("dense_keys: Ndense * string")
    .Input("dense_defaults: Tdense")
    .Output("sparse_indices: Nsparse * int64")
    .Output("sparse_values: sparse_types")
    .Output("sparse_shapes: Nsparse * int64")
    .Output("dense_values: Tdense")
    .Attr("Nsparse: int >= 0")  // Inferred from sparse_keys
    .Attr("Ndense: int >= 0")   // Inferred from dense_keys
    .Attr("sparse_types: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("Tdense: list({float,double,int64,int32,string,bool}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .Attr("schema: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

      ParseAvroAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c));

      // Get the batch size and load it into input
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));

      string schema;
      avro_schema_t reader_schema;

      // Parse schema
      TF_RETURN_IF_ERROR(c->GetAttr("schema", &schema));
      TF_RETURN_IF_ERROR(avro_schema_from_json_length(schema.c_str(), schema.length(), &reader_schema) == 0 ?
        Status::OK() : errors::InvalidArgument("The provided json schema is invalid. ", avro_strerror()));

      int output_idx = 0;

      // Output sparse_indices, sparse_values, sparse_shapes
      for (int i_sparse = 0; i_sparse < attrs.num_sparse; ++i_sparse) {
        c->set_output(output_idx++, c->Matrix(c->UnknownDim(), 2));
      }
      for (int i_sparse = 0; i_sparse < attrs.num_sparse; ++i_sparse) {
        c->set_output(output_idx++, c->Vector(c->UnknownDim()));
      }
      for (int i_sparse = 0; i_sparse < attrs.num_sparse; ++i_sparse) {
        c->set_output(output_idx++, c->Vector(2));
      }

      // Output dense_values
      for (int i_dense = 0; i_dense < attrs.num_dense; ++i_dense) {
        ShapeHandle dense;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(attrs.dense_infos[i_dense].shape, &dense));
        TF_RETURN_IF_ERROR(c->Concatenate(input, dense, &dense));
        c->set_output(output_idx++, dense);
      }

      // Cleanup of the reader schema
      avro_schema_decref(reader_schema);

      // All ok
      return Status::OK();

    }).Doc(R"doc(
      Parses a serialized avro record that follows the supplied schema into typed tensors.
      serialized: A vector containing a batch of binary serialized avro records.
      dense_keys: A list of n_dense string Tensors (scalars).
        The keys expected are associated with dense values.
      dense_defaults: A list of Ndense Tensors (some may be empty).
        dense_defaults[j] provides default values
        when the example's feature_map lacks dense_key[j].  If an empty Tensor is
        provided for dense_defaults[j], then the Feature dense_keys[j] is required.
        The input type is inferred from dense_defaults[j], even when it's empty.
        If dense_defaults[j] is not empty, and dense_shapes[j] is fully defined,
        then the shape of dense_defaults[j] must match that of dense_shapes[j].
        If dense_shapes[j] has an undefined major dimension (variable strides dense
        feature), dense_defaults[j] must contain a single element:
        the padding element.
      dense_shapes: A list of Ndense shapes; the shapes of data in each Feature
        given in dense_keys.
        The number of elements in the Feature corresponding to dense_key[j]
        must always equal dense_shapes[j].NumEntries().
        If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
        Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
        The dense outputs are just the inputs row-stacked by batch.
        This works for dense_shapes[j] = (-1, D1, ..., DN).  In this case
        the shape of the output Tensor dense_values[j] will be
        (|serialized|, M, D1, .., DN), where M is the maximum number of blocks
        of elements of length D1 * .... * DN, across all minibatch entries
        in the input.  Any minibatch entry with less than M blocks of elements of
        length D1 * ... * DN will be padded with the corresponding default_value
        scalar element along the second dimension.
      dense_types: A list of n_dense types; the type of data in each Feature given in dense_keys.
      sparse_keys: A list of n_sparse string Tensors (scalars).
        The keys expected are associated with sparse values.
      sparse_types: A list of n_sparse types; the data types of data in each Feature
        given in sparse_keys.
      schema: A string that describes the avro schema of the underlying serialized avro string.
        Currently the parse function supports the primitive types DT_STRING, DT_DOUBLE, DT_FLOAT,
        DT_INT64, DT_INT32, and DT_BOOL.
    )doc");

template <typename T>
using SmallVector = gtl::InlinedVector<T, 4>; // Up to 4 items are stored without allocating heap memory

// Splits a string into tokens along the separator.
// This function is based on: http://stackoverflow.com/questions/236129/split-a-string-in-c.
//
// 'str' is the string that we split.
//
// 'spe' is the separator used for the split.
//
// returns A vector of strings.
//
std::vector<string> stringSplit(const string& str, char sep) {
  std::vector<string> tokens;
  size_t start = 0, end = 0;
  while ((end = str.find(sep, start)) != string::npos) {
    tokens.push_back(str.substr(start, end - start));
    start = end + 1;
  }
  tokens.push_back(str.substr(start));
  return tokens;
}

// Parses the str into an positive integer number. Does not support '-'.
//
// 'str' is the string that represents a positive integer, e.g. '32482'.
//
// returns True if the string can be parsed into a positive integer, otherwise false.
//
bool isNonNegativeInt(const string& str) {
    return !str.empty() && std::find_if(str.begin(), str.end(), [](char c) { return !std::isdigit(c); }) == str.end();
}

// We use this Avro field representation when parsing user-defined strings to access types fields, maps, arrays in avro.
// An Avro field can be a name, index, key, or asterisk which is used as wildcard when parsing arrays.
// The base class AvroField is an abstract class that defines the avro field types and general methods.
class AvroField {
public:
  enum Type { name, index, key, mapAsterisk, arrayAsterisk, arrayFilter }; // Types for an avro field

  // Get the type for an Avro field.
  //
  // returns The avro type for this field.
  //
  virtual Type getType() const = 0;

  // Get a human-readable string representation of this Avro field.
  //
  // returns The string for this field.
  //
  virtual string toString() const = 0;
};

// Represents a field name which is used to access records' fields.
class AvroFieldName : public AvroField {
public:
  // Create an Avro field name.
  //
  // 'name' The name for this Avro field.
  //
  // returns An instance of 'AvroFieldName'.
  //
  AvroFieldName(const string& name) : name(name) {}

  // Get the name for the Avro field.
  //
  // returns The name.
  //
  inline string getName() const { return name; }

  // Get the type, see super-class.
  AvroField::Type getType() const { return AvroField::Type::name; }

  // Get the string representation, see super-class
  string toString() const { return getName(); }
private:
  string name; // A string to hold the name
};

// Represents an index which is used to access elements in arrays.
class AvroFieldIndex : public AvroField {
public:
  // Create an Avro index.
  //
  // 'index' The index for this Avro field.
  //
  // returns An instance of 'AvroFieldIndex'.
  //
  AvroFieldIndex(int index) : index(index) {}

  // Get the index of this Avro field.
  //
  // returns The index.
  //
  inline int getIndex() const { return index; }

  // Get the type, see super-class.
  AvroField::Type getType() const { return AvroField::Type::index; }

  // Get the string representation, see super-class.
  string toString() const { return std::to_string(getIndex()); }
private:
  int index; // index to hold
};

// Represents a key of a map. Side note: We could have used the same type for a name field in a record and a key in
// a map since avro's low level API treats these cases the same way. However, we choose this cleaner design.
class AvroFieldKey : public AvroField {
public:
  // Create an Avro field key.
  //
  // 'key' The key for this Avro field.
  //
  // returns An instance of 'AvroFieldKey'.
  AvroFieldKey(const string& key) : key(key) {}

  // Get the key string for this Avro field.
  //
  // returns The key.
  inline string getKey() const { return key; }

  // Get the type, see super-class.
  AvroField::Type getType() const { return AvroField::Type::key; }

  // Get the string representation, see super-class.
  string toString() const { return getKey(); }
private:
  string key; // the key string
};

// Used to select all values in a map.
class AvroFieldMapAsterisk : public AvroField {
public:
  // Creates the Avro field map asterisk.
  //
  // returns An instance of 'AvroFieldMapAsterisk'.
  AvroFieldMapAsterisk() {}

  // Get the type, see super-class.
  AvroField::Type getType() const { return AvroField::Type::mapAsterisk; }

  // Get the string representation, see super-class.
  string toString() const { return "'*'"; }
};

// Used to select all items in an array.
class AvroFieldArrayAsterisk : public AvroField {
public:
  // Creates the Avro field asterisk.
  //
  // returns An instance of 'AvroFieldAsterisk'.
  AvroFieldArrayAsterisk() {}

  // Get the type, see super-class.
  AvroField::Type getType() const { return AvroField::Type::arrayAsterisk; }

  // Get the string representation, see super-class.
  string toString() const { return "*"; }
};

// Filters are represented through a key and value pair. The key indicates which field the
// filter will be applied and the value is the criterion for the filter. Only supported for arrays.
class AvroFieldArrayFilter : public AvroField {
public:
  // Creates the Avro field filter.
  //
  // returns An instance of 'AvroFieldFilter'.
  AvroFieldArrayFilter(const string& key, const string& value) : key(key), value(value) {}

  // Get the type, see super-class.
  AvroField::Type getType() const { return AvroField::Type::arrayFilter; }

  // Get the key of this filter.
  inline string getKey() const { return key; }

  // Get the value of this filter.
  inline string getValue() const { return value; }

  // Get the string representation, see super-class.
  string toString() const { return getKey() + "=" + getValue(); }
private:
  string key; // the key of this filter
  string value; // the value of this filter
};


// The sparse buffer holds a list with primitive data types. This is used when parsing all tensors.
struct SparseBuffer {
  // Only the list that corresponds to the data type of the tensor is used
  SmallVector<string> string_list;
  SmallVector<double> double_list;
  SmallVector<float> float_list;
  SmallVector<int64> int64_list;
  SmallVector<int32> int32_list;
  SmallVector<bool> bool_list; // TODO: Change to util::bitmap::InlinedBitVector<NBITS>
  std::vector<size_t> end_indices; // End indices per row in the batch
  size_t n_elements; // The total number of elements in the batch; required by 'SetValues' to accumulate.
};

// Template specializations for 'GetListFromBuffer' for the supported types.
template <typename T>
const SmallVector<T>& GetListFromBuffer(const SparseBuffer& buffer);

template <>
const SmallVector<int64>& GetListFromBuffer<int64>(const SparseBuffer& buffer) {
  return buffer.int64_list;
}
template <>
const SmallVector<int32>& GetListFromBuffer<int32>(const SparseBuffer& buffer) {
  return buffer.int32_list;
}
template <>
const SmallVector<float>& GetListFromBuffer<float>(const SparseBuffer& buffer) {
  return buffer.float_list;
}
template <>
const SmallVector<double>& GetListFromBuffer<double>(const SparseBuffer& buffer) {
  return buffer.double_list;
}
template <>
const SmallVector<bool>& GetListFromBuffer<bool>(const SparseBuffer& buffer) {
  return buffer.bool_list;
}
template <>
const SmallVector<string>& GetListFromBuffer<string>(const SparseBuffer& buffer) {
  return buffer.string_list;
}

// Template specialization for 'CopyOrMoveBlock'; Note: 'string' values are moved, others are copied.
template <typename InputIterT, typename OutputIterT>
void CopyOrMoveBlock(const InputIterT b, const InputIterT e, OutputIterT t) {
  std::copy(b, e, t);
}

template <>
void CopyOrMoveBlock(const string* b, const string* e, string* t) {
  std::move(b, e, t);
}

// Checks that default values are available if required for filling in of values.
//
// This method is used to check that fixed len sequence features have the correct default. If any of the rows has less
// than 'n_elements_per_batch' values, we check that these can be filled in from the 'default_value' tensor.
//
// 'key' Name of the element that is parsed.
//
// 'n_elements_per_batch'  The number of elements in a batch.
//
// 'end_indices'  The end indices of the dense tensor as it is.
//
// 'default_value'  Tensor with default values. If we need to fill in this tensor must have at least
//                  'n_elements_per_batch' many elements.
//
// returns OK if we can fill in elements or no elements need to be filled in; otherwise false.
//
Status CheckDefaultsAvailable(const string& key, const size_t n_elements_per_batch,
                              const std::vector<size_t>& end_indices, const Tensor& default_value) {
  const size_t n_batches = end_indices.size();
  const size_t n_total_elements_be = n_batches * n_elements_per_batch;
  size_t n_total_elements_is = 0;
  const size_t n_default_elements = default_value.NumElements();
  const size_t n_elems_be = n_elements_per_batch;  // per row
  const bool not_enough_defaults = n_default_elements < n_elements_per_batch;
  for (size_t i_batches = 0; i_batches < n_batches; ++i_batches) {
      const size_t n_elems_is = end_indices[i_batches] - n_total_elements_is;  // per row
      n_total_elements_is = end_indices[i_batches];
      const size_t n_fill = n_elems_be - n_elems_is;
      if (n_fill > 0 && not_enough_defaults) {
        return errors::InvalidArgument("For key '", key, "' in batch ", i_batches, " found ", n_elems_is, " elements ",
                                       "but for fixed length need ", n_elems_be, " elements but default provides only ",
                                       n_default_elements, " elements.");
      }
  }
  return Status::OK();
}

// Checks that a default value is supplied.
//
// This method is used by the fixed len sequence feature.
//
// 'key' Name of the element that is parsed.
//
// 'default_value'  Tensor with default value. If we need to fill in this tensor must have at least 1 element.
//
// returns OK we have at least one element; otherwise false.
//
Status CheckDefaultAvailable(const string& key, const Tensor& default_value) {
  const bool no_default = default_value.NumElements() <= 0;
  if (no_default) {
    return errors::InvalidArgument("For key '", key, "' no default is set in ", default_value.DebugString());
  }
  return Status::OK();
}

// Fills in defaults from values in the 'default_tensor'. Note, if there is nothing to fill in the method leaves
// 'values' unchanged.
//
// 'n_elements' The total number of elements that shall be.
//
// 'n_elements_per_batch' The number of elements per batch that shall be.
//
// 'end_indices'  The end indices per batch that are.
//
// 'values' The result tensor.
//
template<typename T>
void FillInFromValues(const size_t n_elements, const size_t n_elements_per_batch,
    const Tensor& default_value, const std::vector<size_t>& end_indices, Tensor* values) {
    auto tensor_data_ptr = values->flat<T>().data();
    const size_t n_batches = end_indices.size();
    auto list_ptr = default_value.flat<T>().data();
    size_t n_total_elements = 0;
    for (size_t i_batches = 0; i_batches < n_batches; ++i_batches) {
      const size_t n_elems = end_indices[i_batches] - n_total_elements;
      CopyOrMoveBlock(list_ptr+n_elems, list_ptr+n_elements_per_batch, tensor_data_ptr+n_elems);
      tensor_data_ptr += n_elements_per_batch;
      n_total_elements = end_indices[i_batches];
    }
}

// Copy a variable length dense tensor from the 'buffer' into 'values' using the 'end_indices' to identify the blocks
// that shall be copied per batch.
//
// 'n_elements' The overall number of elements.
//
// 'n_elements_per_batch' The number of elements in a batch.  Note, that the 'buffer' may not have that many elements
//                        per batch and we have separate methods to fill these with defaults.
//
// 'buffer' The buffer that contains the 'end_indices' and a flattened list of all values for one batch.
//
// 'values' The result tensor.
//
template <typename T>
void CopyVarLen(const size_t n_elements, const size_t n_elements_per_batch,
    const SparseBuffer& buffer, Tensor* values) {

  // Data is [batch_size, max_num_elements, data_stride_size]
  //   and num_elements_per_minibatch = max_num_elements * data_stride_size
  auto tensor_data_ptr = values->flat<T>().data();

  // Number of examples being stored in this buffer
  const auto& end_indices = buffer.end_indices;
  const size_t n_batches = end_indices.size();

  const auto& list = GetListFromBuffer<T>(buffer);
  auto list_ptr = list.begin();

  #ifdef DEBUG_LOG_ENABLED
    auto list_ptr_ = list.begin();
    for (size_t i_elem = 0; i_elem < buffer.n_elements; ++i_elem) {
      LOG(INFO) << *list_ptr_ << ", ";
      list_ptr_++;
    }
  #endif

  size_t n_total_elements = 0;
  // Iterate through all the examples stored in this buffer.
  for (size_t i_batches = 0; i_batches < n_batches; ++i_batches) {
    // Number of elements stored for this example.
    const size_t n_elems = end_indices[i_batches] - n_total_elements;
    CopyOrMoveBlock(list_ptr, list_ptr + n_elems, tensor_data_ptr);
    // Move forward this many elements in the varlen buffer.
    list_ptr += n_elems;
    // Move forward to the next batch entry in the values output.
    tensor_data_ptr += n_elements_per_batch;
    n_total_elements = end_indices[i_batches];
  }

  #ifdef DEBUG_LOG_ENABLED
    for (size_t i_batches = 0; i_batches < n_batches; ++i_batches) {
      LOG(INFO) << "End [" << i_batches << "] = " << end_indices[i_batches];
    }
    LOG(INFO) << "Total " << n_total_elements << " elements; counted " << list.size() << " elements.";
    LOG(INFO) << "Number of " << n_elements_per_batch << " per batch.";
  #endif

  DCHECK(n_total_elements == list.size());
}

// Fills in from a scalar in 'default_value' and copies from 'buffer' into 'values'.
//
// Pre-fills all values with 'default_value' and then copies in the supplied values from the 'buffer'.
// This code is mostly borrowed from 'tensorflow/core/util/example_proto_fast_parsing.cc' and therein the method
// 'FillInFixedLen'.
//
// 'n_elements' The total number of elements.
//
// 'n_elements_per_batch' The number of elements in a batch.
//
// 'buffer' The buffer with the parsed elements.
//
// 'default_value' The tensor with the scalar default value, which we assume exists here.
//
// 'values' The return tensors.
//
template <typename T>
void FillFromScalarAndCopy(const size_t n_elements, const size_t n_elements_per_batch,
    const SparseBuffer& buffer, const Tensor& default_value, Tensor* values) {

  // Fill tensor with default to create padding
  std::fill(values->flat<T>().data(), values->flat<T>().data() + n_elements, default_value.flat<T>()(0));

  CopyVarLen<T>(n_elements, n_elements_per_batch, buffer, values);
}


// Defines a TensorFlow operator that parses an Avro string into TensorFlow native tensors.
// It uses avro c and proto (because of TensorFlow's design).
//
// When developing this parse function I took inspiration from: tensorflow/core/util/example_proto_fast_parsing.cc
// TODO: Run valgrind on this to check for memory leaks
// TODO: Support for null union. General support for unions is not possible in c++
class ParseAvroRecordOp : public OpKernel {
public:
  // Constructs the parse op. This function does not include the parsing of the strings into AvroTypes because these
  // strings are considered inputs and might change with each call of the 'Compute' function. So, no caching is possible
  // for these. This follows the design in tensorflow/core/util/example_proto_fast_parsing.cc.
  //
  // 'ctx' The context to the TensorFlow environment that helps to create tensors and error messages.
  //
  // returns An instance of 'ParseAvroRecordOp'.
  //
  explicit ParseAvroRecordOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string schema;

    // Clear error message for avro
    avro_set_error("");

    // Get the schema supplied by the user as string and parse it
    OP_REQUIRES_OK(ctx, ctx->GetAttr("schema", &schema));
    OP_REQUIRES(ctx, avro_schema_from_json_length(schema.c_str(), schema.length(), &reader_schema_) == 0,
                     errors::InvalidArgument("The provided json schema is invalid. ", avro_strerror()));

    // Get a generic Avro class and instance of that class
    p_iface_ = avro_generic_class_from_schema(reader_schema_);
    OP_REQUIRES(ctx, p_iface_ != nullptr,
                errors::InvalidArgument("Unable to create class for user-supplied schema. ", avro_strerror()));

    // Get attributes
    OP_REQUIRES_OK(ctx, attrs_.Init(ctx));

    // Build the compatibility matrix between Avro types and TensorFlow's types
    BuildCompatibilityMatrix();
  }

  // Destructor used for clean-up of avro structures.
  virtual ~ParseAvroRecordOp() {
    avro_schema_decref(reader_schema_);
    avro_value_iface_decref(p_iface_);
  }

  // The compute function parses the user-provided strings to pull data from the avro serialized string.
  //
  // 'ctx' The context for this operator which gives access to the inputs and outputs.
  //
  void Compute(OpKernelContext* ctx) override {
    const Tensor* serialized;
    OpInputList dense_keys;
    OpInputList sparse_keys;
    OpInputList dense_defaults;

    // Get input arguments
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    OP_REQUIRES_OK(ctx, ctx->input_list("dense_keys", &dense_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("sparse_keys", &sparse_keys));
    OP_REQUIRES_OK(ctx, ctx->input_list("dense_defaults", &dense_defaults));

    CHECK_EQ(dense_keys.size(), attrs_.num_dense);
    CHECK_EQ(sparse_keys.size(), attrs_.num_sparse);
    int64 n_serialized = serialized->NumElements();

    // Create and initialize buffers
    std::vector<SparseBuffer> sparse_buffers(attrs_.num_sparse);
    // TODO: Optimize by using a fixed allocation for dense tensors for fixed len features
    std::vector<SparseBuffer> dense_buffers(attrs_.num_dense);
    for (int64 i_sparse = 0; i_sparse < attrs_.num_sparse; ++i_sparse) {
      sparse_buffers[i_sparse].n_elements = 0;
    }
    for (int64 i_dense = 0; i_dense < attrs_.num_dense; ++i_dense) {
      dense_buffers[i_dense].n_elements = 0;
    }

    std::vector<Tensor> dense_values(attrs_.num_dense);
    std::vector<Tensor> sparse_values(attrs_.num_sparse);
    std::vector<Tensor> sparse_indices(attrs_.num_sparse);
    std::vector<Tensor> sparse_shapes(attrs_.num_sparse);

    // Ensure serialized is a vector
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(serialized->shape()),
                errors::InvalidArgument(
                    "Expected serialized to be a vector, got shape: ",
                    serialized->shape().DebugString()));
    // Ensure the number of defaults matches
    OP_REQUIRES(ctx, dense_defaults.size() == attrs_.num_dense,
                errors::InvalidArgument(
                    "Expected len(dense_defaults) == len(dense_keys) but got: ",
                    dense_defaults.size(), " vs. ", attrs_.num_dense));

    // Check that the defaults are set correctly
    for (int i_dense = 0; i_dense < static_cast<int>(attrs_.num_dense); ++i_dense) {

      // Get the default value
      const Tensor& def_value = dense_defaults[i_dense];

      // Get the information about the dense tensor
      const DenseInformation& dense_info = attrs_.dense_infos[i_dense];

      if (dense_info.variable_length) {
        OP_REQUIRES(ctx, def_value.NumElements() == 1,
                    errors::InvalidArgument(
                        "dense_shape[", i_dense, "] is a variable length shape: ",
                        dense_info.shape.DebugString(),
                        ", therefore "
                        "def_value[",
                        i_dense,
                        "] must contain a single element ("
                        "the padding element).  But its shape is: ",
                        def_value.shape().DebugString()));
      } else if (def_value.NumElements() > 0) {
        OP_REQUIRES(ctx,
                    dense_info.shape.IsCompatibleWith(def_value.shape()),
                    errors::InvalidArgument(
                        "def_value[", i_dense,
                        "].shape() == ", def_value.shape().DebugString(),
                        " is not compatible with dense_shapes_[", i_dense,
                        "] == ", dense_info.shape.DebugString()));
      }
      OP_REQUIRES(ctx, def_value.dtype() == dense_info.type,
                  errors::InvalidArgument(
                      "dense_defaults[", i_dense, "].dtype() == ",
                      DataTypeString(def_value.dtype()), " != dense_types_[", i_dense,
                      "] == ", DataTypeString(dense_info.type)));
    }

    // Parse values into memory
    OP_REQUIRES_OK(ctx, ParseAndSetValues(&sparse_buffers, &dense_buffers, *serialized, sparse_keys, dense_keys));

    // Convert sparse into tensors
    OP_REQUIRES_OK(ctx, ConvertSparseBufferIntoSparseValuesIndicesShapes(
                          &sparse_values, &sparse_indices, &sparse_shapes, sparse_keys, sparse_buffers, n_serialized));

    #ifdef DEBUG_LOG_ENABLED
      for (int i_dense = 0; i_dense < static_cast<int>(attrs_.num_dense); ++i_dense) {
        LOG(INFO) << "Dense shape for key '" << dense_keys[i_dense].scalar<string>()() << "' is variable length? "
                  << (attrs_.dense_infos[i_dense].variable_length ? "yes" : "no");
      }
    #endif

    // Convert var len dense into tensors
    OP_REQUIRES_OK(ctx, ConvertVarLenIntoDense(&dense_values, dense_keys, dense_buffers, dense_defaults,
                                               n_serialized));

    // Convert fixed len dense values into tensors
    OP_REQUIRES_OK(ctx, ConvertFixedLenIntoDense(&dense_values, dense_keys, dense_buffers, dense_defaults,
                                                 n_serialized));

    // Get outputs
    OpOutputList dense_values_out;
    OpOutputList sparse_indices_out;
    OpOutputList sparse_values_out;
    OpOutputList sparse_shapes_out;
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_indices", &sparse_indices_out));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_values", &sparse_values_out));
    OP_REQUIRES_OK(ctx, ctx->output_list("sparse_shapes", &sparse_shapes_out));
    OP_REQUIRES_OK(ctx, ctx->output_list("dense_values", &dense_values_out));

    // Set sparse outputs
    for (int64 i_sparse = 0; i_sparse < attrs_.num_sparse; ++i_sparse) {
      sparse_indices_out.set(i_sparse, sparse_indices[i_sparse]);
      sparse_values_out.set(i_sparse, sparse_values[i_sparse]);
      sparse_shapes_out.set(i_sparse, sparse_shapes[i_sparse]);
    }

    // Set dense outputs
    for (int i_dense = 0; i_dense < attrs_.num_dense; ++i_dense) {
      dense_values_out.set(i_dense, dense_values[i_dense]);
    }
  }

  // Get a human readable string representation of the TensorFlow's data type.
  // I could not find a function in TensorFlow that does this.
  //
  // 'data_type' TensorFlow's data type, e.g. DT_INT32.
  //
  // returns a string representation of the data type.
  //
  static string DataTypeToString(DataType data_type) {
    switch (data_type) {
      case DT_STRING:
        return "string";
      case DT_DOUBLE:
        return "double";
      case DT_FLOAT:
        return "float32";
      case DT_INT64:
        return "int64";
      case DT_INT32:
        return "int32";
      case DT_BOOL:
        return "bool";
    }
    return "unknown";
  }

  // Get a human readable string representation of the Avro type.
  // Avro does not have a function like this.
  //
  // 'avro_type' The Avro type.
  //
  // returns a string representation of the Avro type.
  //
  static string AvroTypeToString(avro_type_t avro_type) {
    switch (avro_type) {
      case AVRO_BOOLEAN:
        return "bool";
      case AVRO_BYTES:
        return "byte";
      case AVRO_DOUBLE:
        return "double";
      case AVRO_FLOAT:
        return "float";
      case AVRO_INT64:
        return "long";
      case AVRO_INT32:
        return "int";
      case AVRO_NULL:
        return "null";
      case AVRO_STRING:
        return "string";
      case AVRO_ARRAY:
        return "array";
      case AVRO_ENUM:
        return "enum";
      case AVRO_FIXED:
        return "fixed";
      case AVRO_MAP:
        return "map";
      case AVRO_RECORD:
        return "record";
      case AVRO_UNION:
        return "union";
      case AVRO_LINK:
        return "link";
      default:
        return "unknown";
    }
  }
private:
  // Parses values out of the 'serialized' strings for the 'sparse_keys' and 'dense_keys'. Next, it adds these values to
  // the 'sparse_buffers' and 'dense_buffers' in the same order as they are defined in the keys, respectively.
  Status ParseAndSetValues(std::vector<SparseBuffer>* sparse_buffers, std::vector<SparseBuffer>* dense_buffers,
                           const Tensor& serialized, const OpInputList& sparse_keys, const OpInputList& dense_keys) {

    // Allocate space for and parse sparse and dense features
    std::vector<std::vector<AvroField*>> sparse_features(attrs_.num_sparse); // List of list of sparse Avro fields
    std::vector<std::vector<AvroField*>> dense_features(attrs_.num_dense); // List of list of sparse Avro fields

    for (int i_sparse = 0; i_sparse < static_cast<int>(attrs_.num_sparse); ++i_sparse) {
      TF_RETURN_IF_ERROR(StringToAvroField(sparse_features[i_sparse], sparse_keys[i_sparse].scalar<string>()()));
    }
    for (int i_dense = 0; i_dense < static_cast<int>(attrs_.num_dense); ++i_dense) {
      TF_RETURN_IF_ERROR(StringToAvroField(dense_features[i_dense], dense_keys[i_dense].scalar<string>()()));
    }

    auto serialized_strings = serialized.flat<string>();
    int64 n_serialized = serialized_strings.size();
    for (int i_serialized = 0; i_serialized < static_cast<int>(n_serialized); ++i_serialized) {
      avro_value_t value_;

      // Create instance for reader class
      if (avro_generic_value_new(p_iface_, &value_)) {
        return errors::InvalidArgument("Unable to value for user-supplied schema. ", avro_strerror());
      }

      // Wrap the string with the avro reader
      avro_reader_t reader_ = avro_reader_memory(serialized_strings(i_serialized).data(),
                                                 serialized_strings(i_serialized).size());

      // Read value from string using the avro reader
      avro_value_read(reader_, &value_);

      // Parse sparse features
      for (int i_sparse = 0; i_sparse < static_cast<int>(attrs_.num_sparse); ++i_sparse) {
        TF_RETURN_IF_ERROR(SetValues(&(*sparse_buffers)[i_sparse], sparse_features[i_sparse], value_,
                                     attrs_.sparse_types[i_sparse]));
      }

      // Parse dense features fixed and variable length
      for (int i_dense = 0; i_dense < static_cast<int>(attrs_.num_dense); ++i_dense) {
        TF_RETURN_IF_ERROR(SetValues(&(*dense_buffers)[i_dense], dense_features[i_dense], value_,
                           attrs_.dense_infos[i_dense].type));
      }
      // Free up the reader and value
      avro_reader_free(reader_);
      avro_value_decref(&value_);
    }

    // Clean-up avro field instances for sparse features
    for (int64 i_sparse = 0; i_sparse < sparse_features.size(); ++i_sparse) {
      ClearAvroFields(sparse_features[i_sparse]);
    }
    sparse_features.clear();

    // Clean-up avro field instances for dense features
    for (int64 i_dense = 0; i_dense < dense_features.size(); ++i_dense) {
      ClearAvroFields(dense_features[i_dense]);
    }
    dense_features.clear();

    return Status::OK();
  }

  // Converts sparse buffer data into sparse values, indices, and shapes.
  Status ConvertSparseBufferIntoSparseValuesIndicesShapes(std::vector<Tensor>* sparse_values,
    std::vector<Tensor>* sparse_indices, std::vector<Tensor>* sparse_shapes, const OpInputList& sparse_keys,
    const std::vector<SparseBuffer>& sparse_buffers, const int64 n_serialized) {

    #ifdef DEBUG_LOG_ENABLED
      LOG(INFO) << "Converting " << attrs_.num_sparse << " sparse tensors.";
    #endif

    for (int i_sparse = 0; i_sparse < static_cast<int>(attrs_.num_sparse); ++i_sparse) {
      const size_t n_elements = sparse_buffers[i_sparse].n_elements;

      #ifdef DEBUG_LOG_ENABLED
        LOG(INFO) << "For key '" << sparse_keys[i_sparse].scalar<string>()() << "' found the following end indices.";
        const std::vector<size_t> end_indices = sparse_buffers[i_sparse].end_indices;
        const size_t n_batches = end_indices.size();
        for (size_t i_batches = 0; i_batches < n_batches; ++i_batches) {
          LOG(INFO) << "End [" << i_batches << "] = " << end_indices[i_batches];
        }
        LOG(INFO) << "With " << n_elements << " elements.";
      #endif

      // Sparse shapes
      Tensor shape(DT_INT64, TensorShape({2}));
      auto shape_data = shape.vec<int64>();
      shape_data(0) = n_serialized;
      shape_data(1) = n_elements;
      (*sparse_shapes)[i_sparse] = shape;

      // Sparse indices
      TensorShape indices_shape;
      indices_shape.AddDim(n_elements);
      indices_shape.AddDim(2);
      Tensor indices(DT_INT64, indices_shape);
      (*sparse_indices)[i_sparse] = indices;

      int64* index = &indices.matrix<int64>()(0, 0);
      size_t row_index = 0;

      // The user cannot parse a scalar into a variable length tensor -- it does not make sense
      if (sparse_buffers[i_sparse].end_indices.empty()) {
        return errors::InvalidArgument("Tried to load non-array into VarLenFeature. ",
                                       "Use FixedLenFeature instead for key: '",
                                       sparse_keys[i_sparse].scalar<string>()(), "'.");
      }

      // Build row (for batch) / column index, the row index increases by +1 with each column
      size_t n_total_elements = 0;
      for (size_t end_index : sparse_buffers[i_sparse].end_indices) {
        const size_t n_elems = end_index - n_total_elements;
        for (size_t col_index = 0; col_index < n_elems; ++col_index) {
          *index = row_index;
          index++;
          *index = col_index;
          index++;
        }
        row_index++;
        n_total_elements = end_index;
      }

      #ifdef DEBUG_LOG_ENABLED
        LOG(INFO) << "Created index " << indices.DebugString();
      #endif

      // Sparse values
      TensorShape values_shape;
      values_shape.AddDim(n_elements);
      Tensor values(attrs_.sparse_types[i_sparse], values_shape);
      (*sparse_values)[i_sparse] = values;

      #ifdef DEBUG_LOG_ENABLED
        LOG(INFO) << "Created values " << values.DebugString();
      #endif

      // Based on the type we copy the values
      switch (attrs_.sparse_types[i_sparse]) {
        case DT_STRING: {
          std::move(sparse_buffers[i_sparse].string_list.begin(),
                    sparse_buffers[i_sparse].string_list.end(), values.flat<string>().data());
          break;
        }
        case DT_DOUBLE: {
          std::copy(sparse_buffers[i_sparse].double_list.begin(),
                    sparse_buffers[i_sparse].double_list.end(), values.flat<double>().data());
          break;
        }
        case DT_FLOAT: {
          std::copy(sparse_buffers[i_sparse].float_list.begin(),
                    sparse_buffers[i_sparse].float_list.end(), values.flat<float>().data());
          break;
        }
        case DT_INT64: {
          std::copy(sparse_buffers[i_sparse].int64_list.begin(),
                    sparse_buffers[i_sparse].int64_list.end(), values.flat<int64>().data());
          break;
        }
        case DT_INT32: {
          std::copy(sparse_buffers[i_sparse].int32_list.begin(),
                    sparse_buffers[i_sparse].int32_list.end(), values.flat<int32>().data());
          break;
        }
        case DT_BOOL: {
          std::copy(sparse_buffers[i_sparse].bool_list.begin(),
                    sparse_buffers[i_sparse].bool_list.end(), values.flat<bool>().data());
          break;
        }
        default:
          CHECK(false) << "Should not happen."; /// Error when avro type does not match with tf type
      }
    }

    return Status::OK();
  }

  // Converts variable length dense buffers into dense tensors filling in from defaults if necessary.
  Status ConvertVarLenIntoDense(std::vector<Tensor>* dense_values, const OpInputList& dense_keys,
    const std::vector<SparseBuffer>& dense_buffers, const OpInputList& dense_defaults, const int64 n_serialized) {

    for (int i_dense = 0; i_dense < static_cast<int>(attrs_.num_dense); ++i_dense) {

        // Get the information about the dense tensor
        const DenseInformation& dense_info = attrs_.dense_infos[i_dense];

        // If this is not a variable length dense tensor skip
        if (!dense_info.variable_length) {
          continue;
        }

        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Working on key '" << dense_keys[i_dense].scalar<string>()() << "'.";
        #endif

        // Find the maximum number of features and elements in the batch
        size_t n_max_features = 0;
        const std::vector<size_t>& end_indices = dense_buffers[i_dense].end_indices;
        n_max_features = std::max(n_max_features, end_indices[0]);
        for (size_t i_serialized = 1; i_serialized < end_indices.size(); ++i_serialized) {
            n_max_features = std::max(n_max_features, end_indices[i_serialized] - end_indices[i_serialized-1]);
        }
        size_t n_max_elements = n_max_features/dense_info.elements_per_stride;

        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Found maximum number of " << n_max_elements << " elements.";
        #endif

        // Define the dense tensor shape
        TensorShape out_shape;
        out_shape.AddDim(n_serialized); // This is the number of elements in the batch
        out_shape.AddDim(n_max_elements); // Add the variable length dimension with 'n_max_elements'
        for (int i_dim = 1; i_dim < dense_info.shape.dims(); ++i_dim) {
          out_shape.AddDim(dense_info.shape.dim_size(i_dim));
        }

        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Dense shape for key '" << dense_keys[i_dense].scalar<string>()() << "' is "
                    << out_shape.DebugString();
        #endif

        // Create the dense values for the shape and compute the number of elements per batch
        Tensor values(dense_info.type, out_shape);
        const size_t n_elements = values.NumElements();
        const size_t n_elements_per_batch = n_max_elements * dense_info.elements_per_stride;

        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Maximally we have " << n_max_elements << " elements.";
          LOG(INFO) << "The dense tensor with batches has " << n_elements << " elements.";
        #endif

        // Get the default value tensor
        const Tensor& default_value = dense_defaults[i_dense];

        // Check that we have the correct number of values that need to be filled in
        TF_RETURN_IF_ERROR(CheckDefaultAvailable(dense_keys[i_dense].scalar<string>()(), default_value));

        // Copy the data into the output values
        switch (dense_info.type) {
          case DT_STRING: {
            FillFromScalarAndCopy<string>(n_elements, n_elements_per_batch, dense_buffers[i_dense],
                                          dense_defaults[i_dense], &values);
            break;
          }
          case DT_DOUBLE: {
            FillFromScalarAndCopy<double>(n_elements, n_elements_per_batch, dense_buffers[i_dense],
                                          dense_defaults[i_dense], &values);
            break;
          }
          case DT_FLOAT: {
            FillFromScalarAndCopy<float>(n_elements, n_elements_per_batch, dense_buffers[i_dense],
                                         dense_defaults[i_dense], &values);
            break;
          }
          case DT_INT64: {
            FillFromScalarAndCopy<int64>(n_elements, n_elements_per_batch, dense_buffers[i_dense],
                                         dense_defaults[i_dense], &values);
            break;
          }
          case DT_INT32: {
            FillFromScalarAndCopy<int32>(n_elements, n_elements_per_batch, dense_buffers[i_dense],
                                         dense_defaults[i_dense], &values);
            break;
          }
          case DT_BOOL: {
            FillFromScalarAndCopy<bool>(n_elements, n_elements_per_batch, dense_buffers[i_dense],
                                        dense_defaults[i_dense], &values);
            break;
          }
          default:
            CHECK(false) << "Should not happen."; // Error when avro type does not match with tf type

        }
        (*dense_values)[i_dense] = values;
    }

    return Status::OK();
  }

  // Convert fixed length dense buffers into dense tensors filling in from defaults if necessary.
  Status ConvertFixedLenIntoDense(std::vector<Tensor>* dense_values, const OpInputList& dense_keys,
    const std::vector<SparseBuffer>& dense_buffers, const OpInputList& dense_defaults, const int64 n_serialized) {

    for (int i_dense = 0; i_dense < static_cast<int>(attrs_.num_dense); ++i_dense) {

        // Get the information about the dense tensor
        const DenseInformation& dense_info = attrs_.dense_infos[i_dense];

        // Skip any variable length dense tensors
        if (dense_info.variable_length) {
          continue;
        }

        // Define the output shape
        TensorShape out_shape;
        out_shape.AddDim(n_serialized);
        for (const int64 dim : dense_info.shape.dim_sizes()) {
          out_shape.AddDim(dim);
        }

        // Create the output tensor
        Tensor values(dense_info.type, out_shape);

        // Get the number of values overall and per batch
        size_t n_elements = values.NumElements();
        size_t n_elements_per_batch = dense_info.elements_per_stride;

        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Fixed length dense tensor has " << n_elements << " elements.";
          LOG(INFO) << "The stride size is " << n_elements_per_batch << " elements.";
        #endif

        const Tensor& default_value = dense_defaults[i_dense];

        // Check that we have the correct number of values that need to be filled in
        TF_RETURN_IF_ERROR(CheckDefaultsAvailable(dense_keys[i_dense].scalar<string>()(), n_elements_per_batch,
                                                  dense_buffers[i_dense].end_indices, default_value));

        // Copy values and fill in missing values from the defaults
        switch (dense_info.type) {
          case DT_STRING: {
            CopyVarLen<string>(n_elements, n_elements_per_batch, dense_buffers[i_dense], &values);
            FillInFromValues<string>(n_elements, n_elements_per_batch, default_value,
                                     dense_buffers[i_dense].end_indices, &values);
            break;
          }
          case DT_DOUBLE: {
            CopyVarLen<double>(n_elements, n_elements_per_batch, dense_buffers[i_dense], &values);
            FillInFromValues<double>(n_elements, n_elements_per_batch, default_value,
                                     dense_buffers[i_dense].end_indices, &values);
            break;
          }
          case DT_FLOAT: {
            CopyVarLen<float>(n_elements, n_elements_per_batch, dense_buffers[i_dense], &values);
            FillInFromValues<float>(n_elements, n_elements_per_batch, default_value,
                                    dense_buffers[i_dense].end_indices, &values);
            break;
          }
          case DT_INT64: {
            CopyVarLen<int64>(n_elements, n_elements_per_batch, dense_buffers[i_dense], &values);
            FillInFromValues<int64>(n_elements, n_elements_per_batch, default_value,
                                    dense_buffers[i_dense].end_indices, &values);
            break;
          }
          case DT_INT32: {
            CopyVarLen<int32>(n_elements, n_elements_per_batch, dense_buffers[i_dense], &values);
            FillInFromValues<int32>(n_elements, n_elements_per_batch, default_value,
                                    dense_buffers[i_dense].end_indices, &values);
            break;
          }
          case DT_BOOL: {
            CopyVarLen<bool>(n_elements, n_elements_per_batch, dense_buffers[i_dense], &values);
            FillInFromValues<bool>(n_elements, n_elements_per_batch, default_value,
                                   dense_buffers[i_dense].end_indices, &values);
            break;
          }
          default:
            CHECK(false) << "Should not happen."; /// Error when avro type does not match with tf type
        }

        (*dense_values)[i_dense] = values;
    }

    return Status::OK();
  }

  // Parses a string into avro fields. ASSUMES that the vector for 'avro_fields' is empty!
  //
  // 'avro_fields' An empty vector where we add the avro_fields.
  //
  // 'str' Parse this string into Avro types.
  //
  static Status StringToAvroField(std::vector<AvroField*>& avro_fields, const string& str) {
    // Split into tokens using the separator
    std::vector<string> tokens = stringSplit(str, '/');
    avro_fields.resize(tokens.size());

    // Go through all tokens and get their type depending on the surrounding context, e.g. [], [*], [...=...], ['*'],
    // ['...']
    for (int t = 0; t < static_cast<int>(tokens.size()); ++t) {
      string token = tokens[t];

      // Check for map
      if (token.length() >= 4 && token.front() == '[' && token.back() == ']' && token[1] == '\''
        && token[token.length()-2] == '\'') {
        token = token.substr(2, token.length()-4); // Remove brackets and ticks
        if (token.length() == 1 && token[0] == '*') {
          avro_fields[t] = new AvroFieldMapAsterisk();
        } else if (token.length() > 0) {
          avro_fields[t] = new AvroFieldKey(token);
        } else {
          return errors::InvalidArgument("Unable to parse map '", token, "' in '", str, "'.");
        }

      // Check for array
      } else if (token.length() >= 2 && token.front() == '[' && token.back() == ']') {
        token = token.substr(1, token.length()-2); // Remove the brackets

        // Found the asterisk for arrays
        if (token.length() == 1 && token[0] == '*') {
          avro_fields[t] = new AvroFieldArrayAsterisk();
        } else if (isNonNegativeInt(token)) {
          avro_fields[t] = new AvroFieldIndex(std::stoi(token));
        } else if (std::count(token.begin(), token.end(), '=') == 1) {
          std::vector<string> keyValue = stringSplit(token, '=');
          avro_fields[t] = new AvroFieldArrayFilter(keyValue[0], keyValue[1]);
        } else {
          return errors::InvalidArgument("Unable to parse array '", token, "' in '", str, "'.");
        }

      // Check for avro field
      } else if (token.length() > 0) {
        avro_fields[t] = new AvroFieldName(token);
      } else {
        return errors::InvalidArgument("Unable to parse '", token, "' in '", str, "'.");
      }
    }
    return Status::OK();
  }

  // Frees the AvroTypes* in the vector.
  //
  // 'avro_fields' A vector with Avro fields.
  //
  static inline void ClearAvroFields(std::vector<AvroField*> avro_fields) {
    for (auto& field : avro_fields) { delete field; }
    avro_fields.clear();
  }

  // Builds the type compatibility matrix for primitive TensorFlow types to primitive Avro types.
  //
  void BuildCompatibilityMatrix() {
    // The primitive TensorFlow types
    DataType data_types[] = {DT_STRING, DT_DOUBLE, DT_FLOAT, DT_INT64, DT_INT32, DT_BOOL};
    // The primitive Avro types
    avro_type_t avro_types[] = {AVRO_STRING, AVRO_BYTES, AVRO_INT32, AVRO_INT64, AVRO_FLOAT, AVRO_DOUBLE, AVRO_BOOLEAN,
                        AVRO_NULL, AVRO_RECORD, AVRO_ENUM, AVRO_FIXED, AVRO_MAP, AVRO_ARRAY, AVRO_UNION, AVRO_LINK};
    int max_data_types = 0;
    for (size_t i_data_types = 0; i_data_types < sizeof(data_types)/sizeof(DataType); ++i_data_types) {
      max_data_types = std::max(max_data_types, (int)data_types[i_data_types]);
    }
    int max_avro_types = 0;
    for (size_t i_avro_types = 0; i_avro_types < sizeof(avro_types)/sizeof(avro_type_t); ++i_avro_types) {
      max_avro_types = std::max(max_avro_types, (int)avro_types[i_avro_types]);
    }
    max_data_types++; // +1 because we start from 0
    max_avro_types++; // +1 because we start from 0
    compatibility_matrix.resize(max_data_types);
    for (size_t i_data_types = 0; i_data_types < max_data_types; ++i_data_types) {
      compatibility_matrix[i_data_types].resize(max_avro_types, false);
    }

    // These are the compatible types
    compatibility_matrix[DT_STRING][AVRO_STRING] = true;
    compatibility_matrix[DT_DOUBLE][AVRO_DOUBLE] = true;
    compatibility_matrix[DT_FLOAT][AVRO_FLOAT] = true;
    compatibility_matrix[DT_INT64][AVRO_INT64] = true;
    compatibility_matrix[DT_INT32][AVRO_INT32] = true;
    compatibility_matrix[DT_BOOL][AVRO_BOOLEAN] = true;
    compatibility_matrix[DT_STRING][AVRO_BYTES] = true;

    // Add support for union types
    compatibility_matrix[DT_STRING][AVRO_UNION] = true;
    compatibility_matrix[DT_DOUBLE][AVRO_UNION] = true;
    compatibility_matrix[DT_FLOAT][AVRO_UNION] = true;
    compatibility_matrix[DT_INT64][AVRO_UNION] = true;
    compatibility_matrix[DT_INT32][AVRO_UNION] = true;
    compatibility_matrix[DT_BOOL][AVRO_UNION] = true;

    // Add support for null types
    compatibility_matrix[DT_STRING][AVRO_NULL] = true;
    compatibility_matrix[DT_DOUBLE][AVRO_NULL] = true;
    compatibility_matrix[DT_FLOAT][AVRO_NULL] = true;
    compatibility_matrix[DT_INT64][AVRO_NULL] = true;
    compatibility_matrix[DT_INT32][AVRO_NULL] = true;
    compatibility_matrix[DT_BOOL][AVRO_NULL] = true;
  }

  // Checks the compatibility between data types.
  //
  // 'data_type' The TensorFlow primitive data type.
  //
  // 'avro_type' The Avro primitive data type.
  //
  // returns True if the types are compatible otherwise false.
  //
  bool Compatible(const DataType data_type, const avro_type_t avro_type) const {
    return compatibility_matrix[data_type][avro_type];
  }

  // Resolves union(s).
  //
  // 'resolved_avro_value' The resolved avro value.
  //
  // 'avro_value' The avro value.
  //
  // returns An error status if we were unable to resolve the union; otherwise OK.
  //
  static Status ResolveUnion(avro_value_t* resolved_avro_value, const avro_value_t& avro_value) {
    *resolved_avro_value = avro_value;
    avro_type_t field_type = avro_value_get_type(resolved_avro_value);
    while (field_type == AVRO_UNION) {
      avro_value_t branch_avro_value;
      TF_RETURN_IF_ERROR(avro_value_get_current_branch(resolved_avro_value, &branch_avro_value) == 0 ? Status::OK() :
                         errors::InvalidArgument("Could not resolve union. ", avro_strerror()));
      *resolved_avro_value = branch_avro_value;
      field_type = avro_value_get_type(resolved_avro_value);
    }
    return Status::OK();
  }

  // Adds a primitive data type to the 'data_buffer'. Notice that the 'data_buffer' has vectors for each type.
  // This method picks the correct type according to the type defined by avro and compatible with the TensorFlow type.
  //
  // 'data_buffer' A variable length buffer to add data.
  //
  // 'avro_value' The avro value that is used to pull the data.
  //
  // 'data_type' The TensorFlow data type.
  //
  // returns Status that indicates OK or an error with message.
  //
  Status AddPrimitiveType(SparseBuffer* data_buffer, const avro_value_t& avro_value, const DataType data_type) const {
    avro_value_t next_avro_value;
    avro_type_t field_type;

    // Resolve union(s) for the avro value
    TF_RETURN_IF_ERROR(ResolveUnion(&next_avro_value, avro_value));
    field_type = avro_value_get_type(&next_avro_value);

    // Check for type compatibility
    TF_RETURN_IF_ERROR(Compatible(data_type, field_type) ? Status::OK() : errors::InvalidArgument(
      "Incompatible types! User defined type ", DataTypeToString(data_type), " but avro contains type ",
      AvroTypeToString(field_type)));

    // Add one for the element that we will added in the switch statement (if no element is added an error will occur)
    data_buffer->n_elements += 1;

    // Select the avro type and pull the data accordingly form the 'avro_value'
    switch (field_type) {
      // Handle null types in avro
      case AVRO_NULL: {
        // Note: Can't add default types because these are not fully supported by TF python classes we use here!
        // For now throw an error!
        TF_RETURN_IF_ERROR(errors::InvalidArgument("Could not parse avro null type into tensorflow type'",
                                                   DataTypeToString(data_type), "'."));
        break;
      }
      case AVRO_STRING: {
        const char* field_value = nullptr; // just a pointer to the data not a copy, no need to free this
        size_t field_size = 0;
        TF_RETURN_IF_ERROR(avro_value_get_string(&next_avro_value, &field_value, &field_size) == 0 ? Status::OK() :
                           errors::InvalidArgument("Could not extract string. ", avro_strerror()));
        data_buffer->string_list.push_back(string(field_value, field_size-1)); // -1 to remove the escape character
        break;
      }
      case AVRO_BYTES: {
        const void* field_value = nullptr;
        size_t field_size = 0;
        TF_RETURN_IF_ERROR(avro_value_get_bytes(&next_avro_value, &field_value, &field_size) == 0 ? Status::OK() :
                           errors::InvalidArgument("Could not extract bytes. ", avro_strerror()));
        data_buffer->string_list.push_back(string((const char*)field_value, field_size));
        break;
      }
      case AVRO_FLOAT: {
        float field_value = 0;
        TF_RETURN_IF_ERROR(avro_value_get_float(&next_avro_value, &field_value) == 0 ? Status::OK() :
                           errors::InvalidArgument("Could not extract float. ", avro_strerror()));
        data_buffer->float_list.push_back(field_value);
        break;
      }
      case AVRO_DOUBLE: {
        double field_value = 0;
        TF_RETURN_IF_ERROR(avro_value_get_double(&next_avro_value, &field_value) == 0 ? Status::OK() :
                           errors::InvalidArgument("Could not extract double.", avro_strerror()));
        data_buffer->double_list.push_back(field_value);
        break;
      }
      case AVRO_INT64: {
        long field_value = 0;
        TF_RETURN_IF_ERROR(avro_value_get_long(&next_avro_value, &field_value) == 0 ? Status::OK() :
                           errors::InvalidArgument("Could not extract long. ", avro_strerror()));
        data_buffer->int64_list.push_back(field_value);
        break;
      }
      case AVRO_INT32: {
        int field_value = 0;
        TF_RETURN_IF_ERROR(avro_value_get_int(&next_avro_value, &field_value) == 0 ? Status::OK() :
                           errors::InvalidArgument("Could not extract int. ",  avro_strerror()));
        data_buffer->int32_list.push_back(field_value);
        break;
      }
      case AVRO_BOOLEAN: {
        int field_value = 0;
        TF_RETURN_IF_ERROR(avro_value_get_boolean(&next_avro_value, &field_value) == 0 ? Status::OK() :
                           errors::InvalidArgument("Could not extract boolean.", avro_strerror()));
        data_buffer->bool_list.push_back(field_value == 1 ? true : false);
        break;
      }
      default: {
        TF_RETURN_IF_ERROR(errors::InvalidArgument("Could not parse type '", AvroTypeToString(field_type),
          "', expected primitive type."));
      }
    }
    return Status::OK();
  }

  // This is a custom method to get a string/bytes value from a record. It is used for filters.
  //
  // 'name' The name corresponding to the key.
  //
  // 'avro_value' The avro value that is a record with the 'name' attribute.
  //
  // 'key' The attribute name.
  //
  // returns Status that indicates OK or an error with message.
  //
  static Status GetAttrInRecord(string* name, const avro_value_t& avro_value, const string& key) {
    avro_value_t next_avro_value;
    avro_type_t field_type;
    avro_value_t child_avro_value;

    // Resolve union(s) for the avro value
    TF_RETURN_IF_ERROR(ResolveUnion(&next_avro_value, avro_value));
    field_type = avro_value_get_type(&next_avro_value);

    // Make sure it's a record
    TF_RETURN_IF_ERROR(field_type == AVRO_RECORD ? Status::OK() :
              errors::InvalidArgument("Expected type 'record' but found type '", AvroTypeToString(field_type), "'."));

    // Make sure the attribute exists
    TF_RETURN_IF_ERROR(avro_value_get_by_name(&next_avro_value, key.c_str(), &child_avro_value, NULL) == 0 ? Status::OK() :
                        errors::InvalidArgument("Could not find name'", key, "'."));

    // Resolve union(s) for child avro value
    TF_RETURN_IF_ERROR(ResolveUnion(&child_avro_value, child_avro_value));
    field_type = avro_value_get_type(&child_avro_value);

    // Check for attribute of type string or bytes
    switch (field_type) {
      case AVRO_STRING: {
        const char* field_value = nullptr; // just a pointer to the data not a copy, no need to free this
        size_t field_size = 0;
        TF_RETURN_IF_ERROR(avro_value_get_string(&child_avro_value, &field_value, &field_size) == 0 ?
                           Status::OK() : errors::InvalidArgument("Could not extract string. ", avro_strerror()));
        *name = string(field_value, field_size-1);
        break;
      }
      case AVRO_BYTES: {
        const void* field_value = nullptr;
        size_t field_size = 0;
        TF_RETURN_IF_ERROR(avro_value_get_bytes(&child_avro_value, &field_value, &field_size) == 0 ?
                           Status::OK() : errors::InvalidArgument("Could not extract bytes. ", avro_strerror()));
        *name = string((const char*)field_value, field_size);
        break;
      }
      default: {
        return errors::InvalidArgument("Expected type 'string' or 'bytes' but found type '",
                                        AvroTypeToString(field_type), "'.");
      }
    }
    return Status::OK();
  }

  // Sets values in the buffer for a given avro field.
  //
  // 'data_buffer' Data buffer is add the data.
  //
  // 'features_avro_fields' Avro fields for a single feature string that is defined by the caller of the parser.
  //
  // 'avro_value' The avro value for this record.
  //
  // 'data_type' The expected TensorFlow type for the Avro value as defined by 'features_avro_fields'.
  //
  // 'i_start_avro_field' Start following the fields from this start index on. We need this recursion to support
  //    the asterisk wildcard.
  //
  // returns Status that indicates OK or an error with message.
  //
  Status SetValues(SparseBuffer* data_buffer, const std::vector<AvroField*>& features_avro_fields,
      const avro_value_t& avro_value, const DataType data_type, int i_start_avro_field = 0, bool in_array = false) const {
    avro_value_t this_avro_value = avro_value;
    avro_value_t next_avro_value;
    avro_value_t child_avro_value;
    avro_type_t field_type;
    size_t n_elements;
    size_t i_elements;
    string value_filtered;

    // Find the Avro value for the specified field as given in 'features_avro_fields'
    for (int i_avro_field = i_start_avro_field; i_avro_field < features_avro_fields.size(); ++i_avro_field) {
      AvroField* avro_field = features_avro_fields[i_avro_field];

      // Resolve union(s) for this avro value
      TF_RETURN_IF_ERROR(ResolveUnion(&this_avro_value, this_avro_value));
      field_type = avro_value_get_type(&this_avro_value);

      // User wants attribute in Avro record
      if (avro_field->getType() == AvroField::Type::name) {
        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Parsing attribute " << static_cast<AvroFieldName*>(avro_field)->toString();
        #endif
        TF_RETURN_IF_ERROR(avro_value_get_by_name(&this_avro_value,
                                                  static_cast<AvroFieldName*>(avro_field)->getName().c_str(),
                                                  &next_avro_value, NULL) == 0
                           ? Status::OK() : errors::InvalidArgument("Could not find name '", avro_field->toString(), "'."));

      // User wants all items in an array
      } else if (avro_field->getType() == AvroField::Type::arrayAsterisk) {
        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Parsing asterisk *";
        #endif

        // When the asterisk is defined we expect an array
        TF_RETURN_IF_ERROR(field_type == AVRO_ARRAY ? Status::OK() :
          errors::InvalidArgument("Expected type 'array' but found type '", AvroTypeToString(field_type), "'."));

        // Go over all elements in the array and select the appropriate value in the field underneath
        avro_value_get_size(&this_avro_value, &n_elements);
        for (i_elements = 0; i_elements < n_elements; ++i_elements) {
          // Don't check for errors here because there always should be the correct number of elements as given by size
          avro_value_get_by_index(&this_avro_value, i_elements, &child_avro_value, NULL);
          TF_RETURN_IF_ERROR(SetValues(data_buffer, features_avro_fields, child_avro_value, data_type, i_avro_field+1, true));
        }
        // Add the number of elements to the end indices
        data_buffer->end_indices.push_back(data_buffer->n_elements);
        return Status::OK(); // Done with parsing, return.

      // User defined an Avro array and want's to filter values
      } else if (avro_field->getType() == AvroField::Type::arrayFilter) {
        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Parsing filter " << static_cast<AvroFieldArrayFilter*>(avro_field)->toString();
        #endif
        TF_RETURN_IF_ERROR(field_type == AVRO_ARRAY ? Status::OK() :
          errors::InvalidArgument("Expected type 'array' but found type '", AvroTypeToString(field_type), "'."));

        // This code is similar to the one walking over all entries in an array
        avro_value_get_size(&this_avro_value, &n_elements);

        // Go over all elements in the array and select the appropriate value in the record underneath
        for (i_elements = 0; i_elements < n_elements; ++i_elements) {
          // Don't check for errors here because there always should be the correct number of elements as given by size
          avro_value_get_by_index(&this_avro_value, i_elements, &child_avro_value, NULL);

          // Get the value for the key
          TF_RETURN_IF_ERROR(GetAttrInRecord(&value_filtered,
                                             child_avro_value, static_cast<AvroFieldArrayFilter*>(avro_field)->getKey()));

          #ifdef DEBUG_LOG_ENABLED
            LOG(INFO) << "For " << static_cast<AvroFieldArrayFilter*>(avro_field)->getKey() << " found " << value_filtered;
          #endif

          // If the value matches the filter add it; otherwise not
          if (value_filtered == static_cast<AvroFieldArrayFilter*>(avro_field)->getValue()) {
            TF_RETURN_IF_ERROR(SetValues(data_buffer, features_avro_fields, child_avro_value, data_type, i_avro_field+1, true));
          }
        }
        // Add the number of elements to the end indices
        data_buffer->end_indices.push_back(data_buffer->n_elements);
        return Status::OK(); // Done with parsing, return.

      // User defined an Avro array and want's to get an item for an index
      } else if (avro_field->getType() == AvroField::Type::index) {
        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Parsing array index " << static_cast<AvroFieldIndex*>(avro_field)->toString();
        #endif
        TF_RETURN_IF_ERROR(field_type == AVRO_ARRAY ? Status::OK() :
          errors::InvalidArgument("Expected type 'array' but found type '", AvroTypeToString(field_type), "'."));

        TF_RETURN_IF_ERROR(avro_value_get_by_index(&this_avro_value,
                                                   static_cast<AvroFieldIndex*>(avro_field)->getIndex(),
                                                   &next_avro_value, NULL) == 0
                           ? Status::OK() : errors::InvalidArgument("Could not find index '", avro_field->toString(), "'."));

      // User wants key in a map
      } else if (avro_field->getType() == AvroField::Type::key) {
        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Parsing key " << static_cast<AvroFieldKey*>(avro_field)->toString();
        #endif

        // Make sure we have either a map or array
        TF_RETURN_IF_ERROR(field_type == AVRO_MAP ? Status::OK() :
          errors::InvalidArgument("Expected type 'map' but found type '", AvroTypeToString(field_type), "'."));

        // Same function as for field access that is used to resolve Avro field names
        TF_RETURN_IF_ERROR(avro_value_get_by_name(&this_avro_value,
                                                  static_cast<AvroFieldKey*>(avro_field)->getKey().c_str(),
                                                  &next_avro_value, NULL) == 0
                           ? Status::OK() : errors::InvalidArgument("Could not get key '", avro_field->toString(), "'."));

      // User wants all keys in map
      } else if (avro_field->getType() == AvroField::Type::mapAsterisk) {
        #ifdef DEBUG_LOG_ENABLED
          LOG(INFO) << "Parsing key " << static_cast<AvroFieldMapAsterisk*>(avro_field)->toString();
        #endif

        // Make sure we have either a map or array
        TF_RETURN_IF_ERROR(field_type == AVRO_MAP ? Status::OK() :
          errors::InvalidArgument("Expected type 'map' but found type '", AvroTypeToString(field_type), "'."));

        // This code is the same as for the array asterisk from above (we distinguish the two cases to check Avro types)
        avro_value_get_size(&this_avro_value, &n_elements);
        for (i_elements = 0; i_elements < n_elements; ++i_elements) {
          avro_value_get_by_index(&this_avro_value, i_elements, &child_avro_value, NULL); // NULL is where we could get the key name
          TF_RETURN_IF_ERROR(SetValues(data_buffer, features_avro_fields, child_avro_value, data_type, i_avro_field+1, true));
        }

        // Add the number of elements to the end indices
        data_buffer->end_indices.push_back(data_buffer->n_elements);
        return Status::OK(); // Done with parsing, return.

      // This should not happen with the current implementation since we covered all types; but it's safer in case
      // new types are introduced but not handled here
      } else {
        return errors::InvalidArgument("Found unsupported avro field type '", avro_field->getType(), "'.");
      }

      // If there is an error that means the user-defined name was not defined
      this_avro_value = next_avro_value;
      field_type = avro_value_get_type(&this_avro_value);
    }

    // Here we expect the array/map to contain a primitive type in the items and not any records
    if (field_type == AVRO_ARRAY || field_type == AVRO_MAP) {
      avro_value_get_size(&this_avro_value, &n_elements);
      for (i_elements = 0; i_elements < n_elements; ++i_elements) {
        avro_value_get_by_index(&this_avro_value, i_elements, &child_avro_value, NULL);
        TF_RETURN_IF_ERROR(AddPrimitiveType(data_buffer, child_avro_value, data_type));
      }
      data_buffer->end_indices.push_back(data_buffer->n_elements);

      #ifdef DEBUG_LOG_ENABLED
        LOG(INFO) << "Parsed array or map with " << n_elements << " elements to a total of " << data_buffer->n_elements
                  << " elements.";
      #endif

    } else {
      TF_RETURN_IF_ERROR(AddPrimitiveType(data_buffer, this_avro_value, data_type));

      // If we are not in an array we need to add this one to the end indices -- otherwise done after processing all elements
      if (!in_array) {
        data_buffer->end_indices.push_back(data_buffer->n_elements);
      }
    }

    return Status::OK();
  }
  std::vector<std::vector<bool>> compatibility_matrix; // Type compatibility matrix
  int max_data_types; // Maximum integer for TensorFlow's data types (notice that is not the number of types)
  int max_avro_types; // Maximum integer for Avro types (notice that is not the number of Avro types)
  avro_schema_t reader_schema_; // The Avro reader schema
  avro_value_iface_t* p_iface_; // The class information for Avro values to generate generic data
  ParseAvroAttrs attrs_; // Attributes for this operator that contain number information and shape information
};

// Register the parser with TensorFlow
REGISTER_KERNEL_BUILDER(Name("ParseAvroRecord").Device(DEVICE_CPU), ParseAvroRecordOp);
