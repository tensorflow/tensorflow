#include "avro_helper.h"

#include <vector>

#include "tensorflow/core/lib/core/errors.h"

using namespace tensorflow;

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
