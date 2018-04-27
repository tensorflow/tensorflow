#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"

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