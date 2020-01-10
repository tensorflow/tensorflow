#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

// Functionality common to QueueInterface implementations.
class QueueBase : public QueueInterface {
 public:
  // As a possible value of 'capacity'.
  static const int32 kUnbounded = INT_MAX;

  // Args:
  //   component_dtypes: The types of each component in a queue-element tuple.
  //   component_shapes: The shapes of each component in a queue-element tuple,
  //     which must either be empty (if the shapes are not specified) or
  //     or have the same size as component_dtypes.
  //   name: A name to use for the queue.
  QueueBase(const DataTypeVector& component_dtypes,
            const std::vector<TensorShape>& component_shapes,
            const string& name);

  // Implementations of QueueInterface methods --------------------------------
  const DataTypeVector& component_dtypes() const override {
    return component_dtypes_;
  }

  // Other public methods -----------------------------------------------------
  const std::vector<TensorShape>& component_shapes() const {
    return component_shapes_;
  }

 protected:
  // Returns the number of components in a queue-element tuple.
  int32 num_components() const { return component_dtypes_.size(); }

  // True if shapes were specified.  If so, inputs will be validated
  // against them, etc.
  bool specified_shapes() const { return component_shapes_.size() > 0; }

  // Code common to Validate*Tuple().
  Status ValidateTupleCommon(const Tuple& tuple) const;

  // Copies the index^th slice (in the first dimension) of parent into element.
  static Status CopySliceToElement(const Tensor& parent, Tensor* element,
                                   int index);

  // Copies element into the index^th slice (in the first dimension) of parent.
  static Status CopyElementToSlice(const Tensor& element, Tensor* parent,
                                   int index);

  ~QueueBase() override {}

  // Helpers for implementing MatchesNodeDef().
  static string ShapeListString(const gtl::ArraySlice<TensorShape>& shapes);
  Status MatchesNodeDefOp(const NodeDef& node_def, const string& op) const;
  Status MatchesNodeDefCapacity(const NodeDef& node_def, int32 capacity) const;
  Status MatchesNodeDefTypes(const NodeDef& node_def) const;
  Status MatchesNodeDefShapes(const NodeDef& node_def) const;

  const DataTypeVector component_dtypes_;
  const std::vector<TensorShape> component_shapes_;
  const string name_;

  TF_DISALLOW_COPY_AND_ASSIGN(QueueBase);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_
