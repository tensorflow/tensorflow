#include "tensorflow/core/kernels/queue_base.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

namespace {

template <DataType DT>
void HandleSliceToElement(const Tensor& parent, Tensor* element, int index) {
  typedef typename EnumToDataType<DT>::Type T;
  auto parent_as_matrix = parent.flat_outer_dims<T>();
  element->flat<T>() = parent_as_matrix.chip(index, 0);
}

template <DataType DT>
void HandleElementToSlice(const Tensor& element, Tensor* parent, int index) {
  typedef typename EnumToDataType<DT>::Type T;
  auto parent_as_matrix = parent->flat_outer_dims<T>();
  parent_as_matrix.chip(index, 0) = element.flat<T>();
}

}  // namespace

// static
Status QueueBase::CopySliceToElement(const Tensor& parent, Tensor* element,
                                     int index) {
#define HANDLE_TYPE(DT)                               \
  if (parent.dtype() == DT) {                         \
    HandleSliceToElement<DT>(parent, element, index); \
    return Status::OK();                              \
  }
  HANDLE_TYPE(DT_FLOAT);
  HANDLE_TYPE(DT_DOUBLE);
  HANDLE_TYPE(DT_INT32);
  HANDLE_TYPE(DT_UINT8);
  HANDLE_TYPE(DT_INT16);
  HANDLE_TYPE(DT_INT8);
  HANDLE_TYPE(DT_STRING);
  HANDLE_TYPE(DT_INT64);
#undef HANDLE_TYPE
  return errors::Unimplemented("Unhandled data type: ", parent.dtype());
}

// static
Status QueueBase::CopyElementToSlice(const Tensor& element, Tensor* parent,
                                     int index) {
#define HANDLE_TYPE(DT)                               \
  if (element.dtype() == DT) {                        \
    HandleElementToSlice<DT>(element, parent, index); \
    return Status::OK();                              \
  }
  HANDLE_TYPE(DT_FLOAT);
  HANDLE_TYPE(DT_DOUBLE);
  HANDLE_TYPE(DT_INT32);
  HANDLE_TYPE(DT_UINT8);
  HANDLE_TYPE(DT_INT16);
  HANDLE_TYPE(DT_INT8);
  HANDLE_TYPE(DT_STRING);
  HANDLE_TYPE(DT_INT64);
#undef HANDLE_TYPE
  return errors::Unimplemented("Unhandled data type: ", element.dtype());
}

QueueBase::QueueBase(const DataTypeVector& component_dtypes,
                     const std::vector<TensorShape>& component_shapes,
                     const string& name)
    : component_dtypes_(component_dtypes),
      component_shapes_(component_shapes),
      name_(name) {}

Status QueueBase::ValidateTupleCommon(const Tuple& tuple) const {
  if (tuple.size() != static_cast<size_t>(num_components())) {
    return errors::InvalidArgument(
        "Wrong number of components in tuple. Expected ", num_components(),
        ", got ", tuple.size());
  }
  for (size_t i = 0; i < tuple.size(); ++i) {
    if (tuple[i].dtype() != component_dtypes_[i]) {
      return errors::InvalidArgument(
          "Type mismatch in tuple component ", i, ". Expected ",
          DataTypeString(component_dtypes_[i]), ", got ",
          DataTypeString(tuple[i].dtype()));
    }
  }
  return Status::OK();
}

// static
string QueueBase::ShapeListString(const gtl::ArraySlice<TensorShape>& shapes) {
  string result = "[";
  bool first = true;
  for (const TensorShape& shape : shapes) {
    strings::StrAppend(&result, (first ? "" : ", "), shape.ShortDebugString());
    first = false;
  }
  strings::StrAppend(&result, "]");
  return result;
}

Status QueueBase::MatchesNodeDefOp(const NodeDef& node_def,
                                   const string& op) const {
  if (node_def.op() != op) {
    return errors::InvalidArgument("Shared queue '", name_, "' has type '", op,
                                   "' that does not match type of Node '",
                                   node_def.name(), "': ", node_def.op());
  }
  return Status::OK();
}

Status QueueBase::MatchesNodeDefCapacity(const NodeDef& node_def,
                                         int32 capacity) const {
  int32 requested_capacity = -1;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "capacity", &requested_capacity));
  if (requested_capacity < 0) requested_capacity = kUnbounded;
  if (requested_capacity != capacity) {
    return errors::InvalidArgument("Shared queue '", name_, "' has capacity ",
                                   capacity, " but requested capacity was ",
                                   requested_capacity);
  }
  return Status::OK();
}

Status QueueBase::MatchesNodeDefTypes(const NodeDef& node_def) const {
  DataTypeVector requested_dtypes;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node_def, "component_types", &requested_dtypes));
  if (requested_dtypes != component_dtypes_) {
    return errors::InvalidArgument("Shared queue '", name_,
                                   "' has component types ",
                                   DataTypeSliceString(component_dtypes_),
                                   " but requested component types were ",
                                   DataTypeSliceString(requested_dtypes));
  }
  return Status::OK();
}

Status QueueBase::MatchesNodeDefShapes(const NodeDef& node_def) const {
  std::vector<TensorShape> requested_shapes;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "shapes", &requested_shapes));
  if (requested_shapes != component_shapes_) {
    return errors::InvalidArgument("Shared queue '", name_,
                                   "' has component shapes ",
                                   ShapeListString(component_shapes_),
                                   " but requested component shapes were ",
                                   ShapeListString(requested_shapes));
  }
  return Status::OK();
}

}  // namespace tensorflow
