#include "tensorflow/cc/ops/const_op.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace ops {

namespace {
const string& OpName() {
  static const string kOpName = "Const";
  return kOpName;
}
}  // namespace

#define DEFINE_CONST_SCALAR(TYPE)                                         \
  Node* Const(TYPE s, const GraphDefBuilder::Options& options) {          \
    return Const(gtl::ArraySlice<TYPE>(&s, 1), TensorShape({}), options); \
  }

#define DEFINE_CONST_VECTOR(TYPE)                                          \
  Node* Const(gtl::ArraySlice<TYPE> v,                                     \
              const GraphDefBuilder::Options& options) {                   \
    return Const(v, TensorShape({static_cast<int64>(v.size())}), options); \
  }

#define DEFINE_CONST_TENSOR(TYPE, ...)                                         \
  Node* Const(gtl::ArraySlice<TYPE> t, const TensorShape& shape,               \
              const GraphDefBuilder::Options& options) {                       \
    if (options.HaveError()) return nullptr;                                   \
    NodeBuilder node_builder(options.GetNameForOp(OpName()), OpName(),         \
                             options.op_registry());                           \
    const DataType dt = DataTypeToEnum<TYPE>::v();                             \
    if (t.size() == 1) {                                                       \
      TensorProto proto;                                                       \
      proto.set_dtype(dt);                                                     \
      shape.AsProto(proto.mutable_tensor_shape());                             \
      __VA_ARGS__;                                                             \
      node_builder.Attr("dtype", dt).Attr("value", proto);                     \
    } else {                                                                   \
      Tensor tensor(dt, shape);                                                \
      if (tensor.NumElements() != static_cast<int64>(t.size())) {              \
        options.UpdateStatus(errors::InvalidArgument(                          \
            t.size(), " values provided to Const() != ", tensor.NumElements(), \
            " elements for shape ", shape.ShortDebugString()));                \
      } else {                                                                 \
        std::copy_n(t.data(), t.size(), tensor.flat<TYPE>().data());           \
        node_builder.Attr("dtype", dt).Attr("value", tensor);                  \
      }                                                                        \
    }                                                                          \
    return options.FinalizeBuilder(&node_builder);                             \
  }

#define DEFINE_CONST_IMPL(TYPE, ...) \
  DEFINE_CONST_SCALAR(TYPE)          \
  DEFINE_CONST_VECTOR(TYPE)          \
  DEFINE_CONST_TENSOR(TYPE, __VA_ARGS__)

#define DEFINE_CONST(TYPE, FIELD) \
  DEFINE_CONST_IMPL(TYPE, proto.add_##FIELD(*t.begin());)

DEFINE_CONST(float, float_val);
DEFINE_CONST(double, double_val);
DEFINE_CONST(int32, int_val);
DEFINE_CONST(uint8, int_val);
DEFINE_CONST(int16, int_val);
DEFINE_CONST(int8, int_val);
DEFINE_CONST(int64, int64_val);
DEFINE_CONST(bool, bool_val);

DEFINE_CONST_IMPL(complex64, proto.add_scomplex_val(t.begin()->real());
                  proto.add_scomplex_val(t.begin()->imag()););

Node* Const(StringPiece s, const GraphDefBuilder::Options& options) {
  if (options.HaveError()) return nullptr;
  NodeBuilder node_builder(options.GetNameForOp(OpName()), OpName(),
                           options.op_registry());
  TensorProto proto;
  proto.set_dtype(DT_STRING);
  TensorShape({}).AsProto(proto.mutable_tensor_shape());
  proto.add_string_val(s.data(), s.size());
  node_builder.Attr("dtype", DT_STRING).Attr("value", proto);
  return options.FinalizeBuilder(&node_builder);
}

DEFINE_CONST_VECTOR(string)
DEFINE_CONST_TENSOR(string, proto.add_string_val(*t.begin());)

#undef DEFINE_CONST
#undef DEFINE_CONST_IMPL
#undef DEFINE_CONST_TENSOR
#undef DEFINE_CONST_VECTOR
#undef DEFINE_CONST_SCALAR

Node* Const(const Tensor& t, const GraphDefBuilder::Options& options) {
  if (options.HaveError()) return nullptr;
  NodeBuilder node_builder(options.GetNameForOp(OpName()), OpName(),
                           options.op_registry());
  node_builder.Attr("dtype", t.dtype()).Attr("value", t);
  return options.FinalizeBuilder(&node_builder);
}

Node* Const(const TensorProto& proto, const GraphDefBuilder::Options& options) {
  if (options.HaveError()) return nullptr;
  NodeBuilder node_builder(options.GetNameForOp(OpName()), OpName(),
                           options.op_registry());
  node_builder.Attr("dtype", proto.dtype()).Attr("value", proto);
  return options.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
