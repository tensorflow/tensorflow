#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

int64 CountShapes(const Shape& shape) {
  int64 n = 0;
  if (ShapeUtil::IsTuple(shape)) {
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); i++) {
      n += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
    }
    return n;
  } else {
    return 1;
  }
}

std::vector<xla::Shape> FlattenedXlaShape(const xla::Shape& shape) {
  std::vector<xla::Shape> out;
  if (ShapeUtil::IsTuple(shape)) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); i++) {
      std::vector<xla::Shape> shapes =
          FlattenedXlaShape(ShapeUtil::GetTupleElementShape(shape, i));
      out.insert(out.end(), shapes.begin(), shapes.end());
    }
  } else {
    out.push_back(shape);
  }

  return out;
}

StatusOr<std::vector<int64>> LiteralVectorToInt64Vector(
    const xla::Literal& lit) {
  if (lit.shape().dimensions_size() != 1) {
    return xla::FailedPrecondition("Literal rank != 1");
  }

  Literal s64_lit;
  TF_ASSIGN_OR_RETURN(s64_lit, lit.Convert(S64));

  const int64* start = static_cast<const int64*>(s64_lit.untyped_data());
  return std::vector<int64>(start, start + s64_lit.shape().dimensions(0));
}

StatusOr<std::vector<int64>> WideConstToInt64Vector(
    const xla::HloInstruction* bcast, const xla::HloInstruction* constant) {
  CHECK_EQ(bcast->opcode(), HloOpcode::kBroadcast);
  if (bcast->shape().dimensions_size() != 1) {
    return xla::FailedPrecondition("Literal rank != 1");
  }
  CHECK_EQ(constant->opcode(), HloOpcode::kConstant);

  int64 val;
  TF_ASSIGN_OR_RETURN(val, LiteralScalarInt64toInt64(constant->literal()));
  return std::vector<int64>(bcast->shape().dimensions(0), val);
}

StatusOr<int32> LiteralScalarInt32toInt32(const xla::Literal& lit) {
  if (!ShapeUtil::IsScalar(lit.shape())) {
    return xla::FailedPrecondition("Literal is not scalar");
  }

  Literal s32_lit;
  TF_ASSIGN_OR_RETURN(s32_lit, lit.Convert(S32));

  return *static_cast<const int32*>(s32_lit.untyped_data());
}

StatusOr<int64> LiteralScalarInt64toInt64(const xla::Literal& lit) {
  if (!ShapeUtil::IsScalar(lit.shape())) {
    return xla::FailedPrecondition("Literal is not scalar");
  }

  Literal s64_lit;
  TF_ASSIGN_OR_RETURN(s64_lit, lit.Convert(S64));

  return *static_cast<const int64*>(s64_lit.untyped_data());
}

StatusOr<double> LiteralScalarDoubleToDouble(const xla::Literal& lit) {
  if (!ShapeUtil::IsScalar(lit.shape())) {
    return xla::FailedPrecondition("Literal is not scalar");
  }

  Literal double_lit;
  TF_ASSIGN_OR_RETURN(double_lit, lit.Convert(F64));

  return *static_cast<const double*>(double_lit.untyped_data());
}

bool IsPopOpsCall(const xla::HloComputation* comp, const std::string& postfix) {
  return tensorflow::str_util::StartsWith(comp->name(), "_pop_op_" + postfix);
}

bool IsPopOpsCall(const xla::HloInstruction* inst, const std::string& postfix) {
  return inst->opcode() == xla::HloOpcode::kCall &&
         IsPopOpsCall(inst->to_apply(), postfix);
}

bool UseSyntheticData() {
  if (const char* env_c = std::getenv("TF_POPLAR_USE_SYNTHETIC_DATA")) {
    std::string env(env_c);
    std::transform(env.begin(), env.end(), env.begin(), ::tolower);
    return env == "true";
  }
  return false;
}

std::string GetDebugName(const HloInstruction* inst) {
  const std::string& tf_core_name = inst->metadata().op_name();
  return tf_core_name + "/" + inst->name();
}

}  // namespace poplarplugin
}  // namespace xla
