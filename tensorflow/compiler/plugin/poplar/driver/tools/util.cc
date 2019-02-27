#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "absl/types/optional.h"

namespace xla {
namespace poplarplugin {

int64 CountShapes(const Shape& shape) {
  int64 n = 0;
  if (shape.IsTuple()) {
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); i++) {
      n += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
    }
    return n;
  } else {
    return 1;
  }
}

int64 InsertIntoTuple(const Shape& tuple, int64 tuple_index,
                      int64 original_index) {
  // Count up the base tensors inside all tuple element preceeding the
  // tuple_index one.
  int64 tensor_count = 0;
  for (int64 i = 0; i < tuple_index; i++) {
    tensor_count += CountShapes(ShapeUtil::GetTupleElementShape(tuple, i));
  }
  return tensor_count + original_index;
}

std::vector<xla::Shape> FlattenedXlaShape(const xla::Shape& shape) {
  std::vector<xla::Shape> out;
  if (shape.IsTuple()) {
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

template <typename NativeT>
StatusOr<NativeT> LiteralScalarToNativeType(const xla::Literal& lit) {
  auto primitive_type = primitive_util::NativeToPrimitiveType<NativeT>();
  if (!ShapeUtil::IsScalar(lit.shape())) {
    return xla::FailedPrecondition("Literal is not scalar");
  }

  Literal converted_lit;
  TF_ASSIGN_OR_RETURN(converted_lit, lit.Convert(primitive_type));

  return *static_cast<const NativeT*>(converted_lit.untyped_data());
}

template <typename NativeT>
StatusOr<std::vector<NativeT>> LiteralVectorToNativeType(
    const xla::Literal& lit) {
  auto primitive_type = primitive_util::NativeToPrimitiveType<NativeT>();
  if (lit.shape().dimensions_size() != 1) {
    return xla::FailedPrecondition("Literal rank != 1");
  }

  Literal converted_lit;
  TF_ASSIGN_OR_RETURN(converted_lit, lit.Convert(primitive_type));

  const NativeT* start =
      static_cast<const NativeT*>(converted_lit.untyped_data());
  return std::vector<NativeT>(start,
                              start + converted_lit.shape().dimensions(0));
}

template <typename NativeT>
StatusOr<std::vector<NativeT>> WideConstToNativeType(
    const xla::HloInstruction* wide_const) {
  CHECK_EQ(wide_const->opcode(), HloOpcode::kBroadcast);
  if (wide_const->shape().dimensions_size() != 1) {
    return xla::FailedPrecondition("Literal rank != 1");
  }
  const HloInstruction* constant = wide_const->operand(0);
  CHECK_EQ(constant->opcode(), HloOpcode::kConstant);

  TF_ASSIGN_OR_RETURN(NativeT val,
                      LiteralScalarToNativeType<NativeT>(constant->literal()));
  return std::vector<NativeT>(wide_const->shape().dimensions(0), val);
}

template StatusOr<uint8> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<uint16> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<uint32> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<uint64> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<int8> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<int16> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<int32> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<int64> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<half> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<bfloat16> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<float> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<double> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<complex64> LiteralScalarToNativeType(const xla::Literal& lit);
template StatusOr<bool> LiteralScalarToNativeType(const xla::Literal& lit);

#define INITIALISE_FOR_ALL_NATIVE_VECTOR_TYPES(func) \
  template StatusOr<std::vector<uint8>> func;        \
  template StatusOr<std::vector<uint16>> func;       \
  template StatusOr<std::vector<uint32>> func;       \
  template StatusOr<std::vector<uint64>> func;       \
  template StatusOr<std::vector<int8>> func;         \
  template StatusOr<std::vector<int16>> func;        \
  template StatusOr<std::vector<int32>> func;        \
  template StatusOr<std::vector<int64>> func;        \
  template StatusOr<std::vector<half>> func;         \
  template StatusOr<std::vector<bfloat16>> func;     \
  template StatusOr<std::vector<float>> func;        \
  template StatusOr<std::vector<double>> func;       \
  template StatusOr<std::vector<complex64>> func;    \
  template StatusOr<std::vector<bool>> func;

INITIALISE_FOR_ALL_NATIVE_VECTOR_TYPES(
    LiteralVectorToNativeType(const xla::Literal& lit));
INITIALISE_FOR_ALL_NATIVE_VECTOR_TYPES(
    WideConstToNativeType(const xla::HloInstruction* wide_const));

#undef INITIALISE_FOR_ALL_NATIVE_VECTOR_TYPES

bool IsPopOpsFusion(const xla::HloComputation* comp,
                    const std::string& postfix) {
  return comp->IsFusionComputation() &&
         tensorflow::str_util::StartsWith(comp->name(), "_pop_op_" + postfix);
}

bool IsPopOpsFusion(const xla::HloInstruction* inst,
                    const std::string& postfix) {
  return inst->opcode() == xla::HloOpcode::kFusion &&
         IsPopOpsFusion(inst->fused_instructions_computation(), postfix);
}

bool IsRepeatLoop(const xla::HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kCall) {
    auto statusor = inst->backend_config<PoplarBackendConfig>();
    if (statusor.ok()) {
      PoplarBackendConfig cfg = statusor.ValueOrDie();
      return cfg.repeat_config().is_repeat_loop();
    }
  }
  return false;
}

bool IsSupportedSharding(const HloSharding& sharding) {
  // We currently only support sharding with unique devices.
  return sharding.HasUniqueDevice();
}

bool IsInterIpuCopy(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kCustomCall &&
         inst->custom_call_target() == "inter_ipu_copy";
}

const HloInstruction* GetOperandLookThroughInterIpuCopy(
    const HloInstruction* inst, const int64 operand_idx) {
  const HloInstruction* operand = inst->operand(operand_idx);
  return IsInterIpuCopy(operand) ? operand->operand(0) : operand;
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

void GetAllDeps(const HloInstruction* base,
                std::vector<HloInstruction*>& deps) {
  for (auto* inst : base->operands()) {
    if (inst->opcode() != HloOpcode::kAfterAll) {
      deps.push_back(inst);
    } else {
      GetAllDeps(inst, deps);
    }
  }
}

void GetAllDepNames(const HloInstruction* base,
                    std::vector<std::string>& names) {
  std::vector<HloInstruction*> deps;
  GetAllDeps(base, deps);
  for (const auto* d : deps) {
    names.push_back(d->name());
  }
}

}  // namespace poplarplugin
}  // namespace xla
