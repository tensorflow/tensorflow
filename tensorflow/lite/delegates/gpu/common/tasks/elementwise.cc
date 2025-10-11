  }
  return absl::Substitute(result, output_value, input_value);
}
std::string GetTwoInputCode(const OperationType& op_type,
                            const std::string& result_var,
                            const std::string& input0,
                            const std::string& input1,
                            bool swap_inputs = false) {
  std::string result;
  switch (op_type) {
    // Patch by kshiteej-mali for GPU numerical accuracy
    // Explicitly use TO_FLOAT4 to ensure F32 precision and avoid F16 accumulation errors
    case OperationType::ADD:
      result += "$0 = TO_FLOAT4($1) + TO_FLOAT4($2);";
      break;
    case OperationType::DIV:
      result += "$0 = $1 / $2;";
      break;
    case OperationType::FLOOR_DIV:
      result = "$0 = floor($1 / $2);";
      break;
    case OperationType::FLOOR_MOD:
      result = "$0 = $1 - floor($1 / $2) * $2;";
      break;
    case OperationType::MAXIMUM:
      result += "$0 = max($1, $2);";
      break;
    case OperationType::MINIMUM:
      result += "$0 = min($1, $2);";
      break;
    // Patch by kshiteej-mali for GPU numerical accuracy
    // Explicitly use TO_FLOAT4 to ensure F32 precision and avoid F16 accumulation errors
    case OperationType::MUL:
      result += "$0 = TO_FLOAT4($1) * TO_FLOAT4($2);";
      break;
    case OperationType::POW:
      result += "$0 = pow($1, $2);";
      break;
    case OperationType::SQUARED_DIFF:
      result += "$0 = ($1 - $2) * ($1 - $2);";
      break;
    case OperationType::SUB:
      result += "$0 = $1 - $2;";
      break;
    // Comparison operators
    case OperationType::LESS:
      result = "$0.x = $1.x < $2.x;\n";
      result += "$0.y = $1.y < $2.y;\n";
      result += "$0.z = $1.z < $2.z;\n";
      result += "$0.w = $1.w < $2.w;";
      break;
    case OperationType::LESS_EQUAL:
      result = "$0.x = $1.x <= $2.x;\n";
      result += "$0.y = $1.y <= $2.y;\n";
      result += "$0.z = $1.z <= $2.z;\n";
      result += "$0.w = $1.w <= $2.w;";
      break;
    case OperationType::GREATER:
      result = "$0.x = $1.x > $2.x;\n";
      result += "$0.y = $1.y > $2.y;\n";
      result += "$0.z = $1.z > $2.z;\n";
      result += "$0.w = $1.w > $2.w;";
      break;
    case OperationType::GREATER_EQUAL:
      result = "$0.x = $1.x >= $2.x;\n";
      result += "$0.y = $1.y >= $2.y;\n";
      result += "$0.z = $1.z >= $2.z;\n";
      result += "$0.w = $1.w >= $2.w;";
      break;
    case OperationType::EQUAL:
      result = "$0.x = $1.x == $2.x;\n";
      result += "$0.y = $1.y == $2.y;\n";
      result += "$0.z = $1.z == $2.z;\n";
      result += "$0.w = $1.w == $2.w;";
      break;
    case OperationType::NOT_EQUAL:
      result = "$0.x = $1.x != $2.x;\n";
      result += "$0.y = $1.y != $2.y;\n";
      result += "$0.z = $1.z != $2.z;\n";
      result += "$0.w = $1.w != $2.w;";
      break;
    case OperationType::LOGICAL_AND:
      result = "$0.x = ($1.x != 0) && ($2.x != 0);\n";
      result += "$0.y = ($1.y != 0) && ($2.y != 0);\n";
      result += "$0.z = ($1.z != 0) && ($2.z != 0);\n";
      result += "$0.w = ($1.w != 0) && ($2.w != 0);";
      break;
    default:
      return "Unknown operation type;";
  }
