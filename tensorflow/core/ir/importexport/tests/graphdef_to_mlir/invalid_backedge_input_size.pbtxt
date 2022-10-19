# RUN: not tfg-translate -graphdef-to-mlir %s 2>&1 | FileCheck %s

# CHECK: Type attr not found

library {
  function {
    signature {
      name: "\344\264\264"
      description: "value"
      is_distributed_communication: true
    }
    node_def {
      op: "Const"
      input: "|"
    }
    control_ret {
      key: ""
      value: ""
    }
  }
}