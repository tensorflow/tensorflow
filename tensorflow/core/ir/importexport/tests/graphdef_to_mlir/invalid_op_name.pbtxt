# RUN: not tfg-translate -graphdef-to-mlir %s 2>&1 | FileCheck %s

# CHECK: Node  has an empty op name

library {
  function {
    signature {
      name: "\\344\\264\\264"
      description: "value"
      is_distributed_communication: true
    }
    node_def {
      input: "|"
    }
    control_ret {
      key: ""
      value: ""
    }
  }
}