meta_info_def {
  stripped_op_list {
    op {
      name: "Assign"
      input_arg {
        name: "ref"
        type_attr: "T"
        is_ref: true
      }
      input_arg {
        name: "value"
        type_attr: "T"
      }
      output_arg {
        name: "output_ref"
        type_attr: "T"
        is_ref: true
      }
      attr {
        name: "T"
        type: "type"
      }
      attr {
        name: "validate_shape"
        type: "bool"
        default_value {
          b: true
        }
      }
      attr {
        name: "use_locking"
        type: "bool"
        default_value {
          b: true
        }
      }
      allows_uninitialized_input: true
    }
    op {
      name: "AssignAdd"
      input_arg {
        name: "ref"
        type_attr: "T"
        is_ref: true
      }
      input_arg {
        name: "value"
        type_attr: "T"
      }
      output_arg {
        name: "output_ref"
        type_attr: "T"
        is_ref: true
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_INT64
            type: DT_INT32
            type: DT_UINT8
            type: DT_UINT16
            type: DT_INT16
            type: DT_INT8
            type: DT_COMPLEX64
            type: DT_COMPLEX128
            type: DT_QINT8
            type: DT_QUINT8
            type: DT_QINT32
            type: DT_HALF
          }
        }
      }
      attr {
        name: "use_locking"
        type: "bool"
        default_value {
          b: false
        }
      }
    }
    op {
      name: "Cast"
      input_arg {
        name: "x"
        type_attr: "SrcT"
      }
      output_arg {
        name: "y"
        type_attr: "DstT"
      }
      attr {
        name: "SrcT"
        type: "type"
      }
      attr {
        name: "DstT"
        type: "type"
      }
    }
    op {
      name: "Const"
      output_arg {
        name: "output"
        type_attr: "dtype"
      }
      attr {
        name: "value"
        type: "tensor"
      }
      attr {
        name: "dtype"
        type: "type"
      }
    }
    op {
      name: "FIFOQueueV2"
      output_arg {
        name: "handle"
        type: DT_RESOURCE
      }
      attr {
        name: "component_types"
        type: "list(type)"
        has_minimum: true
        minimum: 1
      }
      attr {
        name: "shapes"
        type: "list(shape)"
        default_value {
          list {
          }
        }
        has_minimum: true
      }
      attr {
        name: "capacity"
        type: "int"
        default_value {
          i: -1
        }
      }
      attr {
        name: "container"
        type: "string"
        default_value {
          s: ""
        }
      }
      attr {
        name: "shared_name"
        type: "string"
        default_value {
          s: ""
        }
      }
      is_stateful: true
    }
    op {
      name: "Greater"
      input_arg {
        name: "x"
        type_attr: "T"
      }
      input_arg {
        name: "y"
        type_attr: "T"
      }
      output_arg {
        name: "z"
        type: DT_BOOL
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_INT32
            type: DT_INT64
            type: DT_UINT8
            type: DT_INT16
            type: DT_INT8
            type: DT_UINT16
            type: DT_HALF
          }
        }
      }
    }
    op {
      name: "Identity"
      input_arg {
        name: "input"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
      }
    }
    op {
      name: "NoOp"
    }
    op {
      name: "QueueDequeueV2"
      input_arg {
        name: "handle"
        type: DT_RESOURCE
      }
      output_arg {
        name: "components"
        type_list_attr: "component_types"
      }
      attr {
        name: "component_types"
        type: "list(type)"
        has_minimum: true
        minimum: 1
      }
      attr {
        name: "timeout_ms"
        type: "int"
        default_value {
          i: -1
        }
      }
      is_stateful: true
    }
    op {
      name: "QueueEnqueueV2"
      input_arg {
        name: "handle"
        type: DT_RESOURCE
      }
      input_arg {
        name: "components"
        type_list_attr: "Tcomponents"
      }
      attr {
        name: "Tcomponents"
        type: "list(type)"
        has_minimum: true
        minimum: 1
      }
      attr {
        name: "timeout_ms"
        type: "int"
        default_value {
          i: -1
        }
      }
      is_stateful: true
    }
    op {
      name: "RealDiv"
      input_arg {
        name: "x"
        type_attr: "T"
      }
      input_arg {
        name: "y"
        type_attr: "T"
      }
      output_arg {
        name: "z"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_UINT8
            type: DT_INT8
            type: DT_UINT16
            type: DT_INT16
            type: DT_INT32
            type: DT_INT64
            type: DT_COMPLEX64
            type: DT_COMPLEX128
          }
        }
      }
    }
    op {
      name: "Select"
      input_arg {
        name: "condition"
        type: DT_BOOL
      }
      input_arg {
        name: "t"
        type_attr: "T"
      }
      input_arg {
        name: "e"
        type_attr: "T"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "T"
        type: "type"
      }
    }
    op {
      name: "Sum"
      input_arg {
        name: "input"
        type_attr: "T"
      }
      input_arg {
        name: "reduction_indices"
        type_attr: "Tidx"
      }
      output_arg {
        name: "output"
        type_attr: "T"
      }
      attr {
        name: "keep_dims"
        type: "bool"
        default_value {
          b: false
        }
      }
      attr {
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_FLOAT
            type: DT_DOUBLE
            type: DT_INT64
            type: DT_INT32
            type: DT_UINT8
            type: DT_UINT16
            type: DT_INT16
            type: DT_INT8
            type: DT_COMPLEX64
            type: DT_COMPLEX128
            type: DT_QINT8
            type: DT_QUINT8
            type: DT_QINT32
            type: DT_HALF
          }
        }
      }
      attr {
        name: "Tidx"
        type: "type"
        default_value {
          type: DT_INT32
        }
        allowed_values {
          list {
            type: DT_INT32
            type: DT_INT64
          }
        }
      }
    }
    op {
      name: "VariableV2"
      output_arg {
        name: "ref"
        type_attr: "dtype"
        is_ref: true
      }
      attr {
        name: "shape"
        type: "shape"
      }
      attr {
        name: "dtype"
        type: "type"
      }
      attr {
        name: "container"
        type: "string"
        default_value {
          s: ""
        }
      }
      attr {
        name: "shared_name"
        type: "string"
        default_value {
          s: ""
        }
      }
      is_stateful: true
    }
  }
  tensorflow_version: "1.1.0-rc2"
  tensorflow_git_version: "unknown"
}
graph_def {
  node {
    name: "fifo_queue"
    op: "FIFOQueueV2"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "capacity"
      value {
        i: 4
      }
    }
    attr {
      key: "component_types"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "Const"
    op: "Const"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 1
            }
            dim {
              size: 2
            }
          }
          float_val: 0.
          float_val: 1.
        }
      }
    }
  }
  node {
    name: "fifo_queue_enqueue"
    op: "QueueEnqueueV2"
    input: "fifo_queue"
    input: "Const"
    device: "/device:CPU:0"
    attr {
      key: "Tcomponents"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
    attr {
      key: "timeout_ms"
      value {
        i: -1
      }
    }
  }
  node {
    name: "Const_1"
    op: "Const"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 1
            }
            dim {
              size: 2
            }
          }
          float_val: -4.2
          float_val: 9.1
        }
      }
    }
  }
  node {
    name: "fifo_queue_enqueue_1"
    op: "QueueEnqueueV2"
    input: "fifo_queue"
    input: "Const_1"
    device: "/device:CPU:0"
    attr {
      key: "Tcomponents"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
    attr {
      key: "timeout_ms"
      value {
        i: -1
      }
    }
  }
  node {
    name: "Const_2"
    op: "Const"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 1
            }
            dim {
              size: 2
            }
          }
          float_val: 6.5
          float_val: 0.
        }
      }
    }
  }
  node {
    name: "fifo_queue_enqueue_2"
    op: "QueueEnqueueV2"
    input: "fifo_queue"
    input: "Const_2"
    device: "/device:CPU:0"
    attr {
      key: "Tcomponents"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
    attr {
      key: "timeout_ms"
      value {
        i: -1
      }
    }
  }
  node {
    name: "Const_3"
    op: "Const"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: 1
            }
            dim {
              size: 2
            }
          }
          float_val: -3.2
          float_val: 0.
        }
      }
    }
  }
  node {
    name: "fifo_queue_enqueue_3"
    op: "QueueEnqueueV2"
    input: "fifo_queue"
    input: "Const_3"
    device: "/device:CPU:0"
    attr {
      key: "Tcomponents"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
    attr {
      key: "timeout_ms"
      value {
        i: -1
      }
    }
  }
  node {
    name: "fifo_queue_Dequeue"
    op: "QueueDequeueV2"
    input: "fifo_queue"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "component_types"
      value {
        list {
          type: DT_FLOAT
        }
      }
    }
    attr {
      key: "timeout_ms"
      value {
        i: -1
      }
    }
  }
  node {
    name: "mean/total/Initializer/zeros"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@mean/total"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.0
        }
      }
    }
  }
  node {
    name: "mean/total"
    op: "VariableV2"
    device: "/device:CPU:0"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@mean/total"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "mean/total/Assign"
    op: "Assign"
    input: "mean/total"
    input: "mean/total/Initializer/zeros"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@mean/total"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "mean/total/read"
    op: "Identity"
    input: "mean/total"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@mean/total"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "mean/count/Initializer/zeros"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@mean/count"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.0
        }
      }
    }
  }
  node {
    name: "mean/count"
    op: "VariableV2"
    device: "/device:CPU:0"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@mean/count"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
  node {
    name: "mean/count/Assign"
    op: "Assign"
    input: "mean/count"
    input: "mean/count/Initializer/zeros"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@mean/count"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: true
      }
    }
    attr {
      key: "validate_shape"
      value {
        b: true
      }
    }
  }
  node {
    name: "mean/count/read"
    op: "Identity"
    input: "mean/count"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@mean/count"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "mean/Size"
    op: "Const"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
          }
          int_val: 2
        }
      }
    }
  }
  node {
    name: "mean/ToFloat_1"
    op: "Cast"
    input: "mean/Size"
    device: "/device:CPU:0"
    attr {
      key: "DstT"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "SrcT"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "mean/Const"
    op: "Const"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 2
            }
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: 2
            }
          }
          int_val: 0
          int_val: 1
        }
      }
    }
  }
  node {
    name: "mean/Sum"
    op: "Sum"
    input: "fifo_queue_Dequeue"
    input: "mean/Const"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "Tidx"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "keep_dims"
      value {
        b: false
      }
    }
  }
  node {
    name: "mean/AssignAdd"
    op: "AssignAdd"
    input: "mean/total"
    input: "mean/Sum"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@mean/total"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "mean/AssignAdd_1"
    op: "AssignAdd"
    input: "mean/count"
    input: "mean/ToFloat_1"
    input: "^fifo_queue_Dequeue"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@mean/count"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "use_locking"
      value {
        b: false
      }
    }
  }
  node {
    name: "mean/Greater/y"
    op: "Const"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.0
        }
      }
    }
  }
  node {
    name: "mean/Greater"
    op: "Greater"
    input: "mean/count/read"
    input: "mean/Greater/y"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "mean/truediv"
    op: "RealDiv"
    input: "mean/total/read"
    input: "mean/count/read"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "mean/value/e"
    op: "Const"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.0
        }
      }
    }
  }
  node {
    name: "mean/value"
    op: "Select"
    input: "mean/Greater"
    input: "mean/truediv"
    input: "mean/value/e"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "mean/Greater_1/y"
    op: "Const"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.0
        }
      }
    }
  }
  node {
    name: "mean/Greater_1"
    op: "Greater"
    input: "mean/AssignAdd_1"
    input: "mean/Greater_1/y"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "mean/truediv_1"
    op: "RealDiv"
    input: "mean/AssignAdd"
    input: "mean/AssignAdd_1"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "mean/update_op/e"
    op: "Const"
    device: "/device:CPU:0"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.0
        }
      }
    }
  }
  node {
    name: "mean/update_op"
    op: "Select"
    input: "mean/Greater_1"
    input: "mean/truediv_1"
    input: "mean/update_op/e"
    device: "/device:CPU:0"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
  }
  node {
    name: "init"
    op: "NoOp"
    input: "^mean/total/Assign"
    input: "^mean/count/Assign"
    device: "/device:CPU:0"
  }
  versions {
    producer: 23
  }
}
collection_def {
  key: "local_variables"
  value {
    node_list {
      value: "mean/total:0"
      value: "mean/count:0"
    }
  }
}
collection_def {
  key: "update_op"
  value {
    node_list {
      value: "mean/update_op:0"
    }
  }
}
