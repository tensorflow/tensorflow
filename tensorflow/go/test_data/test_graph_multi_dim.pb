node {
  name: "input1"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2
        }
        dim {
          size: 2
        }
        dim {
          size: 2
        }
      }
    }
  }
}
node {
  name: "input2"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2
        }
        dim {
          size: 2
        }
        dim {
          size: 2
        }
      }
    }
  }
}
node {
  name: "output"
  op: "Add"
  input: "input1"
  input: "input2"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
}
