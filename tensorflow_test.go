package tensorflow_test

import (
	"fmt"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/golang/protobuf/proto"
	"github.com/tensorflow/tensorflow"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var exampleGraph = `node {
  name: "output1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "Hello, TensorFlow!"
      }
    }
  }
}
node {
  name: "Const_1"
  op: "Const"
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
        int_val: 10
      }
    }
  }
}
node {
  name: "Const_2"
  op: "Const"
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
        int_val: 32
      }
    }
  }
}
node {
  name: "output2"
  op: "Add"
  input: "Const_1"
  input: "Const_2"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
version: 5`

func TestNewSession(t *testing.T) {
	graph := &tf.GraphDef{}
	if err := proto.UnmarshalText(exampleGraph, graph); err != nil {
		t.Fatal(err)
	}
	s, err := tensorflow.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		t.Fatal(err)
	}
	output, err := s.Run(nil, []string{"output1", "output2"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(output)
	for _, out := range output {
		spew.Dump(out, out.Data())
	}
}
