package tensorflow_test

import (
	"fmt"
	"testing"

	//"github.com/davecgh/go-spew/spew"
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
	outputs := []string{
		"output1",
		"output2",
	}

	output, err := s.Run(nil, outputs, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(output) != len(outputs) {
		t.Fatal("There was", len(outputs), "expected outputs, but:", len(output), "obtained")
	}
}

func getTensorFromGraph(t *testing.T, graphStr string) *tensorflow.Tensor {
	graph := &tf.GraphDef{}
	if err := proto.UnmarshalText(graphStr, graph); err != nil {
		t.Fatal(err)
	}
	s, err := tensorflow.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		t.Fatal(err)
	}

	output, err := s.Run(nil, []string{"output"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(output) != 1 {
		t.Fatalf("The expexted number of tensors is 1 but there was %d tensors returned", len(output))
	}
	return output[0]
}

func TestStrDecode(t *testing.T) {
	expectedResult := "Hello!"
	tensor := getTensorFromGraph(t, fmt.Sprintf(`
		node {
		  name: "output"
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
			string_val: "%s"
		      }
		    }
		  }
		}
		version: 5`, expectedResult),
	)

	result, err := tensor.AsStr()
	if err != nil {
		t.Error("Problem trying to cast a tensor into string slice, Error:", err)
		t.FailNow()
	}

	fmt.Println([]byte(expectedResult))
	fmt.Println([]byte(result))
	if result != expectedResult {
		t.Errorf("The expected value is: %s, but the returned is: %s", expectedResult, result)
		t.FailNow()
	}
}

func TestInt32Decode(t *testing.T) {
	expectedResult := int32(123)
	tensor := getTensorFromGraph(t, fmt.Sprintf(`
		node {
		  name: "output"
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
			int_val: %d
		      }
		    }
		  }
		}
		version: 5`, expectedResult),
	)

	result, err := tensor.AsInt32()
	if err != nil {
		t.Error("Problem trying to cast a tensor into int32 slice, Error:", err)
		t.FailNow()
	}

	if len(result) != 1 {
		t.Error("The expected length for the returned slice is 1 but the returned slice length was:", len(result))
		t.FailNow()
	}

	if result[0] != expectedResult {
		t.Errorf("The expected value is: %d, but the returned is: %d", expectedResult, result[0])
		t.FailNow()
	}
}

func TestInt64Decode(t *testing.T) {
	expectedResult := int64(123)
	tensor := getTensorFromGraph(t, fmt.Sprintf(`
		node {
		  name: "output"
		  op: "Const"
		  attr {
		    key: "dtype"
		    value {
		      type: DT_INT64
		    }
		  }
		  attr {
		    key: "value"
		    value {
		      tensor {
			dtype: DT_INT64
			tensor_shape {
			}
			int64_val: %d
		      }
		    }
		  }
		}
		version: 5`, expectedResult),
	)

	result, err := tensor.AsInt64()
	if err != nil {
		t.Error("Problem trying to cast a tensor into int64 slice, Error:", err)
		t.FailNow()
	}

	if len(result) != 1 {
		t.Error("The expected length for the returned slice is 1 but the returned slice length was:", len(result))
		t.FailNow()
	}

	if result[0] != expectedResult {
		t.Errorf("The expected value is: %d, but the returned is: %d", expectedResult, result[0])
		t.FailNow()
	}
}
