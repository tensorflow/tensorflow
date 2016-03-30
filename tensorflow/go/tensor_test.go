package tensorflow_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/tensorflow/tensorflow/tensorflow/go"
)

func getTensorFromGraph(t *testing.T, graphStr string) *tensorflow.Tensor {
	graph := &tensorflow.GraphDef{}
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
		t.FailNow()
	}

	return output[0]
}

func TestStrDecode(t *testing.T) {
	expectedResult := [][]byte{
		[]byte("Hello1!"),
		[]byte("Hello2!"),
		[]byte("Hello3!"),
	}
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
							dim {
								size: 3
							}
						}
						string_val: "%s"
						string_val: "%s"
						string_val: "%s"
					}
				}
			}
		}
		version: 5`, string(expectedResult[0]), string(expectedResult[1]), string(expectedResult[2])),
	)

	result, err := tensor.AsStr()
	if err != nil {
		t.Error("Problem trying to cast a tensor into string slice, Error:", err)
		t.FailNow()
	}

	if len(result) != 3 {
		t.Errorf("The expected number of strings returned was 3, but %d was returned", len(result))
		t.FailNow()
	}

	if !reflect.DeepEqual(expectedResult, result) {
		t.Errorf("The returned values doesn't coeesponds with the expected strings:", expectedResult, result)
		t.FailNow()
	}
}

func TestFloat32Decode(t *testing.T) {
	expectedResult := float32(10.23)
	tensor := getTensorFromGraph(t, fmt.Sprintf(`
		node {
			name: "output"
			op: "Const"
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
						float_val: %f
					}
				}
			}
		}
		version: 5`, expectedResult),
	)

	result, err := tensor.AsFloat32()
	if err != nil {
		t.Error("Problem trying to cast a tensor into float32 slice, Error:", err)
		t.FailNow()
	}

	if len(result) != 1 {
		t.Error("The expected length for the returned slice is 1 but the returned slice length was:", len(result))
		t.FailNow()
	}

	if result[0] != expectedResult {
		t.Errorf("The expected value is: %f, but the returned is: %f", expectedResult, result[0])
		t.FailNow()
	}
}

func TestFloat64Decode(t *testing.T) {
	expectedResult := float64(10.23)
	tensor := getTensorFromGraph(t, fmt.Sprintf(`
		node {
			name: "output"
			op: "Const"
			attr {
				key: "dtype"
				value {
					type: DT_DOUBLE
				}
			}
			attr {
				key: "value"
				value {
					tensor {
						dtype: DT_DOUBLE
						tensor_shape {
						}
						double_val: %f
					}
				}
			}
		}
		version: 5`, expectedResult),
	)

	result, err := tensor.AsFloat64()
	if err != nil {
		t.Error("Problem trying to cast a tensor into float64 slice, Error:", err)
		t.FailNow()
	}

	if len(result) != 1 {
		t.Error("The expected length for the returned slice is 1 but the returned slice length was:", len(result))
		t.FailNow()
	}

	if result[0] != expectedResult {
		t.Errorf("The expected value is: %f, but the returned is: %f", expectedResult, result[0])
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

func TestMultDimFloat32Decode(t *testing.T) {
	expectedResult := [][][]float32{
		{{1.00, 1.01}, {2.00, 2.01}},
		{{1.10, 1.11}, {2.10, 2.11}},
		{{1.00, 1.01}, {2.00, 2.01}},
	}
	tensor := getTensorFromGraph(t, `
		node {
			name: "output"
			op: "Const"
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
								size: 3
							}
							dim {
								size: 2
							}
							dim {
								size: 2
							}
						}
						tensor_content: "\000\000\200?\256G\201?\000\000\000@\327\243\000@\315\314\214?{\024\216?ff\006@=\n\007@\000\000\200?\256G\201?\000\000\000@\327\243\000@",
					}
				}
			}
		}
	`)

	result, err := tensor.AsFloat32()
	if err != nil {
		t.Error("Problem trying to cast a tensor into float32 slice, Error:", err)
		t.FailNow()
	}

	if len(result) != 12 {
		t.Error("The expected length for the returned slice is 12 but the returned slice length was:", len(result))
		t.FailNow()
	}

	for x := 0; x < len(expectedResult); x++ {
		for y := 0; y < len(expectedResult[x]); y++ {
			for z := 0; z < len(expectedResult[x][y]); z++ {
				value, err := tensor.GetVal(x, y, z)
				if err != nil {
					t.Error("Error returned when accessing to position:", x, y, z, "Error:", err)
				}
				valueFloat := value.(float32)
				if valueFloat != expectedResult[x][y][z] {
					t.Errorf(
						"The expected value for position: %d %d %d is: %f but the one returned was: %f",
						x,
						y,
						z,
						expectedResult[x][y][z],
						valueFloat)
					t.FailNow()
				}
			}
		}
	}
}

func TestUint8Decode(t *testing.T) {
	expectedResult := int32(21)
	tensor := getTensorFromGraph(t, fmt.Sprintf(`
		node {
			name: "output"
			op: "Const"
			attr {
				key: "dtype"
				value {
					type: DT_UINT8
				}
			}
			attr {
				key: "value"
				value {
					tensor {
						dtype: DT_UINT8
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
		t.Error("Problem trying to cast a tensor into uint8 slice, Error:", err)
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

/*func TestUint16Decode(t *testing.T) {
	expectedResult := uint16(321)
	tensor := getTensorFromGraph(t, fmt.Sprintf(`
		node {
			name: "output"
			op: "Const"
			attr {
				key: "dtype"
				value {
					type: DT_UINT16
				}
			}
			attr {
				key: "value"
				value {
					tensor {
						dtype: DT_UINT16
						tensor_shape {
						}
						int_val: %d
					}
				}
			}
		}
		version: 5`, expectedResult),
	)

	result, err := tensor.AsUint16()
	if err != nil {
		t.Error("Problem trying to cast a tensor into uint16 slice, Error:", err)
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
}*/

func TestInt16(t *testing.T) {
	expectedResult := int32(21)
	tensor := getTensorFromGraph(t, fmt.Sprintf(`
		node {
			name: "output"
			op: "Const"
			attr {
				key: "dtype"
				value {
					type: DT_INT16
				}
			}
			attr {
				key: "value"
				value {
					tensor {
						dtype: DT_INT16
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
		t.Error("Problem trying to cast a tensor into int16 slice, Error:", err)
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

func TestInt8(t *testing.T) {
	expectedResult := int32(21)
	tensor := getTensorFromGraph(t, fmt.Sprintf(`
		node {
			name: "output"
			op: "Const"
			attr {
				key: "dtype"
				value {
					type: DT_INT8
				}
			}
			attr {
				key: "value"
				value {
					tensor {
						dtype: DT_INT8
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
		t.Error("Problem trying to cast a tensor into int8 slice, Error:", err)
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

func TestBool(t *testing.T) {
	expectedResult := []bool{true, false, true, false}
	tensor := getTensorFromGraph(t, fmt.Sprintf(`
		node {
			name: "output"
			op: "Const"
			attr {
				key: "dtype"
				value {
					type: DT_BOOL
				}
			}
			attr {
				key: "value"
				value {
					tensor {
						dtype: DT_BOOL
						tensor_shape {
							dim {
								size: 4
							}
						}
						bool_val: %t
						bool_val: %t
						bool_val: %t
						bool_val: %t
					}
				}
			}
		}
		version: 5`, expectedResult[0], expectedResult[1], expectedResult[2], expectedResult[3]),
	)

	result, err := tensor.AsBool()
	if err != nil {
		t.Error("Problem trying to cast a tensor into bool slice, Error:", err)
		t.FailNow()
	}

	if len(result) != 4 {
		t.Error("The expected length for the returned slice is 4 but the returned slice length was:", len(result))
		t.FailNow()
	}

	for i, v := range expectedResult {
		if result[i] != v {
			t.Errorf("The expected value is: %d, but the returned is: %d", v, result[i])
			t.FailNow()
		}
	}
}

func TestConstant(t *testing.T) {
	tensor, err := tensorflow.Constant([][][]int64{
		{
			{10, 12},
			{14, 16},
		},
		{
			{18, 20},
			{22, 24},
		},
	})
	if err != nil {
		t.Error("Problem trying to instance the Constant, Error:", err)
		t.FailNow()
	}

	tensorToCompare := getTensorFromGraph(t, fmt.Sprintf(`
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
						tensor_content: "\n\000\000\000\000\000\000\000\014\000\000\000\000\000\000\000\016\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\022\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000"
					}
				}
			}
		}
		version: 5`),
	)

	tensorSlice, err := tensorToCompare.AsInt64()
	if err != nil {
		t.Error("Problem trying to get the tensor as slice of integers, Error:", err)
		t.FailNow()
	}

	resultSlice, err := tensor.AsInt64()
	if err != nil {
		t.Error("Problem trying to get the tensor as slice of integers, Error:", err)
		t.FailNow()
	}

	if !reflect.DeepEqual(tensorSlice, resultSlice) {
		t.Error("The returned values doesn't coeesponds with the expected strings:", tensorSlice, resultSlice)
		t.FailNow()
	}
}
