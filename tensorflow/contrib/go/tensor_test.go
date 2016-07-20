package tensorflow_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/contrib/go"
)

func getTensorFromGraph(t *testing.T, dType, shapeVal string) *tf.Tensor {
	graph, err := tf.NewGraphFromReader(strings.NewReader(fmt.Sprintf(`
		node {
			name: "output"
			op: "Const"
			attr {
				key: "dtype"
				value {
					type: %s
				}
			}
			attr {
				key: "value"
				value {
					tensor {
						dtype: %s
						%s
					}
				}
			}
		}
		version: 5`,
		dType, dType, shapeVal)), true)
	if err != nil {
		t.Fatal(err)
	}
	s, err := tf.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		t.Fatal(err)
	}

	output, err := s.Run(nil, []string{"output"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(output) != 1 {
		t.Fatalf("Expexted 1 tensor, got: %d tensors", len(output))
	}

	return output[0]
}

func TestStrDecode(t *testing.T) {
	expectedResult := [][]byte{
		[]byte("Hello1!"),
		[]byte("Hello2!"),
		[]byte("Hello3!"),
	}
	tensor := getTensorFromGraph(t, "DT_STRING", fmt.Sprintf(`
		tensor_shape {
			dim {
				size: 3
			}
		}
		string_val: "%s"
		string_val: "%s"
		string_val: "%s"
	`, string(expectedResult[0]), string(expectedResult[1]), string(expectedResult[2])))

	result, err := tensor.ByteSlices()
	if err != nil {
		t.Fatal("Error casting Tensor into string slice:", err)
	}

	if len(result) != 3 {
		t.Fatal("Expected number of strings 3, got:", len(result))
	}

	if !reflect.DeepEqual(expectedResult, result) {
		t.Fatal("Expected:", expectedResult, ", got:", result)
	}
}

func TestFloat32Decode(t *testing.T) {
	expectedResult := float32(10.23)
	tensor := getTensorFromGraph(t, "DT_FLOAT", fmt.Sprintf(`
		float_val: %f`, expectedResult))

	result, err := tensor.Float32s()
	if err != nil {
		t.Fatal("Error casting Tensor into float32 slice:", err)
	}

	if len(result) != 1 {
		t.Fatal("Expected length for the returned slice: 1, got:", len(result))
	}

	if result[0] != expectedResult {
		t.Fatalf("Expected value: %f, got: %f", expectedResult, result[0])
	}
}

func TestFloat64Decode(t *testing.T) {
	expectedResult := float64(10.23)
	tensor := getTensorFromGraph(t, "DT_DOUBLE", fmt.Sprintf(`double_val: %f`, expectedResult))

	result, err := tensor.Float64s()
	if err != nil {
		t.Fatal("Error casting Tensor into float64 slice:", err)
	}

	if len(result) != 1 {
		t.Fatal("Expected length for the returned slice: 1, got:", len(result))
	}

	if result[0] != expectedResult {
		t.Fatalf("Expected value is: %f, got: %f", expectedResult, result[0])
	}
}

func TestInt32Decode(t *testing.T) {
	expectedResult := int32(123)
	tensor := getTensorFromGraph(t, "DT_INT32", fmt.Sprintf(`int_val: %d`, expectedResult))

	result, err := tensor.Int32s()
	if err != nil {
		t.Fatal("Error casting Tensor into int32 slice:", err)
	}

	if len(result) != 1 {
		t.Fatal("Expected length for returned slice: 1, got:", len(result))
	}

	if result[0] != expectedResult {
		t.Fatalf("Expected value: %d, got: %d", expectedResult, result[0])
	}
}

func TestInt64Decode(t *testing.T) {
	expectedResult := int64(123)
	tensor := getTensorFromGraph(t, "DT_INT64", fmt.Sprintf(`int64_val: %d`, expectedResult))

	result, err := tensor.Int64s()
	if err != nil {
		t.Fatal("Error casting Tensor into int64 slice:", err)
	}

	if len(result) != 1 {
		t.Fatal("Expected length for the returned slice: 1, got:", len(result))
	}

	if result[0] != expectedResult {
		t.Fatalf("Expected value: %d, got: %d", expectedResult, result[0])
	}
}

func TestMultDimFloat32Decode(t *testing.T) {
	expectedResult := [][][]float32{
		{{1.00, 1.01}, {2.00, 2.01}},
		{{1.10, 1.11}, {2.10, 2.11}},
		{{1.00, 1.01}, {2.00, 2.01}},
	}
	tensor := getTensorFromGraph(t, "DT_FLOAT", `
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
	`)

	result, err := tensor.Float32s()
	if err != nil {
		t.Fatal("Error casting Tensor into float32 slice:", err)
	}

	if len(result) != 12 {
		t.Fatal("Expected length for the returned slice: 12, got:", len(result))
	}

	for x := 0; x < len(expectedResult); x++ {
		for y := 0; y < len(expectedResult[x]); y++ {
			for z := 0; z < len(expectedResult[x][y]); z++ {
				value, err := tensor.GetVal(int64(x), int64(y), int64(z))
				if err != nil {
					t.Error("Error returned when accessing to position:", x, y, z, "Error:", err)
				}
				valueFloat := value.(float32)
				if valueFloat != expectedResult[x][y][z] {
					t.Fatalf(
						"Expected value for position: %d %d %d: %f, got: %f",
						x,
						y,
						z,
						expectedResult[x][y][z],
						valueFloat)
				}
			}
		}
	}
	_, err = tensor.GetVal(1, 0, 2)
	if err == nil {
		t.Error("Expected ErrIndexOutOfRange")
	}
	_, err = tensor.GetVal(999, 999, 999)
	if err == nil {
		t.Error("Expected ErrIndexOutOfRange")
	}
}

func TestUint8Decode(t *testing.T) {
	expectedResult := int32(21)
	tensor := getTensorFromGraph(t, "DT_UINT8", fmt.Sprintf(`int_val: %d`, expectedResult))

	result, err := tensor.Int32s()
	if err != nil {
		t.Fatal("Error casting Tensor into uint8 slice:", err)
	}

	if len(result) != 1 {
		t.Fatal("Expected length for returned slice: 1, got:", len(result))
	}

	if result[0] != expectedResult {
		t.Fatalf("Expected value: %d, got: %d", expectedResult, result[0])
	}
}

func TestInt16(t *testing.T) {
	expectedResult := int32(21)
	tensor := getTensorFromGraph(t, "DT_INT16", fmt.Sprintf(`int_val: %d`, expectedResult))

	result, err := tensor.Int32s()
	if err != nil {
		t.Fatal("Error casting Tensor into int16 slice:", err)
	}

	if len(result) != 1 {
		t.Fatal("Expected length for the returned slice is: 1, got:", len(result))
	}

	if result[0] != expectedResult {
		t.Fatalf("Expected value: %d, got: %d", expectedResult, result[0])
	}
}

func TestInt8(t *testing.T) {
	expectedResult := int32(21)
	tensor := getTensorFromGraph(t, "DT_INT8", fmt.Sprintf(`int_val: %d`, expectedResult))

	result, err := tensor.Int32s()
	if err != nil {
		t.Fatal("Error casting Tensor into int8 slice:", err)
	}

	if len(result) != 1 {
		t.Fatal("Expected length for the returned slice: 1, got:", len(result))
	}

	if result[0] != expectedResult {
		t.Fatalf("Expected value: %d, got: %d", expectedResult, result[0])
	}
}

func TestBools(t *testing.T) {
	expectedResult := []bool{true, false, true, false}
	tensor := getTensorFromGraph(t, "DT_BOOL", fmt.Sprintf(`
		tensor_shape {
			dim {
				size: 4
			}
		}
		bool_val: %t
		bool_val: %t
		bool_val: %t
		bool_val: %t
	`,
		expectedResult[0],
		expectedResult[1],
		expectedResult[2],
		expectedResult[3]),
	)

	result, err := tensor.Bools()
	if err != nil {
		t.Fatal("Error casting Tensor into boolean slice:", err)
	}

	if len(result) != 4 {
		t.Fatal("Expected length for returned slice: 4, got:", len(result))
	}

	for i, v := range expectedResult {
		if result[i] != v {
			t.Fatalf("Expected value: %d, got: %d", v, result[i])
		}
	}
}

func TestTensor(t *testing.T) {
	tensor, err := tf.NewTensor([][][]int64{
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
		t.Fatal("Error instancing Tensor:", err)
	}

	tensorToCompare := getTensorFromGraph(t, "DT_INT64", fmt.Sprintf(`
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
	`))

	tensorSlice, err := tensorToCompare.Int64s()
	if err != nil {
		t.Fatal("Error getting Tensor as integer slice:", err)
	}

	resultSlice, err := tensor.Int64s()
	if err != nil {
		t.Fatal("Error getting Tensor as integer slice:", err)
	}

	if !reflect.DeepEqual(tensorSlice, resultSlice) {
		t.Fatal("Expected:", tensorSlice, ", got:", resultSlice)
	}
}
