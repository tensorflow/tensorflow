package tensorflow_test

import (
	"fmt"

	"github.com/tensorflow/tensorflow"
)

func ExampleNewSession() {
	s, err := tensorflow.NewSession()
	hello, _ := tensorflow.Constant("Hello, TensorFlow!")
	output, err := s.Run(map[string]*tensorflow.Tensor{
		"greeting": hello,
	}, []string{"Const"}, nil)
	fmt.Println(output, err)
}
