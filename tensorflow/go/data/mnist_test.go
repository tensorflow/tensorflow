// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package data

import (
	"testing"
)

func TestNewDataSet(t *testing.T) {
	const (
		trainSize = 60000
		testSize  = 10000
		imageSize = 28 * 28
	)

	ds := NewDataSet(".")
	if ds == nil {
		t.Fatal("Fail to create MNIST DataSet")
	}

	if len(ds.TrainImages) != trainSize {
		t.Fatalf("Fail to read train images: expected %d, actual %d", trainSize, len(ds.TrainImages))
	}

	if len(ds.TrainImages[0]) != imageSize {
		t.Fatalf("Fail to read train image: expected %d, actual %d", imageSize, len(ds.TrainImages[0]))
	}

	if len(ds.TrainLabels) != trainSize {
		t.Fatalf("Fail to read train labels: expected %d, actual %d", trainSize, len(ds.TrainImages))
	}

	if len(ds.TestImages) != testSize {
		t.Fatalf("Fail to read test images: expected %d, actual %d", testSize, len(ds.TrainImages))
	}

	if len(ds.TestLabels) != testSize {
		t.Fatalf("Fail to read test labels: expected %d, actual %d", testSize, len(ds.TrainImages))
	}
}
