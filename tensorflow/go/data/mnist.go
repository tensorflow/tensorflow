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
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
)

type MnistImage []byte
type MnistLabel uint8

// DataSet stores MNIST image, label data for both training/testing.
type DataSet struct {
	pos         int
	TrainImages []MnistImage
	TrainLabels []MnistLabel
	TestImages  []MnistImage
	TestLabels  []MnistLabel
}

const (
	url            = "http://yann.lecun.com/exdb/mnist/"
	trainImageFile = "train-images-idx3-ubyte.gz"
	trainLabelFile = "train-labels-idx1-ubyte.gz"
	testImageFile  = "t10k-images-idx3-ubyte.gz"
	testLabelFile  = "t10k-labels-idx1-ubyte.gz"
	imageMagic     = 0x00000803
	labelMagic     = 0x00000801
	Width          = 28
	Height         = 28
)

func maybeDownload(dirname string, filename string) {
	filepath := fmt.Sprintf("%s/%s", dirname, filename)
	_, err := os.Stat(filepath)
	if err == nil {
		log.Println(filename, "is already downloaded")
		return
	}
	out, err := os.Create(filepath)
	if err != nil {
		log.Fatal("Fail to create:", err)
		return
	}

	log.Println("Downloading", filename)
	res, err := http.Get(url + filename)
	if err != nil {
		log.Fatal(err)
	}
	defer res.Body.Close()
	n, err := io.Copy(out, res.Body)
	if err != nil || n == 0 {
		log.Fatal("Fail to write file:", err)
		return
	}
}

func extractImages(dirname string, filename string) (images []MnistImage, err error) {
	name := fmt.Sprintf("%s/%s", dirname, filename)
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}

	var (
		magic int32
		n     int32
		nrows int32
		ncols int32
	)

	if err = binary.Read(z, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {
		return nil, os.ErrInvalid
	}
	if err = binary.Read(z, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	if err = binary.Read(z, binary.BigEndian, &nrows); err != nil {
		return nil, err
	}
	if err = binary.Read(z, binary.BigEndian, &ncols); err != nil {
		return nil, err
	}

	images = make([]MnistImage, n)
	size := nrows * ncols
	for i := 0; i < int(n); i++ {
		images[i] = make(MnistImage, size)
		size_, err := io.ReadFull(z, images[i])
		if err != nil {
			return nil, err
		}
		if size_ != int(size) {
			return nil, os.ErrInvalid
		}
	}

	return images, nil
}

func extractLabels(dirname string, filename string) (labels []MnistLabel, err error) {
	name := fmt.Sprintf("%s/%s", dirname, filename)
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	var (
		magic int32
		n     int32
	)

	if err = binary.Read(z, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, os.ErrInvalid
	}
	if err = binary.Read(z, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels = make([]MnistLabel, n)
	for i := 0; i < int(n); i++ {
		var l MnistLabel
		if err := binary.Read(z, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = l
	}
	return labels, nil
}

// Create MNIST data set
//
// Arguments:
//   dirname: Directory where MNIST raw file is downloaded
//
// Returns DataSet which includes TrainImages,TrainLabels,TestImages,TestLabels
func NewDataSet(dirname string) *DataSet {
	files := []string{
		trainImageFile,
		trainLabelFile,
		testImageFile,
		testLabelFile,
	}

	for _, f := range files {
		maybeDownload(dirname, f)
	}

	trainImages, err := extractImages(dirname, trainImageFile)
	if err != nil {
		log.Fatal(err)
		return nil
	}
	trainLabels, err := extractLabels(dirname, trainLabelFile)
	if err != nil {
		log.Fatal(err)
		return nil
	}
	testImages, err := extractImages(dirname, testImageFile)
	if err != nil {
		log.Fatal(err)
		return nil
	}
	testLabels, err := extractLabels(dirname, testLabelFile)
	if err != nil {
		log.Fatal(err)
		return nil
	}

	return &DataSet{
		0,
		trainImages,
		trainLabels,
		testImages,
		testLabels,
	}
}

func (ds *DataSet) updatePos(max int) {
	ds.pos += max
}

// Returns the next training batch
//
// Arguments:
//   max: The max samples of next batch
//
func (ds *DataSet) NextBatch(max int) ([]MnistImage, []MnistLabel) {
	if len(ds.TrainImages) < ds.pos {
		// All samples were read
		return nil, nil
	}
	length := int(math.Min(float64(max), float64(ds.pos+max)))
	defer ds.updatePos(length)
	return ds.TrainImages[ds.pos:length], ds.TrainLabels[ds.pos:length]
}
