// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench_test

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	benchpb "google.golang.org/protobuf/internal/testprotos/benchmarks"
	_ "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message1/proto2"
	_ "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message1/proto3"
	_ "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message2"
	_ "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message3"
	_ "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message4"
)

func BenchmarkWire(b *testing.B) {
	bench(b, "Unmarshal", func(ds dataset, pb *testing.PB) {
		for pb.Next() {
			for _, p := range ds.wire {
				m := ds.messageType.New().Interface()
				if err := proto.Unmarshal(p, m); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	bench(b, "Marshal", func(ds dataset, pb *testing.PB) {
		for pb.Next() {
			for _, m := range ds.messages {
				if _, err := proto.Marshal(m); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	bench(b, "Size", func(ds dataset, pb *testing.PB) {
		for pb.Next() {
			for _, m := range ds.messages {
				proto.Size(m)
			}
		}
	})
}

func BenchmarkText(b *testing.B) {
	bench(b, "Unmarshal", func(ds dataset, pb *testing.PB) {
		for pb.Next() {
			for _, p := range ds.text {
				m := ds.messageType.New().Interface()
				if err := prototext.Unmarshal(p, m); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	bench(b, "Marshal", func(ds dataset, pb *testing.PB) {
		for pb.Next() {
			for _, m := range ds.messages {
				if _, err := prototext.Marshal(m); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}

func BenchmarkJSON(b *testing.B) {
	bench(b, "Unmarshal", func(ds dataset, pb *testing.PB) {
		for pb.Next() {
			for _, p := range ds.json {
				m := ds.messageType.New().Interface()
				if err := protojson.Unmarshal(p, m); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	bench(b, "Marshal", func(ds dataset, pb *testing.PB) {
		for pb.Next() {
			for _, m := range ds.messages {
				if _, err := protojson.Marshal(m); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}

func Benchmark(b *testing.B) {
	bench(b, "Clone", func(ds dataset, pb *testing.PB) {
		for pb.Next() {
			for _, src := range ds.messages {
				proto.Clone(src)
			}
		}
	})
}

func bench(b *testing.B, name string, f func(dataset, *testing.PB)) {
	b.Helper()
	b.Run(name, func(b *testing.B) {
		for _, ds := range datasets {
			b.Run(ds.name, func(b *testing.B) {
				b.RunParallel(func(pb *testing.PB) {
					f(ds, pb)
				})
			})
		}
	})
}

type dataset struct {
	name        string
	messageType protoreflect.MessageType
	messages    []proto.Message
	wire        [][]byte
	text        [][]byte
	json        [][]byte
}

var datasets []dataset

func TestMain(m *testing.M) {
	// Load benchmark data early, to avoid including this step in -cpuprofile/-memprofile.
	//
	// For the larger benchmark datasets (not downloaded by default), preparing
	// this data is quite expensive. In addition, keeping the unmarshaled messages
	// in memory makes GC scans a substantial fraction of runtime CPU cost.
	//
	// It would be nice to avoid loading the data we aren't going to use. Unfortunately,
	// there isn't any simple way to tell what benchmarks are going to run; we can examine
	// the -test.bench flag, but parsing it is quite complicated.
	flag.Parse()
	if v := flag.Lookup("test.bench").Value.(flag.Getter).Get(); v == "" {
		// Don't bother loading data if we aren't going to run any benchmarks.
		// Avoids slowing down go test ./...
		return
	}
	if v := flag.Lookup("test.timeout").Value.(flag.Getter).Get().(time.Duration); v != 0 && v <= 10*time.Minute {
		// The default test timeout of 10m is too short if running all the benchmarks.
		// It's quite frustrating to discover this 10m through a benchmark run, so
		// catch the condition.
		//
		// The -timeout and -test.timeout flags are handled by the go command, which
		// forwards them along to the test binary, so we can't just set the default
		// to something reasonable; the go command will override it with its default.
		// We also can't ignore the timeout, because the go command kills a test which
		// runs more than a minute past its deadline.
		fmt.Fprintf(os.Stderr, "Test timeout of %v is probably too short; set -test.timeout=0.\n", v)
		os.Exit(1)
	}
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").CombinedOutput()
	if err != nil {
		panic(err)
	}
	repoRoot := strings.TrimSpace(string(out))
	dataDir := filepath.Join(repoRoot, ".cache", "benchdata")
	filepath.Walk(dataDir, func(path string, _ os.FileInfo, _ error) error {
		if filepath.Ext(path) != ".pb" {
			return nil
		}
		raw, err := ioutil.ReadFile(path)
		if err != nil {
			panic(err)
		}
		dspb := &benchpb.BenchmarkDataset{}
		if err := proto.Unmarshal(raw, dspb); err != nil {
			panic(err)
		}
		mt, err := protoregistry.GlobalTypes.FindMessageByName(protoreflect.FullName(dspb.MessageName))
		if err != nil {
			panic(err)
		}
		ds := dataset{
			name:        dspb.Name,
			messageType: mt,
			wire:        dspb.Payload,
		}
		for _, payload := range dspb.Payload {
			m := mt.New().Interface()
			if err := proto.Unmarshal(payload, m); err != nil {
				panic(err)
			}
			ds.messages = append(ds.messages, m)
			b, err := prototext.Marshal(m)
			if err != nil {
				panic(err)
			}
			ds.text = append(ds.text, b)
			b, err = protojson.Marshal(m)
			if err != nil {
				panic(err)
			}
			ds.json = append(ds.json, b)
		}
		datasets = append(datasets, ds)
		return nil
	})
	os.Exit(m.Run())
}
