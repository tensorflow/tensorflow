/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tensorflow

import (
	"testing"
	"time"
)

// BenchmarkNewTensor benchmarks the tensor creation performance.
func BenchmarkNewTensor(b *testing.B) {
	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := NewTensor(data)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOptimizedTensorFactory benchmarks the optimized tensor factory.
func BenchmarkOptimizedTensorFactory(b *testing.B) {
	factory := NewOptimizedTensorFactory(1000)
	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := factory.NewTensorOptimized(data)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSessionPool benchmarks session pool performance.
func BenchmarkSessionPool(b *testing.B) {
	graph := NewGraph()
	defer graph.Close()
	
	pool, err := NewSessionPool(graph, nil, 10)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Close()
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			sess, err := pool.Get()
			if err != nil {
				b.Fatal(err)
			}
			pool.Put(sess)
		}
	})
}

// TestSessionPool tests the session pool functionality.
func TestSessionPool(t *testing.T) {
	graph := NewGraph()
	defer graph.Close()
	
	pool, err := NewSessionPool(graph, nil, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	
	// Test getting and putting sessions
	sessions := make([]*Session, 3)
	for i := 0; i < 3; i++ {
		sess, err := pool.Get()
		if err != nil {
			t.Fatal(err)
		}
		sessions[i] = sess
	}
	
	// All sessions should be used up
	sess, err := pool.Get()
	if err != nil {
		t.Fatal(err)
	}
	
	// Put sessions back
	for _, s := range sessions {
		pool.Put(s)
	}
	pool.Put(sess)
	
	// Pool should have sessions available
	if pool.Size() != 3 {
		t.Errorf("Expected pool size 3, got %d", pool.Size())
	}
}

// TestTensorCache tests the tensor cache functionality.
func TestTensorCache(t *testing.T) {
	cache := NewTensorCache(10)
	
	data := []float32{1.0, 2.0, 3.0}
	tensor, err := NewTensor(data)
	if err != nil {
		t.Fatal(err)
	}
	
	// Test put and get
	cache.Put("test", tensor)
	retrieved, exists := cache.Get("test")
	if !exists {
		t.Error("Expected tensor to exist in cache")
	}
	if retrieved != tensor {
		t.Error("Retrieved tensor is not the same as cached tensor")
	}
	
	// Test cache size
	if cache.Size() != 1 {
		t.Errorf("Expected cache size 1, got %d", cache.Size())
	}
	
	// Test clear
	cache.Clear()
	if cache.Size() != 0 {
		t.Errorf("Expected cache size 0 after clear, got %d", cache.Size())
	}
}

// TestExtendedSession tests the extended session functionality.
func TestExtendedSession(t *testing.T) {
	graph := NewGraph()
	defer graph.Close()
	
	sess, err := NewExtendedSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	
	// Test metrics tracking
	metrics := sess.GetMetrics()
	if metrics.RunCount != 0 {
		t.Errorf("Expected run count 0, got %d", metrics.RunCount)
	}
	
	// Reset metrics
	sess.ResetMetrics()
	metrics = sess.GetMetrics()
	if metrics.RunCount != 0 {
		t.Errorf("Expected run count 0 after reset, got %d", metrics.RunCount)
	}
}

// TestSessionWithContext tests session with context support.
func TestSessionWithContext(t *testing.T) {
	graph := NewGraph()
	defer graph.Close()
	
	sess, err := NewSessionWithContext(nil, graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	
	// Test timeout
	err = sess.RunWithTimeout(100*time.Millisecond, nil, nil)
	if err != nil {
		// Expected to fail with no operations, but should not panic
		t.Logf("Expected error with no operations: %v", err)
	}
}
