//go:build go1.21
// +build go1.21

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
	"context"
	"time"
)

// SessionWithContext extends Session with context support for cancellation and timeouts.
type SessionWithContext struct {
	*Session
}

// NewSessionWithContext creates a new session with context support.
func NewSessionWithContext(ctx context.Context, graph *Graph, options *SessionOptions) (*SessionWithContext, error) {
	sess, err := NewSession(graph, options)
	if err != nil {
		return nil, err
	}
	
	return &SessionWithContext{Session: sess}, nil
}

// RunWithContext executes the graph computation with context support.
func (s *SessionWithContext) RunWithContext(ctx context.Context, fetches, outputs []Output) error {
	// Check if context is already cancelled
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	
	// Create a channel to receive the result
	done := make(chan error, 1)
	
	go func() {
		done <- s.Run(fetches, outputs)
	}()
	
	// Wait for either completion or cancellation
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// RunWithTimeout executes the graph computation with a timeout.
func (s *SessionWithContext) RunWithTimeout(timeout time.Duration, fetches, outputs []Output) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return s.RunWithContext(ctx, fetches, outputs)
}

// ExtendedSession provides additional functionality for better TensorFlow integration.
type ExtendedSession struct {
	*Session
	metrics *SessionMetrics
}

// SessionMetrics tracks session performance metrics.
type SessionMetrics struct {
	RunCount    int64
	TotalTime   time.Duration
	ErrorCount  int64
	LastRunTime time.Duration
}

// NewExtendedSession creates a new extended session with metrics tracking.
func NewExtendedSession(graph *Graph, options *SessionOptions) (*ExtendedSession, error) {
	sess, err := NewSession(graph, options)
	if err != nil {
		return nil, err
	}
	
	return &ExtendedSession{
		Session: sess,
		metrics: &SessionMetrics{},
	}, nil
}

// RunWithMetrics executes the graph computation and tracks performance metrics.
func (s *ExtendedSession) RunWithMetrics(fetches, outputs []Output) error {
	start := time.Now()
	err := s.Run(fetches, outputs)
	duration := time.Since(start)
	
	s.metrics.RunCount++
	s.metrics.TotalTime += duration
	s.metrics.LastRunTime = duration
	
	if err != nil {
		s.metrics.ErrorCount++
	}
	
	return err
}

// GetMetrics returns the current session metrics.
func (s *ExtendedSession) GetMetrics() *SessionMetrics {
	return s.metrics
}

// ResetMetrics resets all session metrics.
func (s *ExtendedSession) ResetMetrics() {
	s.metrics = &SessionMetrics{}
}
