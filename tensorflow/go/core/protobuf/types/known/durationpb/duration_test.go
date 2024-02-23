// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package durationpb_test

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/protobuf/internal/detrand"
	"google.golang.org/protobuf/testing/protocmp"

	durpb "google.golang.org/protobuf/types/known/durationpb"
)

func init() {
	detrand.Disable()
}

const (
	minGoSeconds = math.MinInt64 / int64(1e9)
	maxGoSeconds = math.MaxInt64 / int64(1e9)
	absSeconds   = 315576000000 // 10000yr * 365.25day/yr * 24hr/day * 60min/hr * 60sec/min
)

func TestToDuration(t *testing.T) {
	tests := []struct {
		in   time.Duration
		want *durpb.Duration
	}{
		{in: time.Duration(0), want: &durpb.Duration{Seconds: 0, Nanos: 0}},
		{in: -time.Second, want: &durpb.Duration{Seconds: -1, Nanos: 0}},
		{in: +time.Second, want: &durpb.Duration{Seconds: +1, Nanos: 0}},
		{in: -time.Second - time.Millisecond, want: &durpb.Duration{Seconds: -1, Nanos: -1e6}},
		{in: +time.Second + time.Millisecond, want: &durpb.Duration{Seconds: +1, Nanos: +1e6}},
		{in: time.Duration(math.MinInt64), want: &durpb.Duration{Seconds: minGoSeconds, Nanos: int32(math.MinInt64 - 1e9*minGoSeconds)}},
		{in: time.Duration(math.MaxInt64), want: &durpb.Duration{Seconds: maxGoSeconds, Nanos: int32(math.MaxInt64 - 1e9*maxGoSeconds)}},
	}

	for _, tt := range tests {
		got := durpb.New(tt.in)
		if diff := cmp.Diff(tt.want, got, protocmp.Transform()); diff != "" {
			t.Errorf("New(%v) mismatch (-want +got):\n%s", tt.in, diff)
		}
	}
}

func TestFromDuration(t *testing.T) {
	tests := []struct {
		in      *durpb.Duration
		wantDur time.Duration
		wantErr error
	}{
		{in: nil, wantDur: time.Duration(0), wantErr: textError("invalid nil Duration")},
		{in: new(durpb.Duration), wantDur: time.Duration(0)},
		{in: &durpb.Duration{Seconds: -1, Nanos: 0}, wantDur: -time.Second},
		{in: &durpb.Duration{Seconds: +1, Nanos: 0}, wantDur: +time.Second},
		{in: &durpb.Duration{Seconds: 0, Nanos: -1}, wantDur: -time.Nanosecond},
		{in: &durpb.Duration{Seconds: 0, Nanos: +1}, wantDur: +time.Nanosecond},
		{in: &durpb.Duration{Seconds: -100, Nanos: 0}, wantDur: -100 * time.Second},
		{in: &durpb.Duration{Seconds: +100, Nanos: 0}, wantDur: +100 * time.Second},
		{in: &durpb.Duration{Seconds: -100, Nanos: -987}, wantDur: -100*time.Second - 987*time.Nanosecond},
		{in: &durpb.Duration{Seconds: +100, Nanos: +987}, wantDur: +100*time.Second + 987*time.Nanosecond},
		{in: &durpb.Duration{Seconds: minGoSeconds, Nanos: int32(math.MinInt64 - 1e9*minGoSeconds)}, wantDur: time.Duration(math.MinInt64)},
		{in: &durpb.Duration{Seconds: maxGoSeconds, Nanos: int32(math.MaxInt64 - 1e9*maxGoSeconds)}, wantDur: time.Duration(math.MaxInt64)},
		{in: &durpb.Duration{Seconds: minGoSeconds - 1, Nanos: int32(math.MinInt64 - 1e9*minGoSeconds)}, wantDur: time.Duration(math.MinInt64)},
		{in: &durpb.Duration{Seconds: maxGoSeconds + 1, Nanos: int32(math.MaxInt64 - 1e9*maxGoSeconds)}, wantDur: time.Duration(math.MaxInt64)},
		{in: &durpb.Duration{Seconds: minGoSeconds, Nanos: int32(math.MinInt64-1e9*minGoSeconds) - 1}, wantDur: time.Duration(math.MinInt64)},
		{in: &durpb.Duration{Seconds: maxGoSeconds, Nanos: int32(math.MaxInt64-1e9*maxGoSeconds) + 1}, wantDur: time.Duration(math.MaxInt64)},
		{in: &durpb.Duration{Seconds: -123, Nanos: +456}, wantDur: -123*time.Second + 456*time.Nanosecond, wantErr: textError("duration (seconds:-123 nanos:456) has seconds and nanos with different signs")},
		{in: &durpb.Duration{Seconds: +123, Nanos: -456}, wantDur: +123*time.Second - 456*time.Nanosecond, wantErr: textError("duration (seconds:123 nanos:-456) has seconds and nanos with different signs")},
		{in: &durpb.Duration{Seconds: math.MinInt64}, wantDur: time.Duration(math.MinInt64), wantErr: textError("duration (seconds:-9223372036854775808) exceeds -10000 years")},
		{in: &durpb.Duration{Seconds: math.MaxInt64}, wantDur: time.Duration(math.MaxInt64), wantErr: textError("duration (seconds:9223372036854775807) exceeds +10000 years")},
		{in: &durpb.Duration{Seconds: -absSeconds, Nanos: -(1e9 - 1)}, wantDur: time.Duration(math.MinInt64)},
		{in: &durpb.Duration{Seconds: +absSeconds, Nanos: +(1e9 - 1)}, wantDur: time.Duration(math.MaxInt64)},
		{in: &durpb.Duration{Seconds: -absSeconds - 1, Nanos: 0}, wantDur: time.Duration(math.MinInt64), wantErr: textError("duration (seconds:-315576000001) exceeds -10000 years")},
		{in: &durpb.Duration{Seconds: +absSeconds + 1, Nanos: 0}, wantDur: time.Duration(math.MaxInt64), wantErr: textError("duration (seconds:315576000001) exceeds +10000 years")},
	}

	for _, tt := range tests {
		gotDur := tt.in.AsDuration()
		if diff := cmp.Diff(tt.wantDur, gotDur); diff != "" {
			t.Errorf("AsDuration(%v) mismatch (-want +got):\n%s", tt.in, diff)
		}
		gotErr := tt.in.CheckValid()
		if diff := cmp.Diff(tt.wantErr, gotErr, cmpopts.EquateErrors()); diff != "" {
			t.Errorf("CheckValid(%v) mismatch (-want +got):\n%s", tt.in, diff)
		}
	}
}

type textError string

func (e textError) Error() string     { return string(e) }
func (e textError) Is(err error) bool { return err != nil && strings.Contains(err.Error(), e.Error()) }
