// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors

import (
	"errors"
	"strings"
	"testing"
)

func TestErrors(t *testing.T) {
	var sentinel = New("sentinel")
	var foreign = errors.New("foreign")
	for _, test := range []struct {
		what     string
		err      error
		wantText string
		is       []error
		isNot    []error
	}{{
		what:     `New("abc")`,
		err:      New("abc"),
		wantText: "abc",
	}, {
		what:     `New("%v", sentinel)`,
		err:      New("%v", sentinel),
		wantText: "sentinel",
		isNot:    []error{sentinel},
	}, {
		what:     `Wrap(sentinel, "%v", "text")`,
		err:      Wrap(sentinel, "%v", "text"),
		wantText: "text: sentinel",
		is:       []error{sentinel},
	}, {
		what:     `New("%v", foreign)`,
		err:      New("%v", foreign),
		wantText: "foreign",
		isNot:    []error{foreign},
	}, {
		what:     `Wrap(foreign, "%v", "text")`,
		err:      Wrap(foreign, "%v", "text"),
		wantText: "text: foreign",
		is:       []error{foreign},
	}} {
		if got, want := test.err.Error(), prefix; !strings.HasPrefix(got, want) {
			t.Errorf("%v.Error() = %q, want prefix %q", test.what, got, want)
		}
		if got, want := test.err.Error(), prefix+test.wantText; got != want {
			t.Errorf("%v.Error() = %q, want %q", test.what, got, want)
		}
		if got, want := Is(test.err, Error), true; got != want {
			t.Errorf("errors.Is(%v, errors.Error) = %v, want %v", test.what, got, want)
		}
		for _, err := range test.is {
			if got, want := Is(test.err, err), true; got != want {
				t.Errorf("errors.Is(%v, %v) = %v, want %v", test.what, err, got, want)
			}
		}
		for _, err := range test.isNot {
			if got, want := Is(test.err, err), false; got != want {
				t.Errorf("errors.Is(%v, %v) = %v, want %v", test.what, err, got, want)
			}
		}
	}
}
