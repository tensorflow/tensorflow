// Copyright 2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Simplified version of Google's command line flags.
// Does not support parsing the command line.
// If you want to do that, see
// https://gflags.github.io/gflags/

#ifndef CORE_PLAT_FLAGS_H__
#define CORE_PLAT_FLAGS_H__

#define DEFINE_flag(type, name, deflt, desc) \
        namespace re2 { type FLAGS_##name = deflt; }

#define DECLARE_flag(type, name) \
        namespace re2 { extern type FLAGS_##name; }

#define DEFINE_bool(name, deflt, desc) DEFINE_flag(bool, name, deflt, desc)
#define DEFINE_int32(name, deflt, desc) DEFINE_flag(int32, name, deflt, desc)
#define DEFINE_string(name, deflt, desc) DEFINE_flag(string, name, deflt, desc)

#define DECLARE_bool(name) DECLARE_flag(bool, name)
#define DECLARE_int32(name) DECLARE_flag(int32, name)
#define DECLARE_string(name) DECLARE_flag(string, name)

#endif  // CORE_PLAT_FLAGS_H__
