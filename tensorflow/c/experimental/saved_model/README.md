# Tensorflow C SavedModel API

## Overview

These are the new experimental C SavedModel APIs for loading and running
SavedModels in a TF2-idiomatic fashion. See
[RFC 207](https://github.com/tensorflow/community/pull/207) for additional
context.

The directory structure is as follows:

```none
saved_model/

  public/

  internal/

  core/

```

## saved_model/public

`saved_model/public` is intended to house *only the public headers* of the
SavedModel C API.

These headers:

1. declare opaque C types (like `TF_SavedModel`),

2. declare the functions that operate on these types (like `TF_LoadSavedModel`).

Once they leave experimental, these APIs should be considered stable for use
by external clients.

These headers are in a separate directory to make it obvious to clients which
headers they should depend on, and which headers are implementation details.
Separating these public headers by directory also allow future programmatic
checks to ensure that TF public headers only `#include` other public TF headers.

## saved_model/internal

`saved_model/internal` is the "glue" between the C API and the internal C++
implementation.

Its role is to:

1. implement the C API functions declared in `saved_model/public`

2. define the C API types declared in `saved_model/public`

The files fulfilling 1. are named `*.cc` (eg: `concrete_function.cc`), while
the files fulfilling 2. are `*type.h` (eg: `concrete_function_type.h`).

The headers exposing the internal implementation of the opaque C types are only
visible to other implementors of the C API. This is similar to how other
TF C API implementations use `tf_status_internal.h` (to extract the underlying
`tensorflow::Status`). All other targets in this directory are private.

## saved_model/core

`saved_model/core` contains pure C++ "Classes" underlying the C API types
in `saved_model/public/`. These are implementation
details subject to change, and have limited visibility to implementors only.
This is the bottom-most layer of the `C++ -> C -> C++` sandwich.
