# TSL BuildData API

This directory contains the `tsl::builddata` namespace, providing access to
build-time information (e.g., build timestamp, changelist number, build host,
and client status) embedded directly in the binary.

It is designed to be compatible with Bazel environments.

## How to Use the API

### 1. Update BUILD Dependencies

Add the `builddata` library to your target's dependencies in the `BUILD` file:

```python
cc_binary(
    name = "my_binary",
    srcs = ["my_binary.cc"],
    deps = [
        "//tensorflow/compiler/xla/tsl/builddata",
    ],
)
```

### 2. Include Header and Query BuildData

Include the header and invoke the functions in the `tsl::builddata` namespace:

```cpp
#include <iostream>
#include "third_party/tensorflow/compiler/xla/tsl/builddata/builddata.h"

int main() {
  std::cout << "Build Timestamp: " << tsl::builddata::Timestamp() << std::endl;
  std::cout << "Source Revision: " << tsl::builddata::SourceRevision() << std::endl;
  std::cout << "Client Status: " << tsl::builddata::ClientStatusAsString() << std::endl;
  return 0;
}
```

--------------------------------------------------------------------------------

## Build Stamping in Bazel

By default, Bazel builds binaries **without stamping** (`--nostamp`) to maximize
build cache hits. In unstamped mode, the timestamp defaults to Epoch 0 (`Dec 31
1969 16:00:00 (0)`), and other SCM variables will show up as `"unknown"` or
`"-1"`.

To embed the actual build-time information, you must compile/link the target
with the `--stamp` flag:

```bash
bazel build --stamp //third_party/tensorflow/compiler/xla/tsl/builddata:my_binary
```

--------------------------------------------------------------------------------

## Workspace Status Command

To feed SCM (Source Control Management) information (like Git revision and
status) into the stamped build, Bazel relies on a workspace status script.

### 1. Using the Provided Git Script

A helper script is provided in this directory: `workspace_status_command.sh`. It
detects the current Git revision and whether there are local uncommitted
modifications.

To use it, pass the `--workspace_status_command` flag to Bazel:

```bash
bazel build --stamp \
  --workspace_status_command=third_party/tensorflow/compiler/xla/tsl/builddata/workspace_status_command.sh \
  //third_party/tensorflow/compiler/xla/tsl/builddata:my_binary
```

### 2. Supplying a Custom Script

If you are using a different SCM system (e.g., Mercurial/Fig) or want to inject
custom build metadata, you can supply your own script.

The script must print space-separated key-value pairs to stdout. The `builddata`
library looks for the following keys:

*   `BUILD_SCM_REVISION` (String representing SCM revision, e.g. git hash or
    changelist)
*   `BUILD_SCM_STATUS` (Either `"mint"` for clean workspaces, or `"modified"` if
    there are uncommitted changes)

Example custom script:

```bash
#!/bin/bash
echo "BUILD_SCM_REVISION $(my_scm_tool revision)"
echo "BUILD_SCM_STATUS $(my_scm_tool status_is_clean && echo 'mint' || echo 'modified')"
```
