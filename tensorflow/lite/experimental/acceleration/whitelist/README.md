# GPU delegate whitelist

This package provides data and code for deciding if the GPU delegate is
supported on a specific Android device.

## Customizing the GPU whitelist

-   Convert from checked-in flatbuffer to json by running `flatc -t --raw-binary
    --strict-json database.fbs -- gpu_whitelist.bin`
-   Edit the json
-   Convert from json to flatbuffer `flatc -b database.fbs --
    gpu_whitelist.json`
-   Rebuild ../../../java:tensorflow-lite-gpu
