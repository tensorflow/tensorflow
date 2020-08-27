# GPU delegate compatibility database

This package provides data and code for deciding if the GPU delegate is
supported on a specific Android device.

## Customizing the database

-   Convert from checked-in flatbuffer to json by running `flatc -t --raw-binary
    --strict-json database.fbs -- gpu_compatibility.bin`
-   Edit the json
-   Convert from json to flatbuffer `flatc -b database.fbs --
    gpu_compatibility.json`
-   Rebuild ../../../java:tensorflow-lite-gpu
