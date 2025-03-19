## run_model

This is a simple tool to run a model with the CompiledModel API.

```
run_model --graph=<model_path>
```

### Use NPU via Dispatch API

If you're using the Dispatch API, you need to pass the Dispatch library
(libLiteRtDispatch_xxx.so) location via `--dispatch_library_dir`

```
run_model --graph=<model_path> --dispatch_library_dir=<dispatch_library_dir>
```

### Use GPU

If you run a model with GPU accelerator, use `--use_gpu` flag.

```
run_model --graph=<model_path> --use_gpu
```
