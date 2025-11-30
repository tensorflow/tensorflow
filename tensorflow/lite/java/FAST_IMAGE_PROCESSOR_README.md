FastImageProcessor — minimal notes

One-line summary

ImageProcessor is too slow for 1920×1080 FP32 images because it allocates temporaries and does per-op Java work for ~2M pixels. `FastImageProcessor` uses a reusable direct ByteBuffer/FloatBuffer and native `getPixels()`/`setPixels()` calls to eliminate allocations and per-pixel Java dispatch.

When to use

- Use for large FP32 models (e.g., full-HD 1920×1080) where Android `ImageProcessor` causes heavy per-frame allocations and CPU overhead.
- If your model accepts UINT8 inputs/outputs and quality is acceptable, prefer the quantized path (see below).

How it works (short)

- Allocate once at startup: `ByteBuffer.allocateDirect(width * height * channels * 4).order(ByteOrder.nativeOrder())`.
- Preprocess with `bitmap.getPixels()` into an `IntArray`, convert to normalized RGB floats in a tight loop writing into a `FloatBuffer`.
- Feed the interpreter from the direct `ByteBuffer`/`FloatBuffer`.
- Postprocess by converting model floats to ARGB ints in a tight loop and call `bitmap.setPixels()` once.
- Reuse the buffers and bitmaps across frames to avoid allocations.

Usage example (Kotlin)

```kotlin
val processor = FastImageProcessor(width = 1920, height = 1080)
// Preprocess
processor.preprocess(inputBitmap)
val inputBuffer = processor.getInputBuffer() // pass to Interpreter
// Run inference
interpreter.run(inputBuffer, outputBuffer) // depends on interpreter API
// Postprocess
processor.postprocess(outputFloatBuffer, outputBitmap)
```

Benchmarking & validation

- Add timers for: preprocess, inference, postprocess.
- Expected outcome: preprocess+postprocess drops from ~150–180ms to <40ms (device dependent). Inference (~90ms) remains.
- Check Android Studio profiler for allocations (should be near-zero during steady-state).

Optional optimizations

- UINT8 quantized model: if acceptable, copy pixels directly with `bitmap.copyPixelsToBuffer()` and avoid float conversions entirely.
- JNI native loops: if Java loops are still a bottleneck, move the conversion loops to native C/C++.

PR description snippet

Add `FastImageProcessor.kt` that:
- Allocates a direct ByteBuffer once and exposes a FloatBuffer view.
- Uses `bitmap.getPixels()` → FloatBuffer conversion loop for preprocess.
- Uses FloatBuffer → IntArray → `bitmap.setPixels()` for postprocess.
- Reuses buffers/bitmaps across frames and documents when to use it and the optional UINT8 workflow.

Notes

The implementation intentionally keeps logic minimal and dependency-free so it can be copied into Android apps or TFLite example modules. If you want, I can add a small benchmark Activity in the examples module to demonstrate the speedup on device.
