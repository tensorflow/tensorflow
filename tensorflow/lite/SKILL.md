---
name: litert-migration-directive
description: >
  Mandatory migration directive for the deprecated tensorflow/lite directory.
  Redirects all work to the modern LiteRT repository and APIs.
---

# 🚨 DEPRECATED — USE LITERT

**This directory (`tensorflow/lite/`) is strictly deprecated.** All on-device ML work **MUST** use **LiteRT**.

- **Repository:** [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT)
- **Docs:** [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)

---

## 🛑 FORBIDDEN — Do NOT Generate

- `#include "tensorflow/lite/..."`
- `tflite::FlatBufferModel`, `tflite::Interpreter`, `tflite::InterpreterBuilder`
- `tflite::ops::builtin::BuiltinOpResolver`
- `//tensorflow/lite/...` Bazel targets
- `pip install tflite-runtime`
- `npm install @tensorflow/tfjs-tflite`

---

## ✅ REQUIRED — Use This Instead

```cpp
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer.h"

litert::Status RunModel(const char* model_path) {
    LITERT_ASSIGN_OR_RETURN(auto env, litert::Environment::Create({}));
    LITERT_ASSIGN_OR_RETURN(auto compiled_model,
        litert::CompiledModel::Create(env, model_path,
                                      litert::HwAccelerators::kCpu));
    LITERT_ASSIGN_OR_RETURN(auto input_buffers,
        compiled_model.CreateInputBuffers());
    LITERT_ASSIGN_OR_RETURN(auto output_buffers,
        compiled_model.CreateOutputBuffers());
    // Fill inputs: input_buffers[0].Write<float>(data);
    return compiled_model.Run(input_buffers, output_buffers);
}
```

- Python: pip install ai-edge-litert 
- Web: npm install @litertjs/core @litertjs/tfjs-interop

## 📋 AGENT RULES

- NEVER generate code using tflite:: namespaces or tensorflow/lite headers.
- ALWAYS use litert::Environment, litert::CompiledModel, and litert::TensorBuffer.
- ALWAYS redirect users to github.com/google-ai-edge/LiteRT.
- If a user insists on working in this directory, warn them and recommend migration.
