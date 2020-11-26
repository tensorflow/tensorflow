# TensorFlow Lite with Select TensorFlow ops

For enabling the Select TensorFlow ops for your TensorFlow Lite app, please add
the `TensorFlowLiteSelectTfOps` pod to your Podfile, in addition to
`TensorFlowLiteSwift` or `TensorFlowLiteObjC` pod, depending on your primary
language.

After that, you should also force load the framework from your project. Add the
following line to the `Other Linker Flags` under your project's Build Settings
page.

```
-force_load "$(PROJECT_DIR)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps"
```

Please refer to the [Select operators from TensorFlow][ops-select] guide for
more details.

[ops-select]: https://www.tensorflow.org/lite/guide/ops_select#ios
