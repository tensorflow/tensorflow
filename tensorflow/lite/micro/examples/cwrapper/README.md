This is a simple project that can be used to generate 
a library file that can be included in your c projects.

Modify the micro_api.cc to suit your specific application. 

To generate the library file

`make lib`

you can link to libtensorflow-microlite.a and micro_api.h to use your model 
in your project

you can use the parse_micro_ops_function.py  in order to pull out of a tf_lite model
only the necessary functions. 

`python parse_micro_ops_functions.py --model_file=../micro_speech/micro_features/tiny_conv_micro_features_model_data.cc --model_name=g_tiny_conv_micro_features_model_data`

which will overwwrite the micro_api.cc file with a file containing only the operations
that need to be loaded by the micro_mutable_ops_resolver. You can then build a much
smaller library file to include in your project.

to include all ops

`python parse_micro_ops_functions.py`

to include ops from a binary directly

`python parse_micro_ops_functions.py --model_binary=<model_binary>`

you can generate the binary string directly from your tensorflow model

```python
import binascii
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()
binascii.hexlify(tflite_model).decode('ascii')
```








