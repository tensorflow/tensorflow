import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import tensorflow as tf
interpreter = tf.lite.Interpreter(
    model_path="content/retrain_logs/conv.tflite", model_content=None, experimental_delegates=None,
)
print(interpreter.get_input_details())
for tensor_detail in interpreter.get_tensor_details():
    print(tensor_detail)

interpreter.allocate_tensors()
print(interpreter.get_input_details()[0]["index"])
print(interpreter.get_input_details()[1]["index"])
print(interpreter.get_output_details()[0]["index"])
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

import numpy as np
from scipy.io import wavfile

sampling_rate = 44100
freq = 440
samples = 44100

samplerate, data = wavfile.read("data/bulyiya/불이야_강상훈_1.wav")
print(samplerate, data.shape, data.dtype, data)
input_data = np.array(data[:16000]/32767.0, dtype=np.float32).reshape((16000, 1))
print(input_data.shape)
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
interpreter.set_tensor(interpreter.get_input_details()[1]['index'], np.int32(16000))

interpreter.invoke()
print("inference", interpreter.get_tensor(interpreter.get_output_details()[0]['index']))