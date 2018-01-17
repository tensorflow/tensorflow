#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/register.h"

const char *tflite_model_path = "C:\\temp\\test_model\\wtf.lite";
#include <iostream>

int main()
{
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(tflite_model_path);

	std::unique_ptr<tflite::Interpreter> interpreter;

	tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

	TfLiteStatus status = interpreter->AllocateTensors();

    interpreter->SetNumThreads(1);

    float *input = interpreter->typed_input_tensor<float>(0);
	const int N = 784;
	for (int i = 0; i < 1; ++i)
	{
		
		for (size_t j = 0; j < N; j++)
		{
			input[j] = (float)1.0f;
		}
		status = interpreter->Invoke();
		if (status == kTfLiteError)
		{

		}

		float* output = interpreter->typed_output_tensor<float>(0);
		std::cout << *output << "\t" << *(output+1) << std::endl;

	}



	getchar();
    return 0;
}
