#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/register.h"

int main()
{
	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::FlatBufferModel> model;
    return 0;
}
