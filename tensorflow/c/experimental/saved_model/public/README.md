# TensorFlow Saved Model C API

## Small ConcreteFunction Example

The following example loads a saved model from `"/path/to/model"` and
executes a function `f` taking no arguments and returning one single
value (error checking is omitted for simplicity):

```c
TF_Status* status = TF_NewStatus();
TFE_ContextOptions* ctx_options = TFE_NewContextOptions();
TFE_Context* ctx = TFE_NewContext(ctx_options, status);

TF_SavedModel* saved_model = TF_LoadSavedModel("/path/to/model", ctx, status);
TF_ConcreteFunction* f = TF_GetSavedModelConcreteFunction(saved_model, "f", status);
TFE_Op* op = TF_ConcreteFunctionMakeCallOp(f, NULL, 0, status);

TFE_TensorHandle* output;
int nouts = 1;
TFE_Execute(op, &output, &nouts, status);

TFE_DeleteTensorHandle(output);
TFE_DeleteOp(op);
TFE_DeleteSavedModel(saved_model);
TFE_DeleteContext(ctx);
TFE_DeleteContextOptions(ctx_options);
TF_DeleteStatus(status);
```
