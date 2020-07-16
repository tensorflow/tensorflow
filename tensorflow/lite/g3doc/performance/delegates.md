# TensorFlow Lite delegates

Note: Delegate API is still experimental and is subject to change.

## What is a TensorFlow Lite delegate?

A TensorFlow Lite delegate is a way to delegate part or all of graph execution
to another executor.

## Why should I use delegates?

Running inference on compute-heavy machine learning models on mobile devices is
resource demanding due to the devices' limited processing and power.

Instead of relying on the CPU, some devices have hardware accelerators, such as
GPU or DSP, that allows for better performance and higher energy efficiency.

## Using the built-in delegates

TensorFlow Lite provides the following delegates for hardware acceleration:

*   **GPU delegate for cross platform acceleration** - The GPU delegate can be
    used on both Android and iOS. It is optimized to run 32-bit and 16-bit float
    based models where a GPU is available. For an overview of the GPU delegate,
    see [TensorFlow Lite on GPU](gpu_advanced.md). For step-by-step tutorials on
    using the GPU delegate with Android and iOS, see
    [TensorFlow Lite GPU Delegate Tutorial](gpu.md).
*   **NNAPI delegate for newer Android devices** - The NNAPI delegate can be
    used to accelerate models on Android devices with GPU, DSP and / or NPU
    available. It is available in Android 8.1 (API 27+) or higher. For an
    overview of the NNAPI delegate, step-by-step instructions and best
    practices, see [TensorFlow Lite NNAPI delegate](nnapi.md).
*   **Hexagon delegate for older Android devices** - The Hexagon delegate can be
    used to accelerate models on Android devices with Qualcomm Hexagon DSP. It
    can be used on devices older version of Android OS that does not fully
    support NNAPI. See [TensorFlow Lite Hexagon delegate](hexagon_delegate.md)
    for more detail.
*   **Core ML delegate for newer iPhones and iPads** - For newer iPhones and
    iPads where Neural Engine is available, you can use Core ML delegate to
    accelerate inference for 32-bit float based models. Neural Engine is
    available Apple mobile devices with A12 SoC or higher. For an overview of
    the Core ML delegate and step-by-step instructions, see
    [TensorFlow Lite Core ML delegate](coreml_delegate.md).

## How do delegates work?

Let's say we have a simple model graph such as the following:

![Original graph](../images/performance/tflite_delegate_graph_1.png "Original Graph")

If a delegate was provided for specific operations, then TensorFlow Lite will
split the graph into multiple subgraphs where each subgraph will be handled by a
delegate.

Let's assume that a delegate, `MyDelegate`, has a faster implementation for
Conv2D and Mean operations. The resulting main graph will be updated to look
like below.

![Graph with delegate](../images/performance/tflite_delegate_graph_2.png "Graph with delegate")

Each subgraph that is handled by a delegate will be replaced with a node that
evaluates the subgraph on its invoked call.

Depending on the model, the final graph can end up with one node, which means
that all of the graphs were delegated or multiple nodes handled the subgraphs.
In general, you don't want to have multiple subgraphs handled by the delegate,
since each time you switch from delegate to the main graph, there is an overhead
for passing the results from the subgraph to the main graph. It's not always
safe to share memory.

## How to add a delegate

_Note that the API used below is experimental and is subject to change._

Based on the previous section, to add a delegate, we need to do the following:

1.  Define a kernel node that is responsible for evaluating the delegate
    subgraph.
1.  Create an instance of
    [TfLiteDelegate](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L611),
    which is responsible for registering the kernel node and claiming the nodes
    that the delegate can execute.

To see it in code, let's define a delegate and call it `MyDelegate`, which can
execute Conv2D and Mean operations faster.

```c++
#include "tensorflow/lite/util.h"

// This is where the execution of the operations or whole graph happens.
// The class below has an empty implementation just as a guideline
// on the structure.
class MyDelegate {
 public:
  // Returns true if my delegate can handle this type of op.
  static bool SupportedOp(const TfLiteRegistration* registration) {
    switch (registration->builtin_code) {
      case kTfLiteBuiltinConv2d:
      case kTfLiteBuiltinMean:
        return true;
      default:
        return false;
    }
  }

  // Any initialization code needed
  bool Init() {}
  // Any preparation work needed (e.g. allocate buffers)
  bool Prepare(TfLiteContext* context, TfLiteNode* node) {}
  // Actual running of the delegate subgraph.
  bool Invoke(TfLiteContext* context, TfLiteNode* node) {}
  // ... Add any other methods needed.
};

// Create the TfLiteRegistration for the Kernel node which will replace
// the subgraph in the main TfLite graph.
TfLiteRegistration GetMyDelegateNodeRegistration() {
  // This is the registration for the Delegate Node that gets added to
  // the TFLite graph instead of the subgraph it replaces.
  // It is treated as an OP node. But in our case
  // Init will initialize the delegate.
  // Invoke will run the delegate graph.
  // Prepare for preparing the delegate.
  // Free for any cleaning needed by the delegate.
  TfLiteRegistration kernel_registration;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = "MyDelegate";
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<MyDelegate*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                   size_t) -> void* {
    // In the node init phase, initialize MyDelegate instance
    const TfLiteDelegateParams* delegate_params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    MyDelegate* my_delegate = new MyDelegate;
    if (!my_delegate->Init(context, params)) {
      return nullptr;
    }
    return my_delegate;
  };
  kernel_registration.invoke = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    MyDelegate* kernel = reinterpret_cast<MyDelegate*>(node->user_data);
    return kernel->Invoke(context, node);
  };
  kernel_registration.prepare = [](TfLiteContext* context,
                                    TfLiteNode* node) -> TfLiteStatus {
    MyDelegate* kernel = reinterpret_cast<MyDelegate*>(node->user_data);
    return kernel->Prepare(context, node);
  };

  return kernel_registration;
}

// TfLiteDelegate methods

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  // Claim all nodes that can be evaluated by the delegate and ask the
  // framework to update the graph with delegate kernel instead.
  std::vector<int> supported_nodes;
  TfLiteIntArray* plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));
  TfLiteNode* node;
  TfLiteRegistration* registration;
  for (int node_index : TfLiteIntArrayView(plan)) {
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (MyDelegate::SupportedOp(registration)) {
      supported_nodes.push_back(node_index);
    }
  }
  TfLiteRegistration my_delegate_kernel_registration =
      GetMyDelegateNodeRegistration();

  // This call split the graphs into subgraphs, for subgraphs that can be
  // handled by the delegate, it will replace it with a
  // 'my_delegate_kernel_registration'
  TfLiteIntArray* supported_nodes_int_array =
      ::tflite::ConvertVectorToTfLiteIntArray(supported_nodes);
  auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, my_delegate_kernel_registration,
      supported_nodes_int_array, delegate);
  TfLiteIntArrayFree(supported_nodes_int_array);
  return status
}

void FreeBufferHandle(TfLiteContext* context, TfLiteDelegate* delegate,
                      TfLiteBufferHandle* handle) {
  // Do any cleanups.
}

TfLiteStatus CopyToBufferHandle(TfLiteContext* context,
                                TfLiteDelegate* delegate,
                                TfLiteBufferHandle buffer_handle,
                                TfLiteTensor* tensor) {
  // Copies data from tensor to delegate buffer if needed.
  return kTfLiteOk;
}

TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  TfLiteBufferHandle buffer_handle,
                                  TfLiteTensor* tensor) {
  // Copies the data from delegate buffer into the tensor raw memory.
  return kTfLiteOk;
}

// Caller takes ownership of the returned pointer.
TfLiteDelegate* CreateMyDelegate() {
  TfLiteDelegate* delegate = new TfLiteDelegate;

  delegate->data_ = nullptr;
  delegate->flags = kTfLiteDelegateFlagsNone;
  delegate->Prepare = &DelegatePrepare;
  // This cannot be null.
  delegate->CopyFromBufferHandle = &CopyFromBufferHandle;
  // This can be null.
  delegate->CopyToBufferHandle = &CopyToBufferHandle;
  // This can be null.
  delegate->FreeBufferHandle = &FreeBufferHandle;

  return delegate;
}


// To add the delegate you need to call

auto* my_delegate = CreateMyDelegate();
if (interpreter->ModifyGraphWithDelegate(my_delegate) !=
        kTfLiteOk) {
  // Handle error
} else {
  interpreter->Invoke();
}
...
// Don't forget to delete your delegate
delete my_delegate;
```
