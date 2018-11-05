# Optimizing for mobile

Warning: We expect to deprecate TensorFlow Mobile in early 2019

<div class="caution">
  <p>
    <a href="../">TensorFlow Lite</a> is our main mobile and embedded offering. We are
    working hard to close the feature gap between TensorFlow Mobile and
    TensorFlow Lite. We expect to deprecate TensorFlow Mobile in early 2019. We
    will give ample notice to our users when we get to that point and will
    provide help and support to ensure easy migrations.
  </p>
  <p>
    In the meantime, please use TensorFlow Lite. If you have a feature request,
    such as a missing op, please post to our <a
    href="https://github.com/tensorflow/tensorflow/issues">GitHub</a>.
  </p>
</div>

There are some special issues that you have to deal with when you’re trying to
ship on mobile or embedded devices, and you’ll need to think about these as
you’re developing your model.

These issues are:

- Model and Binary Size
- App speed and model loading speed
- Performance and threading

We'll discuss a few of these below.

## What are the minimum device requirements for TensorFlow?

You need at least one megabyte of program memory and several megabytes of RAM to
run the base TensorFlow runtime, so it’s not suitable for DSPs or
microcontrollers. Other than those, the biggest constraint is usually the
calculation speed of the device, and whether you can run the model you need for
your application with a low enough latency. You can use the benchmarking tools
in [How to Profile your Model](#how_to_profile_your_model) to get an idea of how
many FLOPs are required for a model, and then use that to make rule-of-thumb
estimates of how fast they will run on different devices. For example, a modern
smartphone might be able to run 10 GFLOPs per second, so the best you could hope
for from a 5 GFLOP model is two frames per second, though you may do worse
depending on what the exact computation patterns are.

This model dependence means that it’s possible to run TensorFlow even on very
old or constrained phones, as long as you optimize your network to fit within
the latency budget and possibly within limited RAM too. For memory usage, you
mostly need to make sure that the intermediate buffers that TensorFlow creates
aren’t too large, which you can examine in the benchmark output too.

## Speed

One of the highest priorities of most model deployments is figuring out how to
run the inference fast enough to give a good user experience. The first place to
start is by looking at the total number of floating point operations that are
required to execute the graph. You can get a very rough estimate of this by
using the `benchmark_model` tool:

    bazel build -c opt tensorflow/tools/benchmark:benchmark_model && \
    bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph=/tmp/inception_graph.pb --input_layer="Mul:0" \
    --input_layer_shape="1,299,299,3" --input_layer_type="float" \
    --output_layer="softmax:0" --show_run_order=false --show_time=false \
    --show_memory=false --show_summary=true --show_flops=true --logtostderr

This should show you an estimate of how many operations are needed to run the
graph. You can then use that information to figure out how feasible your model
is to run on the devices you’re targeting. For an example, a high-end phone from
2016 might be able to do 20 billion FLOPs per second, so the best speed you
could hope for from a model that requires 10 billion FLOPs is around 500ms. On a
device like the Raspberry Pi 3 that can do about 5 billion FLOPs, you may only
get one inference every two seconds.

Having this estimate helps you plan for what you’ll be able to realistically
achieve on a device. If the model is using too many ops, then there are a lot of
opportunities to optimize the architecture to reduce that number.

Advanced techniques include [SqueezeNet](https://arxiv.org/abs/1602.07360)
and [MobileNet](https://arxiv.org/abs/1704.04861), which are architectures
designed to produce models for mobile -- lean and fast but with a small accuracy
cost.  You can also just look at alternative models, even older ones, which may
be smaller. For example, Inception v1 only has around 7 million parameters,
compared to Inception v3’s 24 million, and requires only 3 billion FLOPs rather
than 9 billion for v3.

## Model Size

Models that run on a device need to be stored somewhere on the device, and very
large neural networks can be hundreds of megabytes. Most users are reluctant to
download very large app bundles from app stores, so you want to make your model
as small as possible. Furthermore, smaller neural networks can persist in and
out of a mobile device's memory faster.

To understand how large your network will be on disk, start by looking at the
size on disk of your `GraphDef` file after you’ve run `freeze_graph` and
`strip_unused_nodes` on it (see <a href="./prepare_models.md">Preparing models</a> for
more details on these tools), since then it should only contain
inference-related nodes. To double-check that your results are as expected, run
the `summarize_graph` tool to see how many parameters are in constants:

    bazel build tensorflow/tools/graph_transforms:summarize_graph && \
    bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
    --in_graph=/tmp/tensorflow_inception_graph.pb

That command should give you output that looks something like this:

    No inputs spotted.
    Found 1 possible outputs: (name=softmax, op=Softmax)
    Found 23885411 (23.89M) const parameters, 0 (0) variable parameters,
    and 99 control_edges
    Op types used: 489 Const, 99 CheckNumerics, 99 Identity, 94
    BatchNormWithGlobalNormalization, 94 Conv2D, 94 Relu, 11 Concat, 9 AvgPool,
    5 MaxPool, 1 Sub, 1 Softmax, 1 ResizeBilinear, 1 Reshape, 1 Mul, 1 MatMul,
    1 ExpandDims, 1 DecodeJpeg, 1 Cast, 1 BiasAdd

The important part for our current purposes is the number of const
parameters. In most models these will be stored as 32-bit floats to start, so if
you multiply the number of const parameters by four, you should get something
that’s close to the size of the file on disk. You can often get away with only
eight-bits per parameter with very little loss of accuracy in the final result,
so if your file size is too large you can try using
<a href="https://www.tensorflow.org/performance/quantization">quantize_weights</a>
to transform the parameters down.

    bazel build tensorflow/tools/graph_transforms:transform_graph && \
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=/tmp/tensorflow_inception_optimized.pb \
    --out_graph=/tmp/tensorflow_inception_quantized.pb \
    --inputs='Mul:0' --outputs='softmax:0' --transforms='quantize_weights'

If you look at the resulting file size, you should see that it’s about a quarter
of the original at 23MB.

Another transform is `round_weights`, which doesn't make the file smaller, but it
makes the file compressible to about the same size as when `quantize_weights` is
used. This is particularly useful for mobile development, taking advantage of
the fact that app bundles are compressed before they’re downloaded by consumers.

The original file does not compress well with standard algorithms, because the
bit patterns of even very similar numbers can be very different. The
`round_weights` transform keeps the weight parameters stored as floats, but
rounds them to a set number of step values. This means there are a lot more
repeated byte patterns in the stored model, and so compression can often bring
the size down dramatically, in many cases to near the size it would be if they
were stored as eight bit.

Another advantage of `round_weights` is that the framework doesn’t have to
allocate a temporary buffer to unpack the parameters into, as we have to when
we just use `quantize_weights`. This saves a little bit of latency (though the
results should be cached so it’s only costly on the first run) and makes it
possible to use memory mapping, as described later.

## Binary Size

One of the biggest differences between mobile and server development is the
importance of binary size. On desktop machines it’s not unusual to have
executables that are hundreds of megabytes on disk, but for mobile and embedded
apps it’s vital to keep the binary as small as possible so that user downloads
are easy. As mentioned above, TensorFlow only includes a subset of op
implementations by default, but this still results in a 12 MB final
executable. To reduce this, you can set up the library to only include the
implementations of the ops that you actually need, based on automatically
analyzing your model. To use it:

- Run `tools/print_required_ops/print_selective_registration_header.py` on your
  model to produce a header file that only enables the ops it uses.

- Place the `ops_to_register.h` file somewhere that the compiler can find
  it. This can be in the root of your TensorFlow source folder.

- Build TensorFlow with `SELECTIVE_REGISTRATION` defined, for example by passing
  in `--copts=”-DSELECTIVE_REGISTRATION”` to your Bazel build command.

This process recompiles the library so that only the needed ops and types are
included, which can dramatically reduce the executable size. For example, with
Inception v3, the new size is only 1.5MB.

## How to Profile your Model

Once you have an idea of what your device's peak performance range is, it’s
worth looking at its actual current performance. Using a standalone TensorFlow
benchmark, rather than running it inside a larger app, helps isolate just the
Tensorflow contribution to the
latency. The
[tensorflow/tools/benchmark](https://www.tensorflow.org/code/tensorflow/tools/benchmark/) tool
is designed to help you do this. To run it on Inception v3 on your desktop
machine, build this benchmark model:

    bazel build -c opt tensorflow/tools/benchmark:benchmark_model && \
    bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph=/tmp/tensorflow_inception_graph.pb --input_layer="Mul" \
    --input_layer_shape="1,299,299,3" --input_layer_type="float" \
    --output_layer="softmax:0" --show_run_order=false --show_time=false \
    --show_memory=false --show_summary=true --show_flops=true --logtostderr

You should see output that looks something like this:

<pre>
============================== Top by Computation Time ==============================
[node
 type]  [start]  [first] [avg ms]     [%]  [cdf%]  [mem KB]  [Name]
Conv2D   22.859   14.212   13.700  4.972%  4.972%  3871.488  conv_4/Conv2D
Conv2D    8.116    8.964   11.315  4.106%  9.078%  5531.904  conv_2/Conv2D
Conv2D   62.066   16.504    7.274  2.640% 11.717%   443.904  mixed_3/conv/Conv2D
Conv2D    2.530    6.226    4.939  1.792% 13.510%  2765.952  conv_1/Conv2D
Conv2D   55.585    4.605    4.665  1.693% 15.203%   313.600  mixed_2/tower/conv_1/Conv2D
Conv2D  127.114    5.469    4.630  1.680% 16.883%    81.920  mixed_10/conv/Conv2D
Conv2D   47.391    6.994    4.588  1.665% 18.548%   313.600  mixed_1/tower/conv_1/Conv2D
Conv2D   39.463    7.878    4.336  1.574% 20.122%   313.600  mixed/tower/conv_1/Conv2D
Conv2D  127.113    4.192    3.894  1.413% 21.535%   114.688  mixed_10/tower_1/conv/Conv2D
Conv2D   70.188    5.205    3.626  1.316% 22.850%   221.952  mixed_4/conv/Conv2D

============================== Summary by node type ==============================
[Node type]  [count]  [avg ms]    [avg %]    [cdf %]  [mem KB]
Conv2D            94   244.899    88.952%    88.952% 35869.953
BiasAdd           95     9.664     3.510%    92.462% 35873.984
AvgPool            9     7.990     2.902%    95.364%  7493.504
Relu              94     5.727     2.080%    97.444% 35869.953
MaxPool            5     3.485     1.266%    98.710%  3358.848
Const            192     1.727     0.627%    99.337%     0.000
Concat            11     1.081     0.393%    99.730%  9892.096
MatMul             1     0.665     0.242%    99.971%     4.032
Softmax            1     0.040     0.015%    99.986%     4.032
<>                 1     0.032     0.012%    99.997%     0.000
Reshape            1     0.007     0.003%   100.000%     0.000

Timings (microseconds): count=50 first=330849 curr=274803 min=232354 max=415352 avg=275563 std=44193
Memory (bytes): count=50 curr=128366400(all same)
514 nodes defined 504 nodes observed
</pre>

This is the summary view, which is enabled by the show_summary flag. To
interpret it, the first table is a list of the nodes that took the most time, in
order by how long they took. From left to right, the columns are:

- Node type, what kind of operation this was.

- Start time of the op, showing where it falls in the sequence of operations.

- First time in milliseconds. This is how long the operation took on the first
  run of the benchmark, since by default 20 runs are executed to get more
  reliable statistics. The first time is useful to spot which ops are doing
  expensive calculations on the first run, and then caching the results.

- Average time for the operation across all runs, in milliseconds.

- What percentage of the total time for one run the op took. This is useful to
  understand where the hotspots are.

- The cumulative total time of this and the previous ops in the table. This is
  handy for understanding what the distribution of work is across the layers, to
  see if just a few of the nodes are taking up most of the time.
  
- The amount of memory consumed by outputs of this type of op.

- Name of the node.

The second table is similar, but instead of breaking down the timings by
particular named nodes, it groups them by the kind of op. This is very useful to
understand which op implementations you might want to optimize or eliminate from
your graph. The table is arranged with the most costly operations at the start,
and only shows the top ten entries, with a placeholder for other nodes. The
columns from left to right are:

- Type of the nodes being analyzed.

- Accumulated average time taken by all nodes of this type, in milliseconds.

- What percentage of the total time was taken by this type of operation.

- Cumulative time taken by this and op types higher in the table, so you can
  understand the distribution of the workload.

-  How much memory the outputs of this op type took up.

Both of these tables are set up so that you can easily copy and paste their
results into spreadsheet documents, since they are output with tabs as
separators between the columns. The summary by node type can be the most useful
when looking for optimization opportunities, since it’s a pointer to the code
that’s taking the most time. In this case, you can see that the Conv2D ops are
almost 90% of the execution time. This is a sign that the graph is pretty
optimal, since convolutions and matrix multiplies are expected to be the bulk of
a neural network’s computing workload.

As a rule of thumb, it’s more worrying if you see a lot of other operations
taking up more than a small fraction of the time. For neural networks, the ops
that don’t involve large matrix multiplications should usually be dwarfed by the
ones that do, so if you see a lot of time going into those it’s a sign that
either your network is non-optimally constructed, or the code implementing those
ops is not as optimized as it could
be. [Performance bugs](https://github.com/tensorflow/tensorflow/issues) or
patches are always welcome if you do encounter this situation, especially if
they include an attached model exhibiting this behavior and the command line
used to run the benchmark tool on it.

The run above was on your desktop, but the tool also works on Android, which is
where it’s most useful for mobile development. Here’s an example command line to
run it on a 64-bit ARM device:

    bazel build -c opt --config=android_arm64 \
    tensorflow/tools/benchmark:benchmark_model
    adb push bazel-bin/tensorflow/tools/benchmark/benchmark_model /data/local/tmp
    adb push /tmp/tensorflow_inception_graph.pb /data/local/tmp/
    adb shell '/data/local/tmp/benchmark_model \
    --graph=/data/local/tmp/tensorflow_inception_graph.pb --input_layer="Mul" \
    --input_layer_shape="1,299,299,3" --input_layer_type="float" \
    --output_layer="softmax:0" --show_run_order=false --show_time=false \
    --show_memory=false --show_summary=true'

You can interpret the results in exactly the same way as the desktop version
above. If you have any trouble figuring out what the right input and output
names and types are, take a look at the
<a href="./prepare_models">Preparing models</a>
page for details about detecting these for your model, and look at the
`summarize_graph` tool which may give you
helpful information.

There isn’t good support for command line tools on iOS, so instead there’s a
separate example
at
[tensorflow/examples/ios/benchmark](https://www.tensorflow.org/code/tensorflow/examples/ios/benchmark) that
packages the same functionality inside a standalone app. This outputs the
statistics to both the screen of the device and the debug log. If you want
on-screen statistics for the Android example apps, you can turn them on by
pressing the volume-up button.

## Profiling within your own app

The output you see from the benchmark tool is generated from modules that are
included as part of the standard TensorFlow runtime, which means you have access
to them within your own applications too. You can see an example of how to do
that [here](https://www.tensorflow.org/code/tensorflow/examples/ios/benchmark/BenchmarkViewController.mm?l=139).

The basic steps are:

1. Create a StatSummarizer object:

        tensorflow::StatSummarizer stat_summarizer(tensorflow_graph);

2. Set up the options:

        tensorflow::RunOptions run_options;
        run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
        tensorflow::RunMetadata run_metadata;

3. Run the graph:

        run_status = session->Run(run_options, inputs, output_layer_names, {},
                                  output_layers, &run_metadata);

4. Calculate the results and print them out:

        assert(run_metadata.has_step_stats());
        const tensorflow::StepStats& step_stats = run_metadata.step_stats();
        stat_summarizer->ProcessStepStats(step_stats);
        stat_summarizer->PrintStepStats();

## Visualizing Models

The most effective way to speed up your code is by altering your model so it
does less work. To do that, you need to understand what your model is doing, and
visualizing it is a good first step. To get a high-level overview of your graph,
use [TensorBoard](https://github.com/tensorflow/tensorboard).

## Threading

The desktop version of TensorFlow has a sophisticated threading model, and will
try to run multiple operations in parallel if it can. In our terminology this is
called “inter-op parallelism” (though to avoid confusion with “intra-op”, you
could think of it as “between-op” instead), and can be set by specifying
`inter_op_parallelism_threads` in the session options.

By default, mobile devices run operations serially; that is,
`inter_op_parallelism_threads` is set to 1. Mobile processors usually have few
cores and a small cache, so running multiple operations accessing disjoint parts
of memory usually doesn’t help performance. “Intra-op parallelism” (or
“within-op”) can be very helpful though, especially for computation-bound
operations like convolutions where different threads can feed off the same small
set of memory.

On mobile, how many threads an op will use is set to the number of cores by
default, or 2 when the number of cores can't be determined. You can override the
default number of threads that ops are using by setting
`intra_op_parallelism_threads` in the session options.  It’s a good idea to
reduce the default if your app has its own threads doing heavy processing, so
that they don’t interfere with each other.

To see more details on session options, look at [ConfigProto](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto).

## Retrain with mobile data

The biggest cause of accuracy problems when running models on mobile apps is
unrepresentative training data. For example, most of the Imagenet photos are
well-framed so that the object is in the center of the picture, well-lit, and
shot with a normal lens. Photos from mobile devices are often poorly framed,
badly lit, and can have fisheye distortions, especially selfies.

The solution is to expand your training set with data actually captured from
your application. This step can involve extra work, since you’ll have to label
the examples yourself, but even if you just use it to expand your original
training data, it can help the training set dramatically. Improving the training
set by doing this, and by fixing other quality issues like duplicates or badly
labeled examples is the single best way to improve accuracy. It’s usually a
bigger help than altering your model architecture or using different techniques.

## Reducing model loading time and/or memory footprint

Most operating systems allow you to load a file using memory mapping, rather
than going through the usual I/O APIs. Instead of allocating an area of memory
on the heap and then copying bytes from disk into it, you simply tell the
operating system to make the entire contents of a file appear directly in
memory. This has several advantages:

* Speeds loading
* Reduces paging (increases performance)
* Does not count towards RAM budget for your app

TensorFlow has support for memory mapping the weights that form the bulk of most
model files. Because of limitations in the `ProtoBuf` serialization format, we
have to make a few changes to our model loading and processing code. The
way memory mapping works is that we have a single file where the first part is a
normal `GraphDef` serialized into the protocol buffer wire format, but then the
weights are appended in a form that can be directly mapped.

To create this file, run the
`tensorflow/contrib/util:convert_graphdef_memmapped_format` tool. This takes in
a `GraphDef` file that’s been run through `freeze_graph` and converts it to the
format that has the weights appended at the end. Since that file’s no longer a
standard `GraphDef` protobuf, you then need to make some changes to the loading
code. You can see an example of this in
the
[iOS Camera demo app](https://www.tensorflow.org/code/tensorflow/examples/ios/camera/tensorflow_utils.mm?l=147),
in the `LoadMemoryMappedModel()` function.

The same code (with the Objective C calls for getting the filenames substituted)
can be used on other platforms too. Because we’re using memory mapping, we need
to start by creating a special TensorFlow environment object that’s set up with
the file we’ll be using:

    std::unique_ptr<tensorflow::MemmappedEnv> memmapped_env;
    memmapped_env->reset(
          new tensorflow::MemmappedEnv(tensorflow::Env::Default()));
    tensorflow::Status mmap_status =
          (memmapped_env->get())->InitializeFromFile(file_path);

You then need to pass in this environment to subsequent calls, like this one for
loading the graph:

    tensorflow::GraphDef tensorflow_graph;
    tensorflow::Status load_graph_status = ReadBinaryProto(
        memmapped_env->get(),
        tensorflow::MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
        &tensorflow_graph);

You also need to create the session with a pointer to the environment you’ve
created:

    tensorflow::SessionOptions options;
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(::tensorflow::OptimizerOptions::L0);
    options.env = memmapped_env->get();

    tensorflow::Session* session_pointer = nullptr;
    tensorflow::Status session_status =
        tensorflow::NewSession(options, &session_pointer);

One thing to notice here is that we’re also disabling automatic optimizations,
since in some cases these will fold constant sub-trees, and so create copies of
tensor values that we don’t want and use up more RAM.

Once you’ve gone through these steps, you can use the session and graph as
normal, and you should see a reduction in loading time and memory usage.

## Protecting model files from easy copying

By default, your models will be stored in the standard serialized protobuf
format on disk. In theory this means that anybody can copy your model, which you
may not want. However, in practice, most models are so application-specific and
obfuscated by optimizations that the risk is similar to that of competitors
disassembling and reusing your code, but if you do want to make it tougher for
casual users to access your files it is possible to take some basic steps.

Most of our examples use
the
[ReadBinaryProto()](https://www.tensorflow.org/code/tensorflow/core/platform/env.cc?q=core/platform/env.cc&l=409) convenience
call to load a `GraphDef` from disk. This does require an unencrypted protobuf on
disk. Luckily though, the implementation of the call is pretty straightforward
and it should be easy to write an equivalent that can decrypt in memory. Here's
some code that shows how you can read and decrypt a protobuf using your own
decryption routine:

    Status ReadEncryptedProto(Env* env, const string& fname,
                              ::tensorflow::protobuf::MessageLite* proto) {
      string data;
      TF_RETURN_IF_ERROR(ReadFileToString(env, fname, &data));

      DecryptData(&data);  // Your own function here.

      if (!proto->ParseFromString(&data)) {
        TF_RETURN_IF_ERROR(stream->status());
        return errors::DataLoss("Can't parse ", fname, " as binary proto");
      }
      return Status::OK();
    }

To use this you’d need to define the DecryptData() function yourself. It could
be as simple as something like:

    void DecryptData(string* data) {
      for (int i = 0; i < data.size(); ++i) {
        data[i] = data[i] ^ 0x23;
      }
    }

You may want something more complex, but exactly what you’ll need is outside the
current scope here.
