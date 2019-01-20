## TFLite accuracy library.

This library provides evaluation pipelines that can be used to evaluate
accuracy and other metrics of a model. The resulting binary can be run on
a desktop or on a mobile device.

## Usage
The tool provides an evaluation pipeline with different stages. Each
stage outputs a Tensorflow graph.
A sample usage is shown below.

```C++
// First build the pipeline.
EvalPipelineBuilder builder;
std::unique_ptr<EvalPipeline> eval_pipeline;
auto status = builder.WithInput("pipeline_input", DT_FLOAT)
     .WithInputStage(&input_stage)
     .WithRunModelStage(&run_model_stage)
     .WithPreprocessingStage(&preprocess_stage)
     .WithAccuracyEval(&eval)
     .Build(scope, &eval_pipeline);
TF_CHECK_OK(status);

// Now run the pipeline with inputs and outputs.
std::unique_ptr<Session> session(NewSession(SessionOptions()));
TF_CHECK_OK(eval_pipeline.AttachSession(std::move(session)));
Tensor input = ... read input for the model ...
Tensor ground_truth = ... read ground truth for the model ...
TF_CHECK_OK(eval_pipeline.Run(input1, ground_truth1));
```
For further examples, check the usage in [imagenet accuracy evaluation binary](ilsvrc/imagenet_model_evaluator.cc)

## Measuring accuracy of published models.

### ILSVRC (Imagenet Large Scale Visual Recognition Contest) classification task
For measuring accuracy for [ILSVRC 2012 image classification task](http://www.image-net.org/challenges/LSVRC/2012/), the binary can be built
using these
[instructions.](ilsvrc/)
