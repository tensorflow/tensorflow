# Gemma2 2B Keras model

Scripts to run Gemma2 2B Keras model on CPU.

Model link: https://www.kaggle.com/models/google/gemma-2/keras

Instructions:

*   Set up your Kaggle API key by following
    [these instructions](https://www.kaggle.com/docs/api#authentication).
*   `$ bash setup.sh`
    *   This only needs to be run once. It will create a virtual environment at
        a location read from `config.sh` and install the necessary dependencies.
    *   Change the `VENV_BASE` variable in `config.sh` before running `setup.sh`
        if you want to use a different location.
*   `$ KERAS_BACKEND=jax bash run.sh`
    *   This script activates the right virtual environment and runs the
        benchmark in `benchmark.py`.
    *   Set `KERAS_BACKEND=tensorflow` or `torch` to run with TensorFlow or
        PyTorch backend.
*   (Optional) Delete the virtual environment: `$ bash cleanup.sh`

To try other model variations with different numbers of parameters, modify the
following line in `benchmark.py`:

```
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_2b_en")
```

Replace "gemma2_2b_en" with other preset names, e.g.,
"gemma2_instruct_2b_en","gemma2_9b_en", etc. See the full preset list
[here](https://github.com/keras-team/keras-hub/blob/86607dc921999e33f5b8a0bcf81ec987b60c9dee/keras_hub/src/models/gemma/gemma_presets.py#L5-L200).
