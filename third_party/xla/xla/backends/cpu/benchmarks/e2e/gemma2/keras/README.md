# Gemma Keras model

Scripts to run Gemma Keras model on CPU.

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

To try other model variations with different numbers of parameters,
pass `--model_name` when running the script:

```
$ KERAS_BACKEND=jax bash run.sh --model_name=gemma2_2b_en
```

Replace `gemma2_2b_en` with other preset names, e.g.,
`gemma3_1b`, `gemma4_2b`, etc. See the full preset list
[here](https://keras.io/keras_hub/presets/).

