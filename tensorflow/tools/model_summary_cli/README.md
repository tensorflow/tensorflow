# TensorFlow Model Summary CLI

A command-line tool for inspecting TensorFlow/Keras model architectures without writing Python code.

## Usage

```bash
tf_model_summary <model_path> [options]
```

## Supported Formats

- **SavedModel**: Directory containing `saved_model.pb`
- **HDF5**: `.h5`, `.hdf5` files
- **Keras**: `.keras` files

## Options

| Option | Description |
|--------|-------------|
| `--plot FILE` | Export model graph to PNG or SVG file |
| `--json` | Output model summary in JSON format |
| `--line-length N` | Width of printed summary lines |
| `--show-shapes` | Show input/output shapes in plot (default: True) |
| `--show-layer-names` | Show layer names in plot (default: True) |
| `--expand-nested` | Expand nested models in plot (default: False) |
| `--dpi N` | DPI for plot output (default: 96) |
| `-v, --version` | Show version number |

## Examples

### View Model Architecture

```bash
tf_model_summary ./saved_model/
tf_model_summary ./model.h5
tf_model_summary ./model.keras
```

### Export Visualization

```bash
tf_model_summary ./model.h5 --plot architecture.png
tf_model_summary ./model.h5 --plot architecture.svg
```

### JSON Output for Scripting

```bash
tf_model_summary ./model.h5 --json
tf_model_summary ./model.h5 --json | jq '.total_params'
```

### Customize Output

```bash
tf_model_summary ./model.h5 --line-length 120
tf_model_summary ./model.h5 --plot out.png --expand-nested --dpi 150
```

## Sample Output

### Standard Output

```
Model: ./model.keras

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 64)             │           704 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1)              │            33 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2,817 (11.00 KB)
 Trainable params: 2,817 (11.00 KB)
 Non-trainable params: 0 (0.00 B)
```

### JSON Output

```json
{
  "model_name": "sequential",
  "model_class": "Sequential",
  "total_params": 2817,
  "trainable_params": 2817,
  "non_trainable_params": 0,
  "layers": [
    {
      "name": "dense",
      "class": "Dense",
      "output_shape": "(None, 64)",
      "params": 704,
      "trainable": true
    },
    {
      "name": "dense_1",
      "class": "Dense",
      "output_shape": "(None, 32)",
      "params": 2080,
      "trainable": true
    },
    {
      "name": "dense_2",
      "class": "Dense",
      "output_shape": "(None, 1)",
      "params": 33,
      "trainable": true
    }
  ]
}
```

## Optional Dependencies

For visualization export (`--plot`), install:

```bash
pip install pydot
# Plus graphviz via system package manager:
# macOS: brew install graphviz
# Ubuntu: apt-get install graphviz
# Windows: choco install graphviz
```

## Module Structure

```
tensorflow/tools/model_summary_cli/
├── __init__.py        # Package marker
├── cli.py             # Main CLI entry point
├── model_loader.py    # Model loading with format detection
├── layer_parser.py    # Layer information extraction
├── formatter.py       # Output formatting (JSON)
├── visualization.py   # PNG/SVG export wrapper
└── BUILD              # Bazel build file
```

## Entry Point

Registered in `tensorflow/tools/pip_package/setup.py.tpl`:

```python
'tf_model_summary = tensorflow.tools.model_summary_cli.cli:cli_main'
```
