# TensorFlow Installation and Setup Guide

## Introduction

This guide provides comprehensive instructions for installing and setting up TensorFlow for machine learning development.

## System Requirements

- Python 3.9 or later
- pip (Python package installer)
- 8GB RAM minimum (16GB recommended)
- GPU support requires CUDA 11.8+ and cuDNN 8.6+

## Installation Methods

### 1. CPU-Only Installation

```bash
pip install tensorflow
```

This is the simplest installation method suitable for development and testing.

### 2. GPU-Enabled Installation

```bash
pip install tensorflow[and-cuda]
```

## Configuration Best Practices

1. **Virtual Environment**: Always use a virtual environment to avoid package conflicts
   ```bash
   python -m venv tf-env
   source tf-env/bin/activate
   ```

2. **Version Compatibility**: Check compatibility between TensorFlow, CUDA, and cuDNN versions

3. **Memory Management**: For large datasets, configure memory growth to avoid OOM errors
   ```python
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   ```

## Verification

Verify installation with:
```python
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_available())
```

## Common Issues and Solutions

- **CUDA Library Issues**: Ensure CUDA paths are correctly set in environment variables
- **Out of Memory**: Reduce batch size or enable memory growth
- **Import Errors**: Verify Python version compatibility

## Additional Resources

- Official TensorFlow Documentation: https://www.tensorflow.org/guide
- Installation Troubleshooting: https://www.tensorflow.org/install/errors
