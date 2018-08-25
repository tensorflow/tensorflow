#!/usr/bin/env bash
python -c 'import tensorflow as tf; tf.test.is_gpu_available() or exit(1)'
