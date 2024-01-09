"""Definitions for targets in DPB (Delegate Performance Benchmark)."""

def latency_benchmark_extra_deps():
    """Defines extra dependencies for latency benchmark. Currently empty."""
    return []

def accuracy_benchmark_extra_deps():
    """Defines extra dependencies for accuracy benchmark. Currently empty."""
    return []

def latency_benchmark_extra_models():
    """Defines extra models for latency benchmark. Currently empty.

    Returns a list of tuples where each tuple has two fields: 1) the model name and 2) the model target label. Example:
    [
        ("model1.tflite", "@repo//package:model1.tflite"),
        ("model2.tflite", "@repo//package:model2.tflite"),
    ]
    """
    return []

def accuracy_benchmark_extra_models():
    """Defines extra models for accuracy benchmark. Currently empty.

    Returns a list of tuples where each tuple has two fields: 1) the model name and 2) the model target label. Example:
    [
        ("model1_with_validation.tflite", "@repo//package:model1_with_validation.tflite"),
        ("model2_with_validation.tflite", "@repo//package:model2_with_validation.tflite"),
    ]
    """
    return []
