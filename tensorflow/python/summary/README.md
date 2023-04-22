# TensorFlow Event Processing

This folder contains classes useful for analyzing and visualizing TensorFlow
events files. The code is primarily being developed to support TensorBoard,
but it can be used by anyone who wishes to analyze or visualize TensorFlow
events files.

If you wish to load TensorFlow events, you should use an EventAccumulator
(to load from a single events file) or an EventMultiplexer (to load from
multiple events files).
