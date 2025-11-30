# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Visualization utilities for tf_model_summary CLI."""


class VisualizationError(Exception):
  """Raised when visualization fails."""
  pass


def check_visualization_deps():
  """Check if visualization dependencies are available.

  Returns:
    Tuple of (available: bool, message: str).
  """
  try:
    from tensorflow.python.keras.utils import vis_utils
    if vis_utils.check_pydot():
      return True, "Visualization dependencies available."
    else:
      return False, (
          "Visualization requires pydot and graphviz.\n"
          "Install with:\n"
          "  pip install pydot\n"
          "  # And install graphviz via your system package manager"
      )
  except ImportError as e:
    return False, f"Missing dependency: {e}"


def export_model_plot(model, output_path, show_shapes=True,
                      show_layer_names=True, expand_nested=False,
                      dpi=96):
  """Export model architecture to image file.

  Args:
    model: A Keras model instance.
    output_path: Path for output file (PNG or SVG).
    show_shapes: Whether to show shapes in the plot.
    show_layer_names: Whether to show layer names.
    expand_nested: Whether to expand nested models.
    dpi: Image resolution.

  Raises:
    VisualizationError: If export fails.
  """
  available, message = check_visualization_deps()
  if not available:
    raise VisualizationError(message)

  try:
    from tensorflow.python.keras.utils import vis_utils
    vis_utils.plot_model(
        model,
        to_file=output_path,
        show_shapes=show_shapes,
        show_layer_names=show_layer_names,
        expand_nested=expand_nested,
        dpi=dpi
    )
  except Exception as e:
    raise VisualizationError(f"Failed to export plot: {e}")
