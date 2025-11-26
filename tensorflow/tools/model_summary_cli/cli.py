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
"""TensorFlow Model Summary CLI - inspect model architectures from command line."""

import argparse
import sys

from absl import app

from tensorflow.tools.model_summary_cli import model_loader


FLAGS = None


def create_parser():
  """Create argument parser for tf_model_summary CLI."""
  parser = argparse.ArgumentParser(
      prog='tf_model_summary',
      description='Inspect TensorFlow/Keras model architecture without writing Python code.',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
  # View model architecture
  tf_model_summary ./saved_model/
  tf_model_summary ./model.h5
  tf_model_summary ./model.keras

  # Export visualization
  tf_model_summary ./model.h5 --plot architecture.png
  tf_model_summary ./model.h5 --plot architecture.svg

  # JSON output for scripting
  tf_model_summary ./model.h5 --json
  tf_model_summary ./model.h5 --json | jq '.total_params'

  # Customize output
  tf_model_summary ./model.h5 --line-length 120
  tf_model_summary ./model.h5 --plot out.png --expand-nested --dpi 150

Supported formats:
  - SavedModel (directory containing saved_model.pb)
  - HDF5 (.h5, .hdf5 files)
  - Keras (.keras files)

For visualization, install optional dependencies:
  pip install pydot
  # Plus graphviz via system package manager
      """)

  parser.register('type', 'bool', lambda v: v.lower() == 'true')

  parser.add_argument(
      'model_path',
      type=str,
      help='Path to the model file or SavedModel directory.')

  parser.add_argument(
      '--plot',
      type=str,
      default=None,
      metavar='FILE',
      help='Export model graph to PNG or SVG file (requires pydot and graphviz).')

  parser.add_argument(
      '--line-length',
      type=int,
      default=None,
      dest='line_length',
      help='Width of printed summary lines (default: auto-detect based on model type).')

  parser.add_argument(
      '--show-shapes',
      nargs='?',
      const=True,
      type='bool',
      default=True,
      dest='show_shapes',
      help='Show input/output shapes in plot output (default: True).')

  parser.add_argument(
      '--show-layer-names',
      nargs='?',
      const=True,
      type='bool',
      default=True,
      dest='show_layer_names',
      help='Show layer names in plot output (default: True).')

  parser.add_argument(
      '--expand-nested',
      nargs='?',
      const=True,
      type='bool',
      default=False,
      dest='expand_nested',
      help='Expand nested models in plot output (default: False).')

  parser.add_argument(
      '--dpi',
      type=int,
      default=96,
      help='DPI for plot output (default: 96).')

  parser.add_argument(
      '--json',
      action='store_true',
      default=False,
      help='Output model summary in JSON format.')

  parser.add_argument(
      '-v', '--version',
      action='version',
      version='tf_model_summary 1.0.0')

  return parser


def run_summary(args):
  """Execute the model summary command."""
  try:
    model = model_loader.load_model(args.model_path)
  except model_loader.ModelLoadError as e:
    print(str(e), file=sys.stderr)
    return 1

  # JSON output mode
  if args.json:
    from tensorflow.tools.model_summary_cli import layer_parser
    from tensorflow.tools.model_summary_cli import formatter
    model_info = layer_parser.get_model_info(model)
    print(formatter.format_model_summary_json(model_info))
  else:
    # Standard summary output
    print(f"Model: {args.model_path}")
    print()
    try:
      model.summary(line_length=args.line_length)
    except Exception as e:
      print(f"Error generating summary: {e}", file=sys.stderr)
      return 1

  # Handle plot export if requested
  if args.plot:
    from tensorflow.tools.model_summary_cli import visualization
    try:
      visualization.export_model_plot(
          model,
          args.plot,
          show_shapes=args.show_shapes,
          show_layer_names=args.show_layer_names,
          expand_nested=args.expand_nested,
          dpi=args.dpi
      )
      if not args.json:
        print(f"\nModel graph exported to: {args.plot}")
    except visualization.VisualizationError as e:
      print(f"\nError: {e}", file=sys.stderr)
      return 1

  return 0


def main(_):
  """Main entry point."""
  global FLAGS
  return run_summary(FLAGS)


def cli_main():
  """CLI entry point for console_scripts."""
  global FLAGS
  parser = create_parser()
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)


if __name__ == '__main__':
  cli_main()
