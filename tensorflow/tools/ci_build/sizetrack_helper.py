#!/usr/bin/env python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Used for Google-internal artifact size tracking.

See go/tf-devinfra/sizetrack.

INVOCATION: The following flags are required:

  sizetrack_helper.py \
      --artifact=ARTIFACT, or --manual_bytes=MANUAL_BYTES
      --artifact_id=ARTIFACT_ID \
      --team=TEAM \
      ... other optional args ...

On Windows you might need something like:

    C:\Python38\python.exe C:\path\to\sizetrack_helper.py ...

PREREQUISITES:

  1. Your current activated GCP user must have access scopes and IAM permissions
     to do the following:

      1. Query and load data into BigQuery
      2. Upload files to GCS

  2. Your environment must match the following criteria:

      1. Current directory is a git repository
      2. CL-based commits have a PiperOrigin-RevId trailer. This is the case
         for any use of Copybara Single-source-of-truth, e.g. TensorFlow.
         Only these commits are considered when running commands.
"""

import argparse
import csv
import datetime
import os
import os.path
import pathlib
import platform
import re
import subprocess

parser = argparse.ArgumentParser(
    usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--project",
    type=str,
    default="tensorflow-testing",
    help="GCP project you can access.")
parser.add_argument(
    "--dataset",
    type=str,
    default="sizetracker",
    help="BigQuery dataset containing --table")
parser.add_argument(
    "--table", type=str, default="tensorflow_devinfra", help="BigQuery table.")
parser.add_argument(
    "--upload",
    action="store_true",
    help="Upload the artifact to --bucket for analysis.")
parser.add_argument(
    "--bucket",
    type=str,
    default="gs://tf-sizetracker-artifacts",
    help="GCS bucket for artifacts.")
parser.add_argument(
    "--team",
    type=str,
    help="For grouping in the dashboard and buckets; e.g. tf-lite-team.")
parser.add_argument(
    "--artifact_id",
    type=str,
    help="Unique ID for your artifact, used for sorting dashboards.")
parser.add_argument(
    "-n",
    "--dry_run",
    action="store_true",
    help="Dry run: do not load to BigQuery or upload to GCS.")
parser.add_argument(
    "--job",
    type=str,
    help="Name of job calling this script. Default: $KOKORO_JOB_NAME.")
parser.add_argument(
    "--build_id",
    type=str,
    help="UUID of build calling this script. Default: $KOKORO_BUILD_ID.")
parser.add_argument(
    "--print_schema",
    action="store_true",
    help="Print the table schema and don't do anything else.")
size = parser.add_mutually_exclusive_group()
size.add_argument(
    "--artifact",
    type=argparse.FileType("r"),
    help="Local file you are measuring.")
size.add_argument(
    "--manual_bytes",
    type=int,
    help="Manually set the recorded size instead of providing an artifact.")
FLAGS = parser.parse_args()

NOW = datetime.datetime.now(
    datetime.timezone.utc).replace(microsecond=0).isoformat()
TABLE_NAME = "{}.{}".format(FLAGS.dataset, FLAGS.table)
PROJECT_LEVEL_TABLE_NAME = "{}:{}".format(FLAGS.project, TABLE_NAME)
CL_TRAILER = "PiperOrigin-RevId"
PRETTY_COMMIT_DATE = "%cI"
# \001 is a byte with value "1", in octal. We use this in git_pretty()
PRETTY_CL = "\001%(trailers)\001"
PRETTY_HEAD_INFO = "%h\t{cl}\t%s\t%ae\t%aI\t%ce\t%cI".format(cl=PRETTY_CL)
PRETTY_EARLY = "%aI\t{cl}\t%cI".format(cl=PRETTY_CL)
PRETTY_COMMIT = "%h"
# This is a BigQuery table schema defined as CSV
# See https://cloud.google.com/bigquery/docs/schemas
SCHEMA = ",".join([
    "id:string",
    "filename:string",
    # These 6 lines are from git's format=pretty
    # %h $CL_PRETTY %s %ae %aI %ce %cI
    "commit:string",
    "cl:int64",
    "description:string",
    "author:string",
    "author_date:timestamp",
    "committer:string",
    "commit_date:timestamp",
    # Done with format=pretty
    "earliest_commit:string",
    "earliest_cl:int64",
    "earliest_author_date:timestamp",
    "earliest_commit_date:timestamp",
    "all_commits:string",
    "all_cls:string",
    "bytes:int64",
    "team:string",
    "logged_date:timestamp",
    "uploaded_to:string",
    "job:string",
    "build_id:string",
])
# Select the earliest recorded commit in the same table for the same artifact
# and team. Used to determine the full range of tested commits for each
# invocation. Returns empty string if there are no earlier records.
BQ_GET_EARLIEST_INCLUDED_COMMIT = """
  SELECT
    commit
  FROM {table} WHERE
    commit_date < '{earlier_than_this_date}'
    AND id = '{artifact_id}'
    AND team = '{team}'
  ORDER BY commit_date DESC LIMIT 1
"""


# pylint: disable=unused-argument
def git_pretty(commit_range, pretty_format, n=None):
  r"""Run git log and return the cleaned results.

  Git is assumed to be available in the PATH.

  The PiperOrigin-RevId trailer always picks up an extra newline, so this splits
  entries on a null byte (\0, or %x00 for git log) and removes newlines.

  Args:
    commit_range: Standard range given to git log, e.g. HEAD~1..HEAD
    pretty_format: See https://git-scm.com/docs/pretty-formats
    n: Number of commits to get. By default, get all within commit_range.

  Returns:
    List of strings of whatever the format string was.
  """
  n = [] if n is None else ["-n", "1"]
  try:
    ret = subprocess.run([
        "git", "log", *n, "--date", "iso", "--grep", CL_TRAILER, commit_range,
        "--pretty=format:" + pretty_format + "%x00"
    ],
                         check=True,
                         universal_newlines=True,
                         stderr=subprocess.PIPE,
                         stdout=subprocess.PIPE)
  except subprocess.CalledProcessError as e:
    print(e.stderr)
    print(e.stdout)
    raise e
  out = ret.stdout.replace("\n", "")
  # Unique case: Old versions of git do not expand the special parts of the
  # trailers formatter. In that case, the entire formatter remains, and we
  # need to extract the information in another way. The %trailers general
  # formatter is available, so we'll use that and regex over it.
  cleaned = list(filter(None, map(str.strip, out.split("\0"))))
  trailers_removed = []
  for row in cleaned:
    # Find: a chunk of text surrounded by \001, and extract the number after
    # PiperOrigin-RevId.
    row = re.sub("\001.*PiperOrigin-RevId: (?P<cl>[0-9]+).*\001", r"\g<1>", row)
    trailers_removed.append(row)
  return trailers_removed


def gcloud(tool, args, stdin=None):
  r"""Run a Google cloud utility.

  On Linux and MacOS, utilities are assumed to be in the PATH.
  On Windows, utilities are assumed to be available as
    C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\{tool}.cmd

  Args:
    tool: CLI tool, e.g. bq, gcloud, gsutil
    args: List of arguments, same format as subprocess.run
    stdin: String to send to stdin

  Returns:
    String, the stdout of the tool
  """

  if platform.system() == "Windows":
    tool = (r"C:\Program Files (x86)\Google\Cloud "
            r"SDK\google-cloud-sdk\bin\{}.cmd").format(tool)

  try:
    ret = subprocess.run([tool, *args],
                         check=True,
                         universal_newlines=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         input=stdin)
  except subprocess.CalledProcessError as e:
    print(e.stderr)
    print(e.stdout)
    raise e
  return ret.stdout.strip()


def bq(args, stdin=None):
  """Helper for running bq, the BigQuery tool."""
  # bq prints extra messages to stdout if ~/.bigqueryrc doesn't exist
  pathlib.Path(pathlib.Path.home() / ".bigqueryrc").touch()
  return gcloud(
      "bq", ["--project_id", FLAGS.project, "--headless", *args], stdin=stdin)


def get_all_tested_commits():
  """Get details about the full commit range tested by this invocation."""
  head_info = git_pretty("HEAD", PRETTY_HEAD_INFO, n=1)
  _, _, _, _, _, _, current_commit_date = head_info[0].split("\t")

  query_earliest_included_commit = BQ_GET_EARLIEST_INCLUDED_COMMIT.format(
      table=TABLE_NAME,
      earlier_than_this_date=current_commit_date,
      artifact_id=FLAGS.artifact_id,
      team=FLAGS.team)

  # --format=csv returns an empty string if no results, or else two lines:
  # commit
  # COMMIT_HASH
  earliest_commit = bq(["query", "--format", "csv", "--nouse_legacy_sql"],
                       stdin=query_earliest_included_commit)

  # Compute the commit/CL range since the last test
  if earliest_commit:

    earliest_commit = earliest_commit.splitlines()[-1]  # Ignore CSV header
    early_author_date, early_cl, early_commit_date = git_pretty(
        earliest_commit, PRETTY_EARLY, n=1)[0].split("\t")

    all_range = "{commit}..HEAD".format(commit=earliest_commit)
    # Reversed: convert to chronological
    all_commits = ",".join(reversed(git_pretty(all_range, PRETTY_COMMIT)))
    all_changelists = ",".join(reversed(git_pretty(all_range, PRETTY_CL)))

    return [
        earliest_commit, early_cl, early_author_date, early_commit_date,
        all_commits, all_changelists
    ]

  # If the artifact has never been tracked before this commit
  # Empty cells in CSV loads are loaded as NULL values
  else:
    return [""] * 6


def get_upload_path():
  """Generate URL for 'gsutil cp'."""
  if FLAGS.upload and FLAGS.artifact:
    artifact_filename = os.path.basename(FLAGS.artifact.name)
    # note: not os.path.join here, because gsutil is always linux-style
    # Using a timestamp prevents duplicate entries
    path = "{bucket}/{team}/{artifact_id}/{now}.{artifact_filename}".format(
        bucket=FLAGS.bucket,
        team=FLAGS.team,
        artifact_id=FLAGS.artifact_id,
        now=NOW,
        artifact_filename=artifact_filename)
    return path
  else:
    return ""


def build_row():
  """Assemble one row of data about this artifact."""
  (earliest_commit, early_cl, early_author_date, early_commit_date, all_commits,
   all_changelists) = get_all_tested_commits()

  # Use UTC to make sure machines in different timezones load consistent data
  current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
  artifact_filename = ("NO_FILE" if not FLAGS.artifact else os.path.basename(
      FLAGS.artifact.name))
  size_bytes = FLAGS.manual_bytes or os.path.getsize(FLAGS.artifact.name)
  head_info = git_pretty("HEAD", PRETTY_HEAD_INFO, n=1)
  all_head_info_items = head_info[0].split("\t")
  return [
      FLAGS.artifact_id,
      artifact_filename,
      *all_head_info_items,
      earliest_commit,
      early_cl,
      early_author_date,
      early_commit_date,
      all_commits,
      all_changelists,
      size_bytes,
      FLAGS.team,
      current_time,
      get_upload_path(),
      FLAGS.job,
      FLAGS.build_id,
  ]


def main():

  # Validate flags
  if FLAGS.print_schema:
    print(SCHEMA)
    exit(0)
  elif not FLAGS.team or not FLAGS.artifact_id or not (FLAGS.artifact or
                                                       FLAGS.manual_bytes):
    print(
        "--team and --artifact_id are required if --print_schema is not "
        "specified.\nYou must also specify one of --artifact or --manual_bytes."
        "\nPass -h or --help for usage.")
    exit(1)

  if not FLAGS.job:
    FLAGS.job = os.environ.get("KOKORO_JOB_NAME", "NO_JOB")
  if not FLAGS.build_id:
    FLAGS.build_id = os.environ.get("KOKORO_BUILD_ID", "NO_BUILD")

  # Generate data about this artifact into a Tab Separated Value file
  next_tsv_row = build_row()

  # Upload artifact into GCS if it exists
  if FLAGS.upload and FLAGS.artifact:
    upload_path = get_upload_path()
    if FLAGS.dry_run:
      print("DRY RUN: Would gsutil cp to:\n{}".format(upload_path))
    else:
      gcloud("gsutil", ["cp", FLAGS.artifact.name, upload_path])

  # Load into BigQuery
  if FLAGS.dry_run:
    print("DRY RUN: Generated this TSV row:")
    print("\t".join(map(str, next_tsv_row)))
  else:
    with open("data.tsv", "w", newline="") as tsvfile:
      writer = csv.writer(
          tsvfile,
          delimiter="\t",
          quoting=csv.QUOTE_MINIMAL,
          lineterminator=os.linesep)
      writer.writerow(next_tsv_row)
    bq([
        "load", "--source_format", "CSV", "--field_delimiter", "tab",
        PROJECT_LEVEL_TABLE_NAME, "data.tsv", SCHEMA
    ])


if __name__ == "__main__":
  main()
