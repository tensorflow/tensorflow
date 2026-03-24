# Copyright 2025 The OpenXLA Authors.
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
from absl.testing import absltest
from google.protobuf import text_format

from xla.backends.gpu.runtime import buffer_debug_log_pb2
from xla.backends.gpu.runtime import thunk_pb2
from xla.tools.buffer_debug_log import checksum_mismatch_report


class ChecksumMismatchReportTest(absltest.TestCase):

  def test_from_protos_loads_metadata(self):
    test_log = ""
    test_metadata = """
thunk_metadata {
  thunk_info {
    thunk_id: 100
    profile_annotation: "thunk1"
  }
  thunk_kind: "kGemm"
}
thunk_metadata {
  thunk_info {
    thunk_id: 101
    profile_annotation: "thunk2"
  }
  thunk_kind: "kConv"
}
"""
    log_proto = text_format.Parse(
        test_log, buffer_debug_log_pb2.BufferDebugLogProto()
    )
    metadata_proto = text_format.Parse(
        test_metadata,
        thunk_pb2.ThunkMetadataListProto(),
    )

    report = checksum_mismatch_report.ChecksumMismatchReport.from_protos(
        {0: log_proto}, metadata_proto
    )

    self.assertEqual(
        report.thunk_metadata,
        {
            100: checksum_mismatch_report.ThunkMetadata(
                thunk_id=100,
                thunk_kind="kGemm",
                profile_annotation="thunk1",
            ),
            101: checksum_mismatch_report.ThunkMetadata(
                thunk_id=101,
                thunk_kind="kConv",
                profile_annotation="thunk2",
            ),
        },
    )

  def test_from_protos_finds_mismatches_in_single_proto(self):
    test_log = """
entries {
  thunk_id: 100
  execution_id: 10
  buffer_idx: 0
  is_input_buffer: true
  checksum: 11111111
}
entries {
  thunk_id: 100
  execution_id: 10
  buffer_idx: 1
  is_input_buffer: false
  checksum: 22222222
}
entries {
  thunk_id: 100
  execution_id: 11
  buffer_idx: 0
  is_input_buffer: true
  checksum: 11111111
}
entries {
  thunk_id: 100
  execution_id: 11
  buffer_idx: 1
  is_input_buffer: false
  checksum: 33333333
}
"""
    test_metadata = ""
    log_proto = text_format.Parse(
        test_log, buffer_debug_log_pb2.BufferDebugLogProto()
    )
    metadata_proto = text_format.Parse(
        test_metadata,
        thunk_pb2.ThunkMetadataListProto(),
    )

    report = checksum_mismatch_report.ChecksumMismatchReport.from_protos(
        {0: log_proto}, metadata_proto
    )

    self.assertEqual(
        report.mismatches,
        {
            # thunk ID
            100: {
                # input checksums
                checksum_mismatch_report.BufferChecksums({0: 11111111}): {
                    # output buffer index => checksums
                    1: {22222222, 33333333},
                },
            },
        },
    )

  def test_from_protos_finds_mismatches_in_multiple_protos(self):
    test_log_template = """
entries {{
  thunk_id: 100
  execution_id: 10
  buffer_idx: 0
  is_input_buffer: true
  checksum: 11111111
}}
entries {{
  thunk_id: 100
  execution_id: 10
  buffer_idx: 1
  is_input_buffer: false
  checksum: {output_checksum}
}}
"""
    test_logs = [
        test_log_template.format(output_checksum=checksum)
        for checksum in [22222222, 33333333]
    ]
    test_metadata = ""
    log_protos = {
        module_id: text_format.Parse(
            test_log, buffer_debug_log_pb2.BufferDebugLogProto()
        )
        for module_id, test_log in enumerate(test_logs)
    }
    metadata_proto = text_format.Parse(
        test_metadata,
        thunk_pb2.ThunkMetadataListProto(),
    )

    report = checksum_mismatch_report.ChecksumMismatchReport.from_protos(
        log_protos, metadata_proto
    )

    self.assertEqual(
        report.mismatches,
        {
            # thunk ID
            100: {
                # input checksums
                checksum_mismatch_report.BufferChecksums({0: 11111111}): {
                    # output buffer index => checksums
                    1: {22222222, 33333333},
                },
            },
        },
    )

  def test_from_protos_does_not_include_consistent_executions(self):
    test_log = """
entries {
  thunk_id: 100
  execution_id: 10
  buffer_idx: 0
  is_input_buffer: true
  checksum: 11111111
}
entries {
  thunk_id: 100
  execution_id: 10
  buffer_idx: 1
  is_input_buffer: false
  checksum: 22222222
}
entries {
  thunk_id: 100
  execution_id: 11
  buffer_idx: 0
  is_input_buffer: true
  checksum: 11111111
}
entries {
  thunk_id: 100
  execution_id: 11
  buffer_idx: 1
  is_input_buffer: false
  checksum: 22222222
}
"""
    test_metadata = ""
    log_proto = text_format.Parse(
        test_log, buffer_debug_log_pb2.BufferDebugLogProto()
    )
    metadata_proto = text_format.Parse(
        test_metadata,
        thunk_pb2.ThunkMetadataListProto(),
    )

    report = checksum_mismatch_report.ChecksumMismatchReport.from_protos(
        {0: log_proto}, metadata_proto
    )

    self.assertEmpty(report.mismatches)


if __name__ == "__main__":
  absltest.main()
