# Copyright 2024 The OpenXLA Authors.
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
"""Unit tests for profile_data.py."""

from absl.testing import absltest
from xla.python import xla_extension

profile_data = xla_extension.profiler


class ProfileDataTest(absltest.TestCase):

  def test_find_plane_with_name(self):
    profile = profile_data.ProfileData.from_text_proto("""
        planes { name: "a" }
        planes { name: "b" }
        """)
    self.assertEqual(profile.find_plane_with_name('a').name, 'a')
    self.assertEqual(profile.find_plane_with_name('b').name, 'b')
    self.assertIsNone(profile.find_plane_with_name('c'))

  def test_visit_space(self):
    space = profile_data.ProfileData.from_text_proto("""
        planes { name: "a" }
        planes { name: "b" }
        """)
    self.assertEqual([plane.name for plane in space.planes], ['a', 'b'])

  def test_visit_empty_space(self):
    space = profile_data.ProfileData.from_text_proto('')
    self.assertEmpty(list(space.planes))

  def test_visit_plane(self):
    profile = profile_data.ProfileData.from_text_proto("""
        planes {
          name: "p0"
          lines { name: "t1" }
          lines { name: "t2" }
          stats { metadata_id: 1 str_value: "world" }
          stat_metadata {
            key: 1
            value { name: "hello" }
          }
        }
        """)
    plane = profile.find_plane_with_name('p0')
    self.assertEqual(plane.name, 'p0')
    self.assertEqual([line.name for line in plane.lines], ['t1', 't2'])
    self.assertEqual(dict(plane.stats), {'hello': 'world'})

  def test_visit_empty_plane(self):
    profile = profile_data.ProfileData.from_text_proto('planes {}')

    plane = next(profile.planes)
    self.assertEmpty(plane.name)
    self.assertEmpty(list(plane.lines))

  def test_visit_line(self):
    profile = profile_data.ProfileData.from_text_proto("""
        planes {
          name: "p0"
          lines {
            name: "t100"
            events { metadata_id: 1 }
            events { metadata_id: 2 }
          }
          event_metadata {
            key: 1
            value { name: "foo" }
          }
          event_metadata {
            key: 2
            value { name: "bar" }
          }
        }
        """)
    plane = next(profile.planes)
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    self.assertEqual(line.name, 't100')
    self.assertListEqual([event.name for event in line.events], ['foo', 'bar'])

  def test_visit_empty_line(self):
    profile = profile_data.ProfileData.from_text_proto("""
        planes {
          name: "p0"
          lines {}
        }
        """)
    plane = next(profile.planes)
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    self.assertEmpty(line.name)
    self.assertEmpty(list(line.events))

  def test_visit_event(self):
    profile = profile_data.ProfileData.from_text_proto("""
        planes {
          name: "p0"
          lines {
            timestamp_ns: 1000
            events {
              metadata_id: 1
              offset_ps: 500000
              duration_ps: 600000
              stats { metadata_id: 1 double_value: 400.0 }
              stats { metadata_id: 2 uint64_value: 1024 }
              stats { metadata_id: 3 ref_value: 4 }
            }
          }
          event_metadata {
            key: 1
            value { name: "hlo" }
          }
          stat_metadata {
            key: 1
            value { name: "flops" }
          }
          stat_metadata {
            key: 2
            value { name: "bytes" }
          }
          stat_metadata {
            key: 3
            value { name: "provenance" }
          }
          stat_metadata {
            key: 4
            value { name: "tf_op" }
          }
        }
        """)
    plane = next(profile.planes)
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    events = list(line.events)
    self.assertLen(events, 1)
    event = events[0]
    self.assertEqual(event.start_ns, 1500.0)
    self.assertEqual(event.duration_ns, 600.0)
    self.assertEqual(event.end_ns, 2100.0)
    self.assertEqual(event.name, 'hlo')
    self.assertEqual(
        dict(event.stats),
        {'flops': 400.0, 'bytes': 1024, 'provenance': 'tf_op'},
    )

  def test_visit_event_missing_metadata(self):
    profile = profile_data.ProfileData.from_text_proto("""
        planes {
          name: "p0"
          lines {
            timestamp_ns: 1000
            events {
              metadata_id: 1
              stats { metadata_id: 1 double_value: 400.0 }
              stats { metadata_id: 2 uint64_value: 1024 }
              stats { metadata_id: 3 ref_value: 4 }
            }
          }
          stat_metadata {
            key: 1
            value { name: "flops" }
          }
          stat_metadata {
            key: 3
            value { name: "provenance" }
          }
        }
        """)
    plane = next(profile.planes)
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    events = list(line.events)
    self.assertLen(events, 1)
    event = events[0]
    self.assertEqual(event.name, '')
    self.assertEqual(
        dict(filter(lambda x: x[0] is not None, event.stats)),
        {'flops': 400.0, 'provenance': ''},
    )

  def test_create_profile_data_from_file(self):
    serialized = profile_data.ProfileData.text_proto_to_serialized_xspace("""
        planes { name: "a" }
        planes { name: "b" }
        """)
    tmp_file = self.create_tempfile().full_path
    with open(tmp_file, 'wb') as f:
      f.write(serialized)
    profile = profile_data.ProfileData.from_file(tmp_file)
    self.assertEqual([plane.name for plane in profile.planes], ['a', 'b'])


if __name__ == '__main__':
  absltest.main()
