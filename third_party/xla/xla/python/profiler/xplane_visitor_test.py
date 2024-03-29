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

from absl.testing import absltest

from xla.python.profiler import xplane_visitor  # pylint: disable=g-direct-tensorflow-import
from tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import


class XplaneVisitorTest(absltest.TestCase):

  def test_find_plane_with_name(self):
    space = xplane_pb2.XSpace(
        planes=[
            xplane_pb2.XPlane(name='a'),
            xplane_pb2.XPlane(name='b'),
        ]
    )
    self.assertEqual(xplane_visitor.find_plane_with_name(space, 'a').name, 'a')
    self.assertEqual(xplane_visitor.find_plane_with_name(space, 'b').name, 'b')
    self.assertIsNone(xplane_visitor.find_plane_with_name(space, 'c'))

  def test_visit_space(self):
    space = xplane_visitor.XSpaceVisitor(xplane_pb2.XSpace(
        planes=[
            xplane_pb2.XPlane(name='a'),
            xplane_pb2.XPlane(name='b'),
        ]
    ))
    self.assertEqual([plane.name for plane in space.planes], ['a', 'b'])

  def test_visit_empty_space(self):
    space = xplane_visitor.XSpaceVisitor(xplane_pb2.XSpace())
    self.assertEmpty(list(space.planes))

  def test_visit_plane(self):
    plane = xplane_visitor.XPlaneVisitor(
        xplane_pb2.XPlane(
            name='p0',
            lines=[
                xplane_pb2.XLine(name='t1'),
                xplane_pb2.XLine(name='t2'),
            ],
            stats=[xplane_pb2.XStat(metadata_id=1, str_value='world')],
            stat_metadata={1: xplane_pb2.XStatMetadata(name='hello')},
        )
    )
    self.assertEqual(plane.name, 'p0')
    self.assertEqual([line.name for line in plane.lines], ['t1', 't2'])
    self.assertEqual(dict(plane.stats), {'hello': 'world'})

  def test_visit_empty_plane(self):
    plane = xplane_visitor.XPlaneVisitor(xplane_pb2.XPlane())
    self.assertEmpty(plane.name)
    self.assertEmpty(list(plane.lines))

  def test_visit_line(self):
    plane = xplane_visitor.XPlaneVisitor(
        xplane_pb2.XPlane(
            lines=[
                xplane_pb2.XLine(
                    name='t100',
                    events=[
                        xplane_pb2.XEvent(metadata_id=1),
                        xplane_pb2.XEvent(metadata_id=2),
                    ],
                )
            ],
            event_metadata={
                1: xplane_pb2.XEventMetadata(name='foo'),
                2: xplane_pb2.XEventMetadata(name='bar'),
            },
        )
    )
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    self.assertEqual(line.name, 't100')
    self.assertListEqual([event.name for event in line.events], ['foo', 'bar'])

  def test_visit_empty_line(self):
    plane = xplane_visitor.XPlaneVisitor(
        xplane_pb2.XPlane(lines=[xplane_pb2.XLine()])
    )
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    self.assertEmpty(line.name)
    self.assertEmpty(list(line.events))

  def test_visit_event(self):
    plane = xplane_visitor.XPlaneVisitor(
        xplane_pb2.XPlane(
            lines=[
                xplane_pb2.XLine(
                    timestamp_ns=1000,
                    events=[
                        xplane_pb2.XEvent(
                            metadata_id=1,
                            offset_ps=500000,
                            duration_ps=600000,
                            stats=[
                                xplane_pb2.XStat(
                                    metadata_id=1, double_value=400.0
                                ),
                                xplane_pb2.XStat(
                                    metadata_id=2, uint64_value=1024
                                ),
                                xplane_pb2.XStat(metadata_id=3, ref_value=4),
                            ],
                        )
                    ],
                )
            ],
            event_metadata={
                1: xplane_pb2.XEventMetadata(name='hlo'),
            },
            stat_metadata={
                1: xplane_pb2.XStatMetadata(name='flops'),
                2: xplane_pb2.XStatMetadata(name='bytes'),
                3: xplane_pb2.XStatMetadata(name='provenance'),
                4: xplane_pb2.XStatMetadata(name='tf_op'),
            },
        )
    )
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
    plane = xplane_visitor.XPlaneVisitor(
        xplane_pb2.XPlane(
            lines=[
                xplane_pb2.XLine(
                    events=[
                        xplane_pb2.XEvent(
                            metadata_id=1,
                            stats=[
                                xplane_pb2.XStat(
                                    metadata_id=1, double_value=400.0
                                ),
                                xplane_pb2.XStat(
                                    metadata_id=2, uint64_value=1024
                                ),
                                xplane_pb2.XStat(metadata_id=3, ref_value=4),
                            ],
                        )
                    ]
                )
            ],
            stat_metadata={
                1: xplane_pb2.XStatMetadata(name='flops'),
                3: xplane_pb2.XStatMetadata(name='provenance'),
            },
        )
    )
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    events = list(line.events)
    self.assertLen(events, 1)
    event = events[0]
    self.assertEqual(event.name, '')
    self.assertEqual(dict(event.stats), {'flops': 400.0, 'provenance': ''})


if __name__ == '__main__':
  absltest.main()
