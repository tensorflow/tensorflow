import os

import tensorflow.python.platform

from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.summary import event_accumulator
from tensorflow.python.summary import event_multiplexer


class _FakeAccumulator(object):

  def __init__(self, path):
    self._path = path
    self.autoupdate_called = False
    self.autoupdate_interval = None
    self.reload_called = False

  def Tags(self):
    return {event_accumulator.IMAGES: ['im1', 'im2'],
            event_accumulator.HISTOGRAMS: ['hst1', 'hst2'],
            event_accumulator.COMPRESSED_HISTOGRAMS: ['cmphst1', 'cmphst2'],
            event_accumulator.SCALARS: ['sv1', 'sv2']}

  def Scalars(self, tag_name):
    if tag_name not in self.Tags()[event_accumulator.SCALARS]:
      raise KeyError
    return ['%s/%s' % (self._path, tag_name)]

  def Histograms(self, tag_name):
    if tag_name not in self.Tags()[event_accumulator.HISTOGRAMS]:
      raise KeyError
    return ['%s/%s' % (self._path, tag_name)]

  def CompressedHistograms(self, tag_name):
    if tag_name not in self.Tags()[event_accumulator.COMPRESSED_HISTOGRAMS]:
      raise KeyError
    return ['%s/%s' % (self._path, tag_name)]

  def Images(self, tag_name):
    if tag_name not in self.Tags()[event_accumulator.IMAGES]:
      raise KeyError
    return ['%s/%s' % (self._path, tag_name)]

  def AutoUpdate(self, interval):
    self.autoupdate_called = True
    self.autoupdate_interval = interval

  def Reload(self):
    self.reload_called = True


def _GetFakeAccumulator(path, size_guidance):  # pylint: disable=unused-argument
  return _FakeAccumulator(path)


class EventMultiplexerTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(EventMultiplexerTest, self).setUp()
    event_accumulator.EventAccumulator = _GetFakeAccumulator

  def testEmptyLoader(self):
    x = event_multiplexer.EventMultiplexer()
    self.assertEqual(x.Runs(), {})

  def testRunNamesRespected(self):
    x = event_multiplexer.EventMultiplexer({'run1': 'path1', 'run2': 'path2'})
    self.assertItemsEqual(x.Runs().keys(), ['run1', 'run2'])
    self.assertEqual(x._GetAccumulator('run1')._path, 'path1')
    self.assertEqual(x._GetAccumulator('run2')._path, 'path2')

  def testReload(self):
    x = event_multiplexer.EventMultiplexer({'run1': 'path1', 'run2': 'path2'})
    self.assertFalse(x._GetAccumulator('run1').reload_called)
    self.assertFalse(x._GetAccumulator('run2').reload_called)
    x.Reload()
    self.assertTrue(x._GetAccumulator('run1').reload_called)
    self.assertTrue(x._GetAccumulator('run2').reload_called)

  def testAutoUpdate(self):
    x = event_multiplexer.EventMultiplexer({'run1': 'path1', 'run2': 'path2'})
    x.AutoUpdate(5)
    self.assertTrue(x._GetAccumulator('run1').autoupdate_called)
    self.assertEqual(x._GetAccumulator('run1').autoupdate_interval, 5)
    self.assertTrue(x._GetAccumulator('run2').autoupdate_called)
    self.assertEqual(x._GetAccumulator('run2').autoupdate_interval, 5)

  def testScalars(self):
    x = event_multiplexer.EventMultiplexer({'run1': 'path1', 'run2': 'path2'})

    run1_actual = x.Scalars('run1', 'sv1')
    run1_expected = ['path1/sv1']

    self.assertEqual(run1_expected, run1_actual)

  def testExceptions(self):
    x = event_multiplexer.EventMultiplexer({'run1': 'path1', 'run2': 'path2'})
    with self.assertRaises(KeyError):
      x.Scalars('sv1', 'xxx')

  def testInitialization(self):
    x = event_multiplexer.EventMultiplexer()
    self.assertEqual(x.Runs(), {})
    x = event_multiplexer.EventMultiplexer({'run1': 'path1', 'run2': 'path2'})
    self.assertItemsEqual(x.Runs(), ['run1', 'run2'])
    self.assertEqual(x._GetAccumulator('run1')._path, 'path1')
    self.assertEqual(x._GetAccumulator('run2')._path, 'path2')

  def testAddRunsFromDirectory(self):
    x = event_multiplexer.EventMultiplexer()
    tmpdir = self.get_temp_dir()
    join = os.path.join
    fakedir = join(tmpdir, 'fake_accumulator_directory')
    realdir = join(tmpdir, 'real_accumulator_directory')
    self.assertEqual(x.Runs(), {})
    x.AddRunsFromDirectory(fakedir)
    self.assertEqual(x.Runs(), {}, 'loading fakedir had no effect')

    if gfile.IsDirectory(realdir):
      gfile.DeleteRecursively(realdir)
    gfile.MkDir(realdir)
    x.AddRunsFromDirectory(realdir)
    self.assertEqual(x.Runs(), {}, 'loading empty directory had no effect')

    path1 = join(realdir, 'path1')
    gfile.MkDir(path1)
    x.AddRunsFromDirectory(realdir)
    self.assertEqual(x.Runs().keys(), ['path1'], 'loaded run: path1')
    loader1 = x._GetAccumulator('path1')
    self.assertEqual(loader1._path, path1, 'has the correct path')

    path2 = join(realdir, 'path2')
    gfile.MkDir(path2)
    x.AddRunsFromDirectory(realdir)
    self.assertItemsEqual(x.Runs().keys(), ['path1', 'path2'])
    self.assertEqual(x._GetAccumulator('path1'), loader1,
                     'loader1 not regenerated')
    loader2 = x._GetAccumulator('path2')

    path2_2 = join(path2, 'path2')
    gfile.MkDir(path2_2)
    x.AddRunsFromDirectory(path2)
    self.assertItemsEqual(x.Runs().keys(), ['path1', 'path2'])
    self.assertNotEqual(loader2, x._GetAccumulator('path2'),
                        'loader2 regenerated')
    self.assertEqual(x._GetAccumulator('path2')._path, path2_2,
                     'loader2 path correct')

  def testAddRunsFromDirectoryThatContainsEvents(self):
    x = event_multiplexer.EventMultiplexer()
    tmpdir = self.get_temp_dir()
    join = os.path.join
    realdir = join(tmpdir, 'event_containing_directory')

    if gfile.IsDirectory(realdir):
      gfile.DeleteRecursively(realdir)
    gfile.MkDir(realdir)

    self.assertEqual(x.Runs(), {})

    with gfile.GFile(join(realdir, 'hypothetical.tfevents.out'), 'w'):
      pass
    x.AddRunsFromDirectory(realdir)
    self.assertItemsEqual(x.Runs(), ['event_containing_directory'])

    subdir = join(realdir, 'subdir')
    gfile.MkDir(subdir)
    x.AddRunsFromDirectory(realdir)
    self.assertItemsEqual(x.Runs(), ['event_containing_directory', 'subdir'])

  def testAddRunsFromDirectoryWithRunNames(self):
    x = event_multiplexer.EventMultiplexer()
    tmpdir = self.get_temp_dir()
    join = os.path.join
    realdir = join(tmpdir, 'event_containing_directory')

    if gfile.IsDirectory(realdir):
      gfile.DeleteRecursively(realdir)
    gfile.MkDir(realdir)

    self.assertEqual(x.Runs(), {})

    with gfile.GFile(join(realdir, 'hypothetical.tfevents.out'), 'w'):
      pass
    x.AddRunsFromDirectory(realdir, 'foo')
    self.assertItemsEqual(x.Runs(), ['foo'])

    subdir = join(realdir, 'subdir')
    gfile.MkDir(subdir)
    x.AddRunsFromDirectory(realdir, 'foo')
    self.assertItemsEqual(x.Runs(), ['foo', 'foo/subdir'])

  def testAddRunsFromDirectoryThrowsException(self):
    x = event_multiplexer.EventMultiplexer()
    tmpdir = self.get_temp_dir()

    filepath = os.path.join(tmpdir, 'bad_file')
    with gfile.GFile(filepath, 'w'):
      pass

    with self.assertRaises(ValueError):
      x.AddRunsFromDirectory(filepath)

  def testAddRun(self):
    x = event_multiplexer.EventMultiplexer()
    x.AddRun('run1_path', 'run1')
    run1 = x._GetAccumulator('run1')
    self.assertEqual(x.Runs().keys(), ['run1'])
    self.assertEqual(run1._path, 'run1_path')

    x.AddRun('run1_path', 'run1')
    self.assertEqual(run1, x._GetAccumulator('run1'), 'loader not recreated')

    x.AddRun('run2_path', 'run1')
    new_run1 = x._GetAccumulator('run1')
    self.assertEqual(new_run1._path, 'run2_path')
    self.assertNotEqual(run1, new_run1)

    x.AddRun('runName3')
    self.assertItemsEqual(x.Runs().keys(), ['run1', 'runName3'])
    self.assertEqual(x._GetAccumulator('runName3')._path, 'runName3')

  def testAddRunMaintainsLoading(self):
    x = event_multiplexer.EventMultiplexer()
    x.Reload()
    x.AddRun('run1')
    x.AddRun('run2')
    self.assertTrue(x._GetAccumulator('run1').reload_called)
    self.assertTrue(x._GetAccumulator('run2').reload_called)

  def testAddRunMaintainsAutoUpdate(self):
    x = event_multiplexer.EventMultiplexer()
    x.AutoUpdate(5)
    x.AddRun('run1')
    x.AddRun('run2')
    self.assertTrue(x._GetAccumulator('run1').autoupdate_called)
    self.assertTrue(x._GetAccumulator('run2').autoupdate_called)
    self.assertEqual(x._GetAccumulator('run1').autoupdate_interval, 5)
    self.assertEqual(x._GetAccumulator('run2').autoupdate_interval, 5)

if __name__ == '__main__':
  googletest.main()
