# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Download manager interface."""

import concurrent.futures
import hashlib
import typing
from typing import Dict, Iterator, Optional, Tuple, Union
import uuid

from absl import logging
import promise
import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.download import checksums
from tensorflow.data.experimental.core.download import downloader
from tensorflow.data.experimental.core.download import extractor
from tensorflow.data.experimental.core.download import kaggle
from tensorflow.data.experimental.core.download import resource as resource_lib
from tensorflow.data.experimental.core.download import util
from tensorflow.data.experimental.core.utils import type_utils

# pylint: disable=logging-format-interpolation

Tree = type_utils.Tree
ReadOnlyPath = type_utils.ReadOnlyPath
ReadWritePath = type_utils.ReadWritePath

Url = Union[str, resource_lib.Resource]
ExtractPath = Union[type_utils.PathLike, resource_lib.Resource]


class NonMatchingChecksumError(Exception):
  """The downloaded file doesn't have expected checksum."""

  def __init__(self, url, tmp_path):
    msg = (
        f'Artifact {url}, downloaded to {tmp_path}, has wrong checksum. '
        'To debug, see: '
        'https://www.tensorflow.org/datasets/overview#fixing_nonmatchingchecksumerror'
    )
    Exception.__init__(self, msg)


class DownloadConfig(object):
  """Configuration for `tfds.core.DatasetBuilder.download_and_prepare`."""

  def __init__(
      self,
      extract_dir=None,
      manual_dir=None,
      download_mode=None,
      compute_stats=None,
      max_examples_per_split=None,
      register_checksums=False,
      force_checksums_validation=False,
      beam_runner=None,
      beam_options=None,
      try_download_gcs=True,
      verify_ssl=True,
  ):
    """Constructs a `DownloadConfig`.

    Args:
      extract_dir: `str`, directory where extracted files are stored.
        Defaults to "<download_dir>/extracted".
      manual_dir: `str`, read-only directory where manually downloaded/extracted
        data is stored. Defaults to `<download_dir>/manual`.
      download_mode: `tfds.GenerateMode`, how to deal with downloads or data
        that already exists. Defaults to `REUSE_DATASET_IF_EXISTS`, which will
        reuse both downloads and data if it already exists.
      compute_stats: `tfds.download.ComputeStats`, whether to compute
        statistics over the generated data. Defaults to `AUTO`.
      max_examples_per_split: `int`, optional max number of examples to write
        into each split (used for testing).
      register_checksums: `bool`, defaults to False. If True, checksum of
        downloaded files are recorded.
      force_checksums_validation: `bool`, defaults to False. If True, raises
        an error if an URL do not have checksums.
      beam_runner: Runner to pass to `beam.Pipeline`, only used for datasets
        based on Beam for the generation.
      beam_options: `PipelineOptions` to pass to `beam.Pipeline`, only used for
        datasets based on Beam for the generation.
      try_download_gcs: `bool`, defaults to True. If True, prepared dataset
        will be downloaded from GCS, when available. If False, dataset will be
        downloaded and prepared from scratch.
      verify_ssl: `bool`, defaults to True. If True, will verify certificate
        when downloading dataset.
    """
    self.extract_dir = extract_dir
    self.manual_dir = manual_dir
    self.download_mode = util.GenerateMode(
        download_mode or util.GenerateMode.REUSE_DATASET_IF_EXISTS)
    self.compute_stats = util.ComputeStatsMode(
        compute_stats or util.ComputeStatsMode.SKIP)
    self.max_examples_per_split = max_examples_per_split
    self.register_checksums = register_checksums
    self.force_checksums_validation = force_checksums_validation
    self.beam_runner = beam_runner
    self.beam_options = beam_options
    self.try_download_gcs = try_download_gcs
    self.verify_ssl = verify_ssl


class DownloadManager(object):
  """Manages the download and extraction of files, as well as caching.

  Downloaded files are cached under `download_dir`. The file name of downloaded
   files follows pattern "{sanitized_url}{content_checksum}.{ext}". Eg:
   'cs.toronto.edu_kriz_cifar-100-pythonJDF[...]I.tar.gz'.

  While a file is being downloaded, it is placed into a directory following a
  similar but different pattern:
  "{sanitized_url}{url_checksum}.tmp.{uuid}".

  When a file is downloaded, a "{fname}.INFO.json" file is created next to it.
  This INFO file contains the following information:
  {"dataset_names": ["name1", "name2"],
   "urls": ["http://url.of/downloaded_file"]}

  Extracted files/dirs are stored under `extract_dir`. The file name or
  directory name is the same as the original name, prefixed with the extraction
  method. E.g.
   "{extract_dir}/TAR_GZ.cs.toronto.edu_kriz_cifar-100-pythonJDF[...]I.tar.gz".

  The function members accept either plain value, or values wrapped into list
  or dict. Giving a data structure will parallelize the downloads.

  Example of usage:

  ```
  # Sequential download: str -> str
  train_dir = dl_manager.download_and_extract('https://abc.org/train.tar.gz')
  test_dir = dl_manager.download_and_extract('https://abc.org/test.tar.gz')

  # Parallel download: list -> list
  image_files = dl_manager.download(
      ['https://a.org/1.jpg', 'https://a.org/2.jpg', ...])

  # Parallel download: dict -> dict
  data_dirs = dl_manager.download_and_extract({
     'train': 'https://abc.org/train.zip',
     'test': 'https://abc.org/test.zip',
  })
  data_dirs['train']
  data_dirs['test']
  ```

  For more customization on the download/extraction (ex: passwords, output_name,
  ...), you can pass a `tfds.download.Resource` as argument.
  """

  def __init__(
      self,
      *,
      download_dir: type_utils.PathLike,
      extract_dir: Optional[type_utils.PathLike] = None,
      manual_dir: Optional[type_utils.PathLike] = None,
      manual_dir_instructions: Optional[str] = None,
      url_infos: Optional[Dict[str, checksums.UrlInfo]] = None,
      dataset_name: Optional[str] = None,
      force_download: bool = False,
      force_extraction: bool = False,
      force_checksums_validation: bool = False,
      register_checksums: bool = False,
      register_checksums_path: Optional[type_utils.PathLike] = None,
      verify_ssl: bool = True,
  ):
    """Download manager constructor.

    Args:
      download_dir: Path to directory where downloads are stored.
      extract_dir: Path to directory where artifacts are extracted.
      manual_dir: Path to manually downloaded/extracted data directory.
      manual_dir_instructions: Human readable instructions on how to
        prepare contents of the manual_dir for this dataset.
      url_infos: Urls info for the checksums.
      dataset_name: Name of dataset this instance will be used for. If
        provided, downloads will contain which datasets they were used for.
      force_download: If True, always [re]download.
      force_extraction: If True, always [re]extract.
      force_checksums_validation: If True, raises an error if an URL do not
        have checksums.
      register_checksums: If True, dl checksums aren't
        checked, but stored into file.
      register_checksums_path: Path were to save checksums. Should be set
        if register_checksums is True.
      verify_ssl: `bool`, defaults to True. If True, will verify certificate
        when downloading dataset.

    Raises:
      FileNotFoundError: Raised if the register_checksums_path does not exists.
    """
    if register_checksums:
      if not register_checksums_path:
        raise ValueError(
            'When register_checksums=True, register_checksums_path should be set.'
        )
      register_checksums_path = utils.as_path(register_checksums_path)
      if not register_checksums_path.exists():
        # Create the file here to make sure user has write access before
        # starting downloads.
        register_checksums_path.touch()
      else:
        # Make sure the user has write access before downloading any files.
        # (e.g. TFDS installed by admin)
        register_checksums_path.write_text(register_checksums_path.read_text())

    download_dir = utils.as_path(download_dir).expanduser()
    if extract_dir:
      extract_dir = utils.as_path(extract_dir).expanduser()
    else:
      extract_dir = download_dir / 'extracted'
    if manual_dir:
      manual_dir = utils.as_path(manual_dir).expanduser()

    self._download_dir: ReadWritePath = download_dir
    self._extract_dir: ReadWritePath = extract_dir
    self._manual_dir: Optional[ReadOnlyPath] = manual_dir
    self._manual_dir_instructions = utils.dedent(manual_dir_instructions)
    self._download_dir.mkdir(parents=True, exist_ok=True)
    self._extract_dir.mkdir(parents=True, exist_ok=True)

    self._force_download = force_download
    self._force_extraction = force_extraction
    self._force_checksums_validation = force_checksums_validation
    self._register_checksums = register_checksums
    self._register_checksums_path = register_checksums_path
    self._verify_ssl = verify_ssl
    self._dataset_name = dataset_name

    # All known URLs: {url: UrlInfo(size=, checksum=)}
    self._url_infos = checksums.get_all_url_infos()
    if url_infos is not None:
      self._url_infos.update(url_infos)

    # To record what is being used: {url: UrlInfo(size, checksum, filename)}
    self._recorded_url_infos: Dict[str, checksums.UrlInfo] = {}
    # These attributes are lazy-initialized since they must be cleared when this
    # object is pickled for Beam. They are then recreated on each worker.
    self.__downloader = None
    self.__extractor = None
    # Executor to avoid blocking other download/extractions when running I/O
    # operations (reading/renaming download file).
    # Only use a single thread as the read/ops are locked by the
    # `build_synchronize_decorator`.
    # Note: This thread is in additions of the download and extraction
    # executors threads.
    self._executor = concurrent.futures.ThreadPoolExecutor(1)

  def __getstate__(self):
    """Remove un-pickleable attributes and return the state."""
    if self._register_checksums:
      # Currently, checksums registration from Beam not supported.
      raise NotImplementedError(
          '`register_checksums` must be disabled in a parallelized '
          'DownloadManager. Please open a PR if you would like this feature.')
    state = self.__dict__.copy()
    state['_DownloadManager__downloader'] = None
    state['_DownloadManager__extractor'] = None
    state['_executor'] = None
    return state

  @property
  def _downloader(self):
    if not self.__downloader:
      self.__downloader = downloader.get_downloader()
    return self.__downloader

  @property
  def _extractor(self):
    if not self.__extractor:
      self.__extractor = extractor.get_extractor()
    return self.__extractor

  @property
  def downloaded_size(self):
    """Returns the total size of downloaded files."""
    return sum(url_info.size for url_info in self._recorded_url_infos.values())

  def _get_dl_path(self, url: str, sha256: str) -> ReadWritePath:
    return self._download_dir / resource_lib.get_dl_fname(url, sha256)

  @property
  def register_checksums(self):
    """Returns whether checksums are being computed and recorded to file."""
    return self._register_checksums

  @utils.build_synchronize_decorator()
  def _record_url_infos(self):
    """Store in file when recorded size/checksum of downloaded files."""
    checksums.save_url_infos(
        self._register_checksums_path,
        self._recorded_url_infos,
    )

  # Synchronize and memoize decorators ensure same resource will only be
  # processed once, even if passed twice to download_manager.
  @utils.build_synchronize_decorator()
  @utils.memoize()
  def _download(
      self, resource: Union[str, resource_lib.Resource]
  ) -> promise.Promise[ReadOnlyPath]:
    """Download resource, returns Promise->path to downloaded file.

    This function:

    1. Reuse cache (`_get_cached_path`) or download the file
    2. Register or validate checksums (`_register_or_validate_checksums`)
    3. Rename download to final path (`_rename_and_get_final_dl_path`)

    Args:
      resource: The URL to download.

    Returns:
      path: The path to the downloaded resource.
    """
    # Normalize the input
    if isinstance(resource, str):
      resource = resource_lib.Resource(url=resource)
    url = resource.url

    expected_url_info = self._url_infos.get(url)

    # 3 possible destinations for the path:
    # * In `manual_dir` (manually downloaded data)
    # * In `downloads/url_path` (checksum unknown)
    # * In `downloads/checksum_path` (checksum registered)
    manually_downloaded_path = _get_manually_downloaded_path(
        manual_dir=self._manual_dir,
        expected_url_info=expected_url_info,
    )
    url_path = self._get_dl_path(
        url, sha256=hashlib.sha256(url.encode('utf-8')).hexdigest()
    )
    checksum_path = self._get_dl_path(
        url, sha256=expected_url_info.checksum
    ) if expected_url_info else None

    # Get the cached path and url_info (if they exists)
    dl_result = _get_cached_path(
        manually_downloaded_path=manually_downloaded_path,
        checksum_path=checksum_path,
        url_path=url_path,
        expected_url_info=expected_url_info,
    )
    if dl_result.path and not self._force_download:  # Download was cached
      logging.info(
          f'Skipping download of {url}: File cached in {dl_result.path}'
      )
      future = promise.Promise.resolve(dl_result)
    else:
      # Download in an empty tmp directory (to avoid name collisions)
      # `download_tmp_dir` is cleaned-up in `_rename_and_get_final_dl_path`
      dirname = f'{resource_lib.get_dl_dirname(url)}.tmp.{uuid.uuid4().hex}'
      download_tmp_dir = self._download_dir / dirname
      download_tmp_dir.mkdir()
      logging.info(f'Downloading {url} into {download_tmp_dir}...')
      future = self._downloader.download(
          url, download_tmp_dir, verify=self._verify_ssl
      )

    # Post-process the result
    return future.then(lambda dl_result: self._register_or_validate_checksums(  # pylint: disable=g-long-lambda
        url=url,
        path=dl_result.path,
        computed_url_info=dl_result.url_info,
        expected_url_info=expected_url_info,
        checksum_path=checksum_path,
        url_path=url_path,
    ))

  def _register_or_validate_checksums(
      self,
      path: ReadWritePath,
      url: str,
      expected_url_info: Optional[checksums.UrlInfo],
      computed_url_info: Optional[checksums.UrlInfo],
      checksum_path: Optional[ReadWritePath],
      url_path: ReadWritePath,
  ) -> ReadOnlyPath:
    """Validates/records checksums and renames final downloaded path."""
    # `path` can be:
    # * Manually downloaded
    # * (cached) checksum_path
    # * (cached) url_path
    # * `tmp_dir/file` (downloaded path)

    if computed_url_info:
      # Used both in `.downloaded_size` and `_record_url_infos()`
      self._recorded_url_infos[url] = computed_url_info

    if self._register_checksums:
      if not computed_url_info:
        raise ValueError(
            f'Cannot register checksums for {url}: no computed chechsum. '
            '--register_checksums with manually downloaded data not supported.'
        )
      # Note:
      # * We save even if `expected_url_info == computed_url_info` as
      #   `expected_url_info` might have been loaded from another dataset.
      # * `register_checksums_path` was validated in `__init__` so this
      #   shouldn't fail.
      self._record_url_infos()

      # Checksum path should now match the new registered checksum (even if
      # checksums were previously registered)
      expected_url_info = computed_url_info
      checksum_path = self._get_dl_path(url, computed_url_info.checksum)
    else:
      # Eventually validate checksums
      # Note:
      # * If path is cached at `url_path` but cached
      #   `computed_url_info != expected_url_info`, a new download has
      #   been triggered (as _get_cached_path returns None)
      # * If path was downloaded but checksums don't match expected, then
      #   the download isn't cached (re-running build will retrigger a new
      #   download). This is expected as it might mean the downloaded file
      #   was corrupted. Note: The tmp file isn't deleted to allow inspection.
      _validate_checksums(
          url=url,
          path=path,
          expected_url_info=expected_url_info,
          computed_url_info=computed_url_info,
          force_checksums_validation=self._force_checksums_validation,
      )

    return self._rename_and_get_final_dl_path(
        url=url,
        path=path,
        expected_url_info=expected_url_info,
        computed_url_info=computed_url_info,
        checksum_path=checksum_path,
        url_path=url_path,
    )

  def _rename_and_get_final_dl_path(
      self,
      url: str,
      path: ReadWritePath,
      expected_url_info: Optional[checksums.UrlInfo],
      computed_url_info: Optional[checksums.UrlInfo],
      checksum_path: Optional[ReadWritePath],
      url_path: ReadWritePath,
  ) -> ReadWritePath:
    """Eventually rename the downloaded file if checksums were recorded."""
    # `path` can be:
    # * Manually downloaded
    # * (cached) checksum_path
    # * (cached) url_path
    # * `tmp_dir/file` (downloaded path)
    if path.is_relative_to(self._manual_dir):  # Manually downloaded data
      return path
    elif path == checksum_path:  # Path already at final destination
      assert computed_url_info == expected_url_info  # Sanity check
      return checksum_path  # pytype: disable=bad-return-type
    elif path == url_path:
      if checksum_path:
        # Checksums were registered: Rename -> checksums_path
        resource_lib.rename_info_file(path, checksum_path, overwrite=True)
        return path.replace(checksum_path)
      else:
        # Checksums not registered: -> do nothing
        return path
    else:  # Path was downloaded in tmp dir
      dst_path = checksum_path or url_path
      resource_lib.write_info_file(
          url=url,
          path=dst_path,
          dataset_name=self._dataset_name,
          original_fname=path.name,
          url_info=computed_url_info,
      )
      path.replace(dst_path)
      path.parent.rmdir()  # Cleanup tmp dir (will fail if dir not empty)
      return dst_path

  @utils.build_synchronize_decorator()
  @utils.memoize()
  def _extract(self, resource: ExtractPath) -> promise.Promise[ReadOnlyPath]:
    """Extract a single archive, returns Promise->path to extraction result."""
    if isinstance(resource, type_utils.PathLikeCls):
      resource = resource_lib.Resource(path=resource)
    path = resource.path
    extract_method = resource.extract_method
    if extract_method == resource_lib.ExtractMethod.NO_EXTRACT:
      logging.info('Skipping extraction for %s (method=NO_EXTRACT).', path)
      return promise.Promise.resolve(path)
    method_name = resource_lib.ExtractMethod(extract_method).name
    extract_path = self._extract_dir / f'{method_name}.{path.name}'
    if not self._force_extraction and extract_path.exists():
      logging.info('Reusing extraction of %s at %s.', path, extract_path)
      return promise.Promise.resolve(extract_path)
    return self._extractor.extract(path, extract_method, extract_path)

  @utils.build_synchronize_decorator()
  @utils.memoize()
  def _download_extract(self, resource):
    """Download-extract `Resource` or url, returns Promise->path."""
    if isinstance(resource, str):
      resource = resource_lib.Resource(url=resource)
    def callback(path):
      resource.path = path
      return self._extract(resource)
    return self._download(resource).then(callback)

  def download_checksums(self, checksums_url):
    """Downloads checksum file from the given URL and adds it to registry."""
    checksums_path = self.download(checksums_url)
    self._url_infos.update(checksums.load_url_infos(checksums_path))

  def download_kaggle_data(self, competition_or_dataset: str) -> ReadWritePath:
    """Download data for a given Kaggle Dataset or competition.

    Note: This function requires the Kaggle CLI tool.
    Read the installation guide at https://www.kaggle.com/docs/api.

    Args:
      competition_or_dataset: Dataset name (`zillow/zecon`) or
        competition name (`titanic`)

    Returns:
      The path to the downloaded files.
    """
    return kaggle.download_kaggle_data(
        competition_or_dataset, self._download_dir)

  @typing.overload
  def download(self, url_or_urls: Url) -> ReadOnlyPath:
    ...

  @typing.overload
  def download(self, url_or_urls: Dict[str, Url]) -> Dict[str, ReadOnlyPath]:
    ...

  @typing.overload
  def download(self, url_or_urls: Tree[Url]) -> Tree[ReadOnlyPath]:
    ...

  def download(self, url_or_urls):
    """Download given url(s).

    Args:
      url_or_urls: url or `list`/`dict` of urls to download and extract. Each
        url can be a `str` or `tfds.download.Resource`.

    Returns:
      downloaded_path(s): `str`, The downloaded paths matching the given input
        url_or_urls.
    """
    # Add progress bar to follow the download state
    with self._downloader.tqdm():
      return _map_promise(self._download, url_or_urls)

  def iter_archive(
      self, resource: ExtractPath,
  ) -> Iterator[Tuple[str, typing.BinaryIO]]:
    """Returns iterator over files within archive.

    **Important Note**: caller should read files as they are yielded.
    Reading out of order is slow.

    Args:
      resource: path to archive or `tfds.download.Resource`.

    Returns:
      Generator yielding tuple (path_within_archive, file_obj).
    """
    if isinstance(resource, type_utils.PathLikeCls):
      resource = resource_lib.Resource(path=resource)
    return extractor.iter_archive(resource.path, resource.extract_method)

  @typing.overload
  def extract(self, path_or_paths: ExtractPath) -> ReadOnlyPath:
    ...

  @typing.overload
  def extract(
      self, path_or_paths: Dict[str, ExtractPath]
  ) -> Dict[str, ReadOnlyPath]:
    ...

  @typing.overload
  def extract(self, path_or_paths: Tree[ExtractPath]) -> Tree[ReadOnlyPath]:
    ...

  def extract(self, path_or_paths):
    """Extract given path(s).

    Args:
      path_or_paths: path or `list`/`dict` of path of file to extract. Each
        path can be a `str` or `tfds.download.Resource`.

    If not explicitly specified in `Resource`, the extraction method is deduced
    from downloaded file name.

    Returns:
      extracted_path(s): `str`, The extracted paths matching the given input
        path_or_paths.
    """
    # Add progress bar to follow the download state
    with self._extractor.tqdm():
      return _map_promise(self._extract, path_or_paths)

  @typing.overload
  def download_and_extract(self, url_or_urls: Url) -> ReadOnlyPath:
    ...

  @typing.overload
  def download_and_extract(
      self, url_or_urls: Dict[str, Url]
  ) -> Dict[str, ReadOnlyPath]:
    ...

  @typing.overload
  def download_and_extract(self, url_or_urls: Tree[Url]) -> Tree[ReadOnlyPath]:
    ...

  def download_and_extract(self, url_or_urls):
    """Download and extract given url_or_urls.

    Is roughly equivalent to:

    ```
    extracted_paths = dl_manager.extract(dl_manager.download(url_or_urls))
    ```

    Args:
      url_or_urls: url or `list`/`dict` of urls to download and extract. Each
        url can be a `str` or `tfds.download.Resource`.

    If not explicitly specified in `Resource`, the extraction method will
    automatically be deduced from downloaded file name.

    Returns:
      extracted_path(s): `str`, extracted paths of given URL(s).
    """
    # Add progress bar to follow the download state
    with self._downloader.tqdm():
      with self._extractor.tqdm():
        return _map_promise(self._download_extract, url_or_urls)

  @property
  def download_dir(self) -> ReadOnlyPath:
    return self._download_dir

  @utils.memoized_property
  def manual_dir(self) -> ReadOnlyPath:
    """Returns the directory containing the manually extracted data."""
    if not self._manual_dir:
      raise AssertionError('Manual directory not enabled.')
    if not self._manual_dir_instructions:
      raise ValueError(
          'To access `dl_manager.manual_dir`, please set '
          '`MANUAL_DOWNLOAD_INSTRUCTIONS` in your dataset.'
      )
    if not self._manual_dir.exists() or not list(self._manual_dir.iterdir()):
      raise AssertionError(
          f'Manual directory {self._manual_dir} does not exist or is empty. '
          'Create it and download/extract dataset artifacts in there using '
          f'instructions:\n{self._manual_dir_instructions}'
      )
    return self._manual_dir


def _get_cached_path(
    manually_downloaded_path: Optional[ReadWritePath],
    checksum_path: Optional[ReadWritePath],
    url_path: ReadWritePath,
    expected_url_info: Optional[checksums.UrlInfo],
) -> downloader.DownloadResult:
  """Returns the downloaded path and computed url-info.

  If the path is not cached, or that `url_path` does not match checksums,
  the file will be downloaded again.

  Path can be cached at three different locations:

  Args:
    manually_downloaded_path: Manually downloaded in `dl_manager.manual_dir`
    checksum_path: Cached in the final destination (if checksum known)
    url_path: Cached in the tmp destination (if checksum unknown).
    expected_url_info: Registered checksum (if known)
  """
  # User has manually downloaded the file.
  if manually_downloaded_path and manually_downloaded_path.exists():
    return downloader.DownloadResult(manually_downloaded_path, url_info=None)  # pytype: disable=wrong-arg-types

  # Download has been cached (checksum known)
  elif checksum_path and resource_lib.Resource.exists_locally(checksum_path):
    # `path = f(checksum)` was found, so url_info match
    return downloader.DownloadResult(checksum_path, url_info=expected_url_info)

  # Download has been cached (checksum unknown)
  elif resource_lib.Resource.exists_locally(url_path):
    # Info restored from `.INFO` file
    computed_url_info = _read_url_info(url_path)
    # If checksums are now registered but do not match, trigger a new
    # download (e.g. previous file corrupted, checksums updated)
    if expected_url_info and computed_url_info != expected_url_info:
      return downloader.DownloadResult(path=None, url_info=None)  # pytype: disable=wrong-arg-types
    else:
      return downloader.DownloadResult(url_path, url_info=computed_url_info)

  # Else file not found (or has bad checksums). (re)download.
  else:
    return downloader.DownloadResult(path=None, url_info=None)  # pytype: disable=wrong-arg-types


def _get_manually_downloaded_path(
    manual_dir: Optional[ReadOnlyPath],
    expected_url_info: Optional[checksums.UrlInfo],
) -> Optional[ReadOnlyPath]:
  """Checks if file is already downloaded in manual_dir."""
  if not manual_dir:  # Manual dir not passed
    return None

  if not expected_url_info or not expected_url_info.filename:
    return None  # Filename unknown.

  manual_path = manual_dir / expected_url_info.filename
  if not manual_path.exists():  # File not manually downloaded
    return None

  return manual_path


def _validate_checksums(
    url: str,
    path: ReadOnlyPath,
    computed_url_info: Optional[checksums.UrlInfo],
    expected_url_info: Optional[checksums.UrlInfo],
    force_checksums_validation: bool,
) -> None:
  """Validate computed_url_info match expected_url_info."""
  # If force-checksums validations, both expected and computed url_info
  # should exists
  if force_checksums_validation:
    # Checksum of the downloaded file unknown (for manually downloaded file)
    if not computed_url_info:
      computed_url_info = utils.read_checksum_digest(path)
    # Checksums have not been registered
    if not expected_url_info:
      raise ValueError(
          f'Missing checksums url: {url}, yet '
          '`force_checksums_validation=True`. '
          'Did you forgot to register checksums ?'
      )

  if (
      expected_url_info
      and computed_url_info
      and expected_url_info != computed_url_info
  ):
    raise NonMatchingChecksumError(url, path)


def _read_url_info(url_path: type_utils.PathLike) -> checksums.UrlInfo:
  """Loads the `UrlInfo` from the `.INFO` file."""
  file_info = resource_lib.read_info_file(url_path)
  if 'url_info' not in file_info:
    raise ValueError(
        'Could not found `url_info` in {}. This likelly indicates that '
        'the files where downloaded with a previous version of TFDS (<=3.1.0). '
    )
  url_info = file_info['url_info']
  url_info.setdefault('filename', None)
  return checksums.UrlInfo(**url_info)


def _map_promise(map_fn, all_inputs):
  """Map the function into each element and resolve the promise."""
  all_promises = tf.nest.map_structure(map_fn, all_inputs)  # Apply the function
  res = tf.nest.map_structure(lambda p: p.get(), all_promises)  # Wait promises
  return res
