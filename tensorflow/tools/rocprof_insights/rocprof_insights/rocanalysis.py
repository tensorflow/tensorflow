"""Perform analysis on ROCm profiler data.

Classes:
    KernelAnalyzer: Analyzes kernel dispatches (latencies, top kernels, etc.).
    ApiAnalyzer: Analyzes HIP or HSA API calls.
    MemcpyAnalyzer: Analyzes Memcpy events (H2D, D2H, D2D).
"""

import pandas as pd

class KernelAnalyzer:
    """Analyzes kernel-related profiling data."""

    def __init__(self, df):
        """Initializes KernelAnalyzer.

        Args:
            df (pd.DataFrame): DataFrame containing kernel-level data.
                              Must have columns 'kernel_name' and 'duration_ms'.
        """
        self.df = df.copy()
        required_cols = {'kernel_name', 'duration_ms'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        self.agg_df = None

    def compute_statistics(self):
        """Computes aggregated kernel statistics (total, average, count).

        Returns:
            pd.DataFrame: Aggregated DataFrame with columns:
                          ['kernel_name', 'total_duration_ms', 'avg_duration_ms', 'count'].
        """
        self.agg_df = self.df.groupby('kernel_name', as_index=False).agg(
            total_duration_ms=('duration_ms', 'sum'),
            avg_duration_ms=('duration_ms', 'mean'),
            count=('duration_ms', 'count')
        )
        self.agg_df.sort_values('total_duration_ms', ascending=False, inplace=True)
        return self.agg_df

    def top_kernels_by_duration(self, top_n=10):
        """Returns the top N kernels by total duration.

        Args:
            top_n (int): Number of top kernels to return.

        Returns:
            pd.DataFrame: Subset of the aggregated DataFrame with top N rows.
        """
        if self.agg_df is None:
            _ = self.compute_statistics()
        return self.agg_df.nlargest(top_n, 'total_duration_ms')


class ApiAnalyzer:
    """Analyzes HIP/HSA API call data."""

    def __init__(self, df, api_column='api_name'):
        """Initializes ApiAnalyzer.

        Args:
            df (pd.DataFrame): DataFrame containing API-level data.
            api_column (str): Column name that contains the API name (e.g. 'api_name').
        """
        self.df = df.copy()
        self.api_column = api_column
        required_cols = {self.api_column, 'duration_ms'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        self.agg_df = None

    def compute_statistics(self):
        """Computes aggregated statistics (total, average, count) for each API.

        Returns:
            pd.DataFrame: Aggregated DataFrame with columns:
                          [api_column, 'total_duration_ms', 'avg_duration_ms', 'count'].
        """
        self.agg_df = self.df.groupby(self.api_column, as_index=False).agg(
            total_duration_ms=('duration_ms', 'sum'),
            avg_duration_ms=('duration_ms', 'mean'),
            count=('duration_ms', 'count')
        )
        self.agg_df.sort_values('total_duration_ms', ascending=False, inplace=True)
        return self.agg_df


class MemcpyAnalyzer:
    """Analyzes Memcpy events such as H2D, D2H, D2D."""

    def __init__(self, df, memcpy_column='memcpy_type'):
        """Initializes MemcpyAnalyzer.

        Args:
            df (pd.DataFrame): DataFrame with Memcpy events data.
            memcpy_column (str): Column name that indicates the memcpy type (e.g. 'memcpy_type').
        """
        self.df = df.copy()
        self.memcpy_column = memcpy_column
        required_cols = {self.memcpy_column, 'duration_ms'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        # We assume possible values are 'H2D', 'D2H', 'D2D', but it could vary.
        # The user can filter or group by these.
        self.agg_df = None

    def compute_statistics(self):
        """Computes aggregated statistics (total, average, count) by memcpy_type.

        Returns:
            pd.DataFrame: Aggregated DataFrame with columns:
                          [memcpy_column, 'total_duration_ms', 'avg_duration_ms', 'count'].
        """
        self.agg_df = self.df.groupby(self.memcpy_column, as_index=False).agg(
            total_duration_ms=('duration_ms', 'sum'),
            avg_duration_ms=('duration_ms', 'mean'),
            count=('duration_ms', 'count')
        )
        self.agg_df.sort_values('total_duration_ms', ascending=False, inplace=True)
        return self.agg_df