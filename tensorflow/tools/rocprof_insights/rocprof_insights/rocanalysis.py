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
        
    def compute_advanced_stats(
        self,
        df = None,
        group_col = 'kernel_name',
        start_col = 'start_ns',
        end_col = 'end_ns'
    ):
        """Computes advanced statistics for a specified group column.

        Specifically:
            nameId = <group_col value>
            total = sum(end - start)
            num = count(*)
            percentage = (group total / overall total) * 100
            avg = mean(end - start)
            med = median(end - start)
            min = min(end - start)
            max = max(end - start)
            stddev = std(end - start)
            q1 = 25th percentile of (end - start)
            q3 = 75th percentile of (end - start)

        Args:
            df (pd.DataFrame): A DataFrame containing profiling data.
            group_col (str): Column to group by (e.g., 'kernel_name', 'api_name').
            start_col (str): Column containing the start timestamp.
            end_col (str): Column containing the end timestamp.

        Returns:
            pd.DataFrame: DataFrame with columns:
                [
                    'nameId', 'total', 'num', 'percentage', 'avg', 'med', 
                    'min', 'max', 'stddev', 'q1', 'q3'
                ]

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        # Check if the required columns exist
        required_cols = {group_col, start_col, end_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        # Make a copy to avoid modifying the original DataFrame
        temp_df = df.copy()

        # Compute duration for each row (end - start)
        temp_df['duration'] = temp_df[end_col] - temp_df[start_col]

        # Aggregate by the group_col
        grouped = temp_df.groupby(group_col)['duration'].agg(
            total='sum',
            num='count',
            avg='mean',
            med='median',
            min='min',
            max='max',
            stddev='std',
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
        ).reset_index()

        # Rename the grouping column to 'nameId' for clarity
        grouped.rename(columns={group_col: 'nameId'}, inplace=True)

        # Compute the percentage of each group's total relative to the sum of all groups
        sum_total = grouped['total'].sum()
        grouped['percentage'] = (grouped['total'] / sum_total * 100) if sum_total != 0 else 0
        grouped['total'] /= 1e9 
        grouped['avg'] /= 1e3
        grouped['med'] /= 1e3
        grouped['min'] /= 1e3
        grouped['max'] /= 1e3

        # create a rename map from your old columns to new columns
        rename_map = {
            'nameId': group_col,
            'total': 'total time [s]',
            'num': 'num calls',
            'percentage': 'percentage',
            'avg': 'avg [us]',
            'med': 'med [us]',
            'min': 'min [us]',
            'max': 'max [us]',
            'stddev': 'stddev [us]',
            'q1': 'q1',
            'q3': 'q3'
        }

        # apply the renaming
        grouped.rename(columns=rename_map, inplace=True)

        # reorder columns (optional, if you need a specific order)
        ordered_cols = [
            group_col, 
            'total time [s]', 
            'num calls', 
            'percentage', 
            'avg [us]', 
            'med [us]',
            'min [us]', 
            'max [us]', 
            'stddev [us]', 
            'q1', 
            'q3'
        ]
        return grouped[ordered_cols]

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