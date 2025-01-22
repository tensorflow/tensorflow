"""Perform analysis on ROCm profiler data.

Classes:
    RocAnalyzer: 
    - Analyzes kernel dispatches (latencies, top kernels, etc.).
    - Analyzes HIP or HSA API calls.
    - Analyzes Memcpy events (H2D, D2H, D2D).
"""

import pandas as pd

class RocAnalyzer:
    """Analyzes roc profiling data."""

    def __init__(self, df, required_cols=None):
        """Initializes KernelAnalyzer.

        Args:
            df (pd.DataFrame): DataFrame containing kernel-level data.
                              Must have columns 'kernel_name' and 'duration_ms'.
        """
        self.df = df.copy()
        if required_cols is None:
            required_cols = {'kernel_name', 'duration_us'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        self.agg_df = None
        
    def compute_advanced_stats(
        self,
        df = None,
        group_col = 'kernel_name',
        start_col = 'start_ts',
        end_col = 'end_ts'
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
        if df is None:
            df = self.df 
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
            'stddev': 'stddev [ns]',
            'q1': 'q1 [ns]',
            'q3': 'q3 [ns]'
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
            'stddev [ns]', 
            'q1 [ns]', 
            'q3 [ns]',
        ]
        return grouped[ordered_cols]
    
    
class MemoryCopyAnalyzer:
    """Analyzes memory copy operations (H2D, D2H, D2D) from a CSV trace.

    This class provides methods to:
      - Load trace data into a pandas DataFrame.
      - Filter data by copy direction.
      - Calculate copy durations.
      - Plot histograms for HOST_TO_DEVICE and DEVICE_TO_HOST durations.
      - Plot a pie chart showing total duration distribution by copy direction.
    """

    def __init__(self, file_path: str):
        """Initializes the MemoryCopyAnalyzer with a path to the CSV trace file.

        Args:
            file_path: The file path to the CSV trace data.
        """
        self.file_path = file_path
        self.data = None
        self.host_to_device = None
        self.device_to_host = None

    def load_data(self) -> None:
        """Loads the CSV data into a pandas DataFrame.

        Raises:
            FileNotFoundError: If the specified file_path cannot be found.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        self.data = pd.read_csv(self.file_path)

    def filter_data(self) -> None:
        """Filters the DataFrame into separate subsets for H2D and D2H.

        Assumes the `Direction` column contains values:
          - 'MEMORY_COPY_HOST_TO_DEVICE'
          - 'MEMORY_COPY_DEVICE_TO_HOST'
          - (Optional) 'MEMORY_COPY_DEVICE_TO_DEVICE' if present.
        """
        self.host_to_device = self.data[self.data['Direction'] == 'MEMORY_COPY_HOST_TO_DEVICE']
        self.device_to_host = self.data[self.data['Direction'] == 'MEMORY_COPY_DEVICE_TO_HOST']
        self.device_to_device = self.data[self.data['Direction'] == 'MEMORY_COPY_DEVICE_TO_DEVICE']

    def calculate_duration(self) -> None:
        """Calculates the duration of each memory copy operation in-place.

        Duration is computed as:
            duration = End_Timestamp - Start_Timestamp (in nanoseconds).

        Raises:
            KeyError: If 'End_Timestamp' or 'Start_Timestamp' columns are missing.
        """
        # Calculate for HOST_TO_DEVICE
        self.host_to_device['Duration'] = (
            self.host_to_device['End_Timestamp'] - self.host_to_device['Start_Timestamp']
        )
        # Calculate for DEVICE_TO_HOST
        self.device_to_host['Duration'] = (
            self.device_to_host['End_Timestamp'] - self.device_to_host['Start_Timestamp']
        )

    def plot_distribution(self, nbins: int = 50) -> None:
        """Plots histograms of memory copy durations for H2D and D2H.

        Args:
            nbins: Number of bins to use in each histogram.
        """
        # Histogram for HOST_TO_DEVICE
        fig_h2d = px.histogram(
            self.host_to_device,
            x='Duration',
            title='Distribution of MEMORY_COPY_HOST_TO_DEVICE Durations',
            labels={'Duration': 'Duration (ns)'},
            nbins=nbins
        )
        fig_h2d.show()

        # Histogram for DEVICE_TO_HOST
        fig_d2h = px.histogram(
            self.device_to_host,
            x='Duration',
            title='Distribution of MEMORY_COPY_DEVICE_TO_HOST Durations',
            labels={'Duration': 'Duration (ns)'},
            nbins=nbins
        )
        fig_d2h.show()

    def plot_direction_pie(self) -> None:
        """Plots a pie chart showing total duration distribution by copy direction.

        This aggregates the DataFrame over 'Direction' by summing Duration.
        Assumes each row's duration has been calculated (via `calculate_duration()`).
        """
        # Merge both subsets back, or just use the entire data if you computed duration for all
        # For demonstration, let's temporarily combine the two DataFrames:
        combined = pd.concat([self.host_to_device, self.device_to_host], ignore_index=True)
        # If you have a self.device_to_device, you can include it here as well.

        # Sum duration by direction
        direction_agg = combined.groupby('Direction', as_index=False)['Duration'].sum()

        fig = px.pie(
            direction_agg,
            names='Direction',
            values='Duration',
            title='Total Memory Copy Duration by Direction'
        )
        fig.show()

    def analyze(self) -> None:
        """Runs the full analysis pipeline: Load, filter, compute durations, plot.

        This function:
          1. Loads data from CSV.
          2. Separates H2D and D2H memory copies.
          3. Calculates durations for each.
          4. Plots their histograms.
          5. Plots a pie chart of directions (H2D, D2H, optionally D2D).
        """
        self.load_data()
        self.filter_data()
        self.calculate_duration()
        self.plot_distribution()
        self.plot_direction_pie()