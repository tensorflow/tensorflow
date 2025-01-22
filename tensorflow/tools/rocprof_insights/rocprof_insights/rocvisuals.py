"""Module containing a class that generates various Plotly plots 
from the DataFrame produced by rocanalysis.py

Classes:
    RocprofStatsVisualizer: Provides methods to create histograms, pie charts,
                       and bar charts, scatter plots for kernels, APIs, 
                       
    MemoryCopyVisualizer: memcpy events.

"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class RocprofStatsVisualizer:
    """Class for creating various plots from a DataFrame of advanced stats.

    This class expects a DataFrame with columns:
        [
            'name',
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

    Attributes:
        df (pd.DataFrame): The DataFrame containing these columns.
    """

    def __init__(self, df: pd.DataFrame, required_cols={}):
        """Initializes the StatsVisualizer.

        Args:
            df (pd.DataFrame): A DataFrame with the advanced stats columns.
        """
        """
        required_cols = {
            'name', 'total time [s]', 'num calls', 'percentage', 'avg [us]',
            'med [us]', 'min [us]', 'max [us]', 'stddev [ns]', 'q1', 'q3'
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        """
        self.df = df.copy()

    def histogram_a_field(
        self,
        df: pd.DataFrame = None,
        field: str = 'total time [s]',
        nbins: int = 50,
        title: str = "Histogram of Total Duration",
        xaxis_title = 'Total duration of kernels [s]',
        output_file: str = None,
    ):
        """Plots a histogram of the 'field' column.
        
        Args:
            nbins (int): Number of histogram bins.
            title (str): Title of the plot.
            output_file (str): If provided, saves the figure to this file (PNG) at 400 DPI.
        """
        if df is not None:
            self.df = df.copy()
            
        if field not in self.df.columns:
            raise ValueError(f"DataFrame must contain {field} column.")

        fig = px.histogram(
            self.df,
            x=field,
            nbins=nbins,
            title=title,
        )
        
        fig.update_layout(
            template="plotly_white",
            # Center the title
            title=dict(
                text=title,
                x=0.5,         # 0.5 = center; 0 = left, 1 = right
                xanchor='center'
            ),
            # Add (or override) axis labels:
            xaxis_title=field,
            yaxis_title="Count [-]"  # or whatever label you want
        )

        if output_file:
            # Save as PNG at ~400 DPI using scale=4
            fig.write_image(output_file, scale=4)
        else:
            fig.show()

    def pie_chart_top10_a_field(
        self,
        field: str = 'kernel_name',
        sort_field: str = 'total time [s]',
        title: str = "Pie Chart of Top 10 by Total Time (s)",
        output_file: str = None
    ):
        """Creates a pie chart of the top 10 items by 'total time [s]'.

        Args:
            title (str): Title of the plot.
            output_file (str): If provided, saves the figure to this file (PNG) at 400 DPI.
        """
        # Sort by total time descending
        sorted_df = self.df.sort_values(by=sort_field, ascending=False)
        top10 = sorted_df.head(10)
        
        def wrap_text(text, width=50):
            # break long lines at ~width characters
            return "<br>".join([text[i:i+width] for i in range(0, len(text), width)])

        wrapped_col = field + '_wrapped'
        top10[wrapped_col] = top10[field].apply(wrap_text)

        fig = px.pie(
            top10,
            names=wrapped_col,
            values=sort_field,
            title=title,
        )

        # Update the figure layout
        # Make the pie chart bigger and more readable
        fig.update_layout(
            width=1200,
            height=900,
            margin=dict(l=200, r=50, t=50, b=100),  # extra space on the left, etc.

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        if output_file:
            fig.write_image(output_file, scale=4)
        else:
            fig.show()

    def bar_chart_top10_a_field(
        self,
        column: str = 'kernel_name',
        title: str = "Top 10 by Total Time (s)",
        output_file: str = None
    ):
        """Creates a bar chart of the top 10 items by 'total time [s]'.

        Args:
            title (str): Title of the plot.
            output_file (str): If provided, saves figure to this file (PNG) at 400 DPI.
        """
        sorted_df = self.df.sort_values(by='total time [s]', ascending=False)
        top10 = sorted_df.head(10)

        fig = px.bar(
            top10,
            x=column,
            y='total time [s]',
            # title=title,
        )
        fig.update_layout(template="plotly_white", xaxis={'categoryorder':'total descending'})
        fig.update_xaxes(tickangle=45)

        if output_file:
            fig.write_image(output_file, scale=4)
        else:
            fig.show()

    def box_plot_time(
        self,
        y_column: str = 'total time [s]',
        title: str = "Box Plot of Time",
        output_file: str = None
    ):
        """Creates a box plot for a specified time column (e.g., total time [s], avg [us], etc.).

        Args:
            y_column (str): Name of the numeric column to plot on the Y axis.
            title (str): Title of the plot.
            output_file (str): If provided, saves figure to this file (PNG) at 400 DPI.
        """
        if y_column not in self.df.columns:
            raise ValueError(f"DataFrame must contain '{y_column}' column.")

        fig = px.box(
            self.df,
            y=y_column,
            title=title,
        )
        fig.update_layout(template="plotly_white")

        if output_file:
            fig.write_image(output_file, scale=4)
        else:
            fig.show()

    def scatter_avg_vs_num_calls(
        self,
        field: str = 'kernel_name',
        title: str = "Scatter Plot: avg [us] vs. num calls",
        output_file: str = None
    ):
        """Example scatter plot comparing average time to the number of calls.

        Args:
            title (str): Title of the plot.
            output_file (str): If provided, saves figure to this file (PNG) at 400 DPI.
        """
        # assume 'avg [us]' and 'num calls' exist
        if 'avg [us]' not in self.df.columns or 'num calls' not in self.df.columns:
            raise ValueError("DataFrame must contain 'avg [us]' and 'num calls' columns.")

        fig = px.scatter(
            self.df,
            x='num calls',
            y='avg [us]',
            hover_data=[field],
            title=title
        )
        fig.update_layout(template="plotly_white")

        if output_file:
            fig.write_image(output_file, scale=4)
        else:
            fig.show()
            
            
class MemoryCopyVisualizer:
    """Analyzes and visualize memory copy operations (H2D, D2H, D2D) from a CSV trace.

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

    def plot_distribution(self, nbins: int = 50, output_file: str = None) -> None:
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
        
        fig_h2d.update_layout(
            template="plotly_white",
            # Center the title
            title=dict(
                x=0.5,         # 0.5 = center; 0 = left, 1 = right
                xanchor='center'
            ),
            # Add (or override) axis labels:
            xaxis_title='Latency [ns]',
            yaxis_title="Count [-]"  # or whatever label you want
        )

        if output_file:
            fig_h2d.write_image(output_file + '_H2D.png', scale=4)
        else:
            fig_h2d.show()

        # Histogram for DEVICE_TO_HOST
        fig_d2h = px.histogram(
            self.device_to_host,
            x='Duration',
            title='Distribution of MEMORY_COPY_DEVICE_TO_HOST Durations',
            labels={'Duration': 'Duration (ns)'},
            nbins=nbins
        )
        
        fig_d2h.update_layout(
            template="plotly_white",
            # Center the title
            title=dict(
                x=0.5,         # 0.5 = center; 0 = left, 1 = right
                xanchor='center'
            ),
            # Add (or override) axis labels:
            xaxis_title='Latency [ns]',
            yaxis_title="Count [-]"  # or whatever label you want
        )

        if output_file:
            fig_d2h.write_image(output_file + '_D2H.png', scale=4)
        else:
            fig_d2h.show()
        

    def plot_direction_pie(self, output_file: str = None) -> None:
        """Plots a pie chart showing total duration distribution by copy direction.

        This aggregates the DataFrame over 'Direction' by summing Duration.
        Assumes each row's duration has been calculated (via `calculate_duration()`).
        """
        # Merge both subsets back, or just use the entire data if you computed duration for all
        combined = pd.concat([self.host_to_device, self.device_to_host], ignore_index=True)

        # Sum duration by direction
        direction_agg = combined.groupby('Direction', as_index=False)['Duration'].sum()

        fig = px.pie(
            direction_agg,
            names='Direction',
            values='Duration',
            title='Total Memory Copy Duration by Direction'
        )
        
        fig.update_layout(template="plotly_white")

        if output_file:
            fig.write_image(output_file + '_pie_chart.png', scale=4)
        else:
            fig.show()

    def analyze(self, dist_output_file: str = None, pie_output_file: str = None) -> None:
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
        self.plot_distribution(output_file = dist_output_file)
        self.plot_direction_pie(output_file = pie_output_file)