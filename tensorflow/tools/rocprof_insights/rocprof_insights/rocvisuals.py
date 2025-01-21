"""Module containing a class that generates various Plotly plots 
from the DataFrame produced by rocanalysis.py

Classes:
    RocprofStatsVisualizer: Provides methods to create histograms, pie charts,
                       and bar charts, scatter plots for kernels, APIs, and memcpy events.

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
            'med [us]', 'min [us]', 'max [us]', 'stddev [us]', 'q1', 'q3'
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        """
        self.df = df.copy()

    def histogram_total_time(
        self,
        nbins: int = 50,
        title: str = "Histogram of Total Time",
        output_file: str = None
    ):
        """Plots a histogram of the 'total time [s]' column.

        Args:
            nbins (int): Number of histogram bins.
            title (str): Title of the plot.
            output_file (str): If provided, saves the figure to this file (PNG) at 400 DPI.
        """
        if 'total time [s]' not in self.df.columns:
            raise ValueError("DataFrame must contain 'total time [s]' column.")

        fig = px.histogram(
            self.df,
            x='total time [s]',
            nbins=nbins,
            title=title,
        )
        fig.update_layout(template="plotly_white")

        if output_file:
            # Save as PNG at ~400 DPI using scale=4
            fig.write_image(output_file, scale=4)
        else:
            fig.show()

    def pie_chart_top10_total_time(
        self,
        column: str = 'kernel_name',
        title: str = "Pie Chart of Top 10 by Total Time (s)",
        output_file: str = None
    ):
        """Creates a pie chart of the top 10 items by 'total time [s]'.

        Args:
            title (str): Title of the plot.
            output_file (str): If provided, saves the figure to this file (PNG) at 400 DPI.
        """
        # Sort by total time descending
        sorted_df = self.df.sort_values(by='total time [s]', ascending=False)
        top10 = sorted_df.head(10)

        fig = px.pie(
            top10,
            names=column,
            values='total time [s]',
            title=title
        )
        fig.update_layout(template="plotly_white")

        # Update the figure layout
        # Make the pie chart bigger and more readable
        fig.update_layout(
            width=1920,
            height=1080,
            margin=dict(l=50, r=50, t=80, b=50),
            legend_title_text=column,
        )

        if output_file:
            fig.write_image(output_file, scale=4)
        else:
            fig.show()

    def bar_chart_top10_total_time(
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
        column: str = 'kernel_name',
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
            hover_data=[column],
            title=title
        )
        fig.update_layout(template="plotly_white")

        if output_file:
            fig.write_image(output_file, scale=4)
        else:
            fig.show()