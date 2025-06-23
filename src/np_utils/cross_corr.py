import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

try:
    import numba as nb
except ImportError:
    raise ImportError("Please install 'numba' in your environment to use this module.")


class CrossCorrelogram:
    """
    A class to compute and plot cross-correlograms between two sets of event times.

    Attributes:
        deltat (int): Time window in milliseconds.
        bin_size (float): Bin size in milliseconds.
        normalize (bool): Normalize the cross-correlogram by the square root of the product of the number of events in each dataframe.

    Methods:
        prepare_dfs(dataframes): Format dataframes for cross-correlogram analysis.
        compute(df1, df2, start_time=None, end_time=None): Compute the cross-correlogram between the Time column of two dataframes.
        plot(cross_corr, from_="", to_=""): Plot the cross-correlogram.
        plot_comparison(cross_corr1, cross_corr2, from_="", to_="", label1=None, label2=None): Plot the cross-correlograms of two dataframes for comparison.
        compute_from_db(fs_id, channel_from, channel_to, start_time, end_time, normalize=True, check_for_triggers=False, keep_dfs=True): Compute the cross-correlogram between two channels from the database.
        compute_cc_grid_from_db(sources, targets, fs_id, start_time, end_time, check_for_triggers=False): Compute a grid of cross-correlograms between sources and targets from the database.
        plot_cc_grid(cross_corrs, channels_i=None, channels_j=None, n_rows=8, n_cols=4, avoid_symmetry=True): PLot a grid of cross-correlograms.
    """

    def __init__(self, deltat: int, bin_size: float, normalize: bool = True):
        """
        Create a CrossCorrelogram object with the given time window and bin size.

        Args:
            deltat (int): Time window in milliseconds.
            bin_size (float): Bin size in milliseconds.
            normalize (bool, optional): Normalize the cross-correlogram by the square root of the product of the number of events in each dataframe.
        """
        self._deltat = deltat
        self._bin_size = bin_size
        self.normalize = normalize

        self._n_bins = None

    @property
    def n_bins(self):
        if self._n_bins is None:
            self._n_bins = int(2 * self.deltat / self.bin_size)
            if self._n_bins % 2 == 0:
                self._n_bins += 1
        return self._n_bins

    @property
    def bin_size(self):
        return self._bin_size

    @bin_size.setter
    def bin_size(self, value):
        if value <= 0:
            raise ValueError("Bin size must be greater than 0.")
        if value > self.deltat:
            raise ValueError("Bin size must be less than or equal to deltat.")
        if value >= self.deltat / 2:
            log.warning(
                "Bin size is greater than half of deltat. This may lead to uninterpretable results."
            )

        self._bin_size = value
        self._n_bins = None

    @property
    def deltat(self):
        return self._deltat

    @deltat.setter
    def deltat(self, value):
        self._deltat = value
        self._n_bins = None

    def prepare_dfs(self, dataframes):
        """
        Format dataframes for cross-correlogram analysis.

        Args:
            dataframes (list of pd.DataFrame): List of dataframes to be formatted.

        Returns:
            list of pd.DataFrame: List of formatted dataframes.
        """
        for df in dataframes:
            df = df.copy()
            df["Time"] = pd.to_datetime(df["Time"])
            df["channel"] = df["channel"].astype(int)
        return dataframes

    @staticmethod
    @nb.njit(parallel=True)
    def _cross_correlogram(
        events_i,
        events_j,
        deltat,
        bin_size,
        normalize=True,
        hide_center_bin=False,
    ):
        """
        Compute the cross-correlogram between two sets of event times.

        Args:
            events_i (np.ndarray): Array of event times from the first dataframe.
            events_j (np.ndarray): Array of event times from the second dataframe.
            deltat (int): Time window in milliseconds.
            bin_size (float): Bin size in milliseconds.
            normalize (bool): Normalize the cross-correlogram by the square root of the product of the number of events in each dataframe.

        Returns:
            np.ndarray: Cross-correlogram array.
        """
        n_i = len(events_i)
        n_j = len(events_j)
        n_bins = int(2 * deltat / bin_size)
        if n_bins % 2 == 0:
            n_bins += 1
        cross_corr = np.zeros(n_bins)

        # Implementation 1 (faster, despite nested loop)
        for i in nb.prange(n_i):
            local_corr = np.zeros(n_bins)
            for j in range(n_j):
                delta = events_j[j] - events_i[i]
                if np.abs(delta) <= deltat:
                    bin_idx = min(int((delta + deltat) / bin_size), n_bins - 1)
                    local_corr[bin_idx] += 1
            cross_corr += local_corr

        # Implementation 2 (slower, likely due to np.where and np.floor + type conversion)
        # for i in nb.prange(n_i):
        #     indices = np.where(np.abs(events_j - events_i[i]) <= deltat)[0]
        #     lag = events_j[indices] - events_i[i]
        #     bin_ids = np.floor(lag / bin_size).astype(np.int32) + n_bins // 2
        #     cross_corr[bin_ids] += 1

        if normalize:
            cross_corr /= np.sqrt(n_i * n_j)

        if hide_center_bin:
            cross_corr[n_bins // 2] = 0

        return cross_corr

    def compute(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        start_time=None,
        end_time=None,
        normalize=None,
        hide_center_bin=False,
    ) -> np.ndarray:
        """
        Compute the cross-correlogram between the Time column of two dataframes.

        Args:
            df1 (pd.DataFrame): First dataframe containing event times.
            df2 (pd.DataFrame): Second dataframe containing event times.
            start_time (pd.Timestamp, optional): Start time for the analysis. If None, the minimum time from both dataframes is used.
            end_time (pd.Timestamp, optional): End time for the analysis. If None, the maximum time from both dataframes is used.
            normalize (bool, optional): Normalize the cross-correlogram by the square root of the product of the number of events in each dataframe. Defaults to True.
            hide_center_bin (bool, optional): Hide the center bin of the cross-correlogram. Defaults to False.

        Returns:
            np.ndarray: Cross-correlogram array.
        """
        if start_time is None:
            start_time = min(df1["Time"].min(), df2["Time"].min())
        if end_time is None:
            end_time = max(df1["Time"].max(), df2["Time"].max())

        df1 = df1.copy()
        df2 = df2.copy()

        df1["Time"] = (df1["Time"] - start_time).dt.total_seconds() * 1e3
        df2["Time"] = (df2["Time"] - start_time).dt.total_seconds() * 1e3

        events_i = df1["Time"].values
        events_j = df2["Time"].values

        return self._cross_correlogram(
            events_i,
            events_j,
            self.deltat,
            self.bin_size,
            normalize=self.normalize if normalize is None else normalize,
            hide_center_bin=hide_center_bin,
        )

    def compute_from_db(
        self,
        fs_id,
        channel_from,
        channel_to,
        start_time,
        end_time,
        normalize=None,
        check_for_triggers=False,
        keep_dfs=True,
        hide_center_bin=False,
    ) -> np.ndarray:
        """
        Compute the cross-correlogram between two channels from the database.

        Note : requires neuroplatform access to be used.

        Args:
            fs_id (str): ID of the fileset.
            channel_from (int): ID of the first channel.
            channel_to (int): ID of the second channel.
            start_time (pd.Timestamp): Start time for the analysis.
            end_time (pd.Timestamp): End time for the analysis.
            normalize (bool, optional): Normalize the cross-correlogram by the square root of the product of the number of events in each dataframe. Defaults to True.
            check_for_triggers (bool, optional): Check for triggers in the provided times for the channels. Defaults to False.
            keep_dfs (bool, optional): If True, returns the dataframes used for the analysis. Defaults to True.
            hide_center_bin (bool, optional): Hide the center bin of the cross-correlogram. Defaults to False.
        """
        try:
            from neuroplatform import Database
        except ImportError:
            raise ImportError("Neuroplatform access is required to use this function.")
        db = Database()

        df_all = db.get_spike_event(start=start_time, stop=end_time, fsname=fs_id)
        df_i = df_all[df_all["channel"] == channel_from]
        df_j = df_all[df_all["channel"] == channel_to]

        if check_for_triggers:
            trigs = db.get_all_triggers(start_time, end_time)
            if len(trigs) > 0:
                print(
                    f"Found {len(trigs)} triggers in the provided time range. Double-check whether there was stimulation on the provided channels, as this will alter the cross-correlogram."
                )

        cc = self.compute(
            df_i,
            df_j,
            start_time,
            end_time,
            normalize=self.normalize if normalize is None else normalize,
            hide_center_bin=hide_center_bin,
        )
        if keep_dfs:
            return cc, df_i, df_j
        return cc

    def _get_firing_rate(self, df, smoothing_window=1):
        """
        Compute the firing rate of the two dataframes.

        Args:
            df (pd.DataFrame): Dataframe containing event times.
            smoothing_window (int, optional): Window size for smoothing the firing rate. Defaults to 1, which means no smoothing.

        Returns:
            pd.DataFrame: Dataframe containing the firing rate of the dataframe.
        """
        df = df.copy(deep=True)
        df.drop(columns=["Amplitude"], inplace=True)
        df.index = pd.DatetimeIndex(df["Time"])
        firing_rate = df.groupby("channel").resample("1s").size()
        firing_rate = firing_rate.reset_index(name="Firing rate")

        if smoothing_window > 1:
            firing_rate["Firing rate"] = (
                firing_rate.groupby("channel")["Firing rate"]
                .rolling(smoothing_window)
                .mean()
                .reset_index(drop=True)
            )

        return firing_rate

    def _plot_firing_rate(
        self, dfs, ax=None, start_time=None, end_time=None, smoothing_window=1
    ):
        """
        Plot the firing rate of the two dataframes.

        Args:
            dfs (list of pd.DataFrame): List of dataframes to be plotted.
            ax (plt.Axes, optional): Axes object to plot on. If None, a new figure is created.
            start_time (pd.Timestamp, optional): Start time for the analysis. If None, the minimum time from both dataframes is used.
            end_time (pd.Timestamp, optional): End time for the analysis. If None, the maximum time from both dataframes is used.
            smoothing_window (int, optional): Window size for smoothing the firing rate. Defaults to 1, which means no smoothing.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(20, 10))
        all_dfs = []
        for df in dfs:
            firing_rate = self._get_firing_rate(df, smoothing_window=smoothing_window)
            all_dfs.append(firing_rate)
        df_all = pd.concat(all_dfs).reset_index(drop=False)
        sns.lineplot(
            data=df_all,
            x="Time",
            y="Firing rate",
            hue="channel",
            ax=ax,
            alpha=0.7,
            palette="tab10",
        )
        ax.set_yscale("log")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title(f'Firing rate of channels {sorted(df_all["channel"].unique())}')
        ax.set_xlim(start_time, end_time)

    def plot(self, cross_corr: np.ndarray, from_: str = "", to_: str = "", ax=None):
        """
        Plot the cross-correlogram.

        Args:
            cross_corr (np.ndarray): Cross-correlogram array to be plotted.
            from_ (str, optional): Name of the first dataframe for the plot title.
            to_ (str, optional): Name of the second dataframe for the plot title.
            ax (plt.Axes, optional): Axes object to plot on. If None, a new figure is created.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        self._plot(cross_corr, ax, from_=from_, to_=to_)

    def plot_comparison(
        self,
        cross_corr1: np.ndarray,
        cross_corr2: np.ndarray,
        from_: str = "",
        to_: str = "",
        ax=None,
        label1=None,
        label2=None,
    ):
        """
        Plot the cross-correlograms of two dataframes for comparison.

        Args:
            cross_corr1 (np.ndarray): Cross-correlogram array of the first dataframe.
            cross_corr2 (np.ndarray): Cross-correlogram array of the second dataframe.
            from_ (str, optional): Name of the first dataframe for the plot title.
            to_ (str, optional): Name of the second dataframe for the plot title.
            ax (plt.Axes, optional): Axes object to plot on. If None, a new figure is created.
            label1 (str, optional): Label for the first dataframe.
            label2 (str, optional): Label for the second dataframe.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        self._plot(
            cross_corr1,
            ax,
            from_=from_,
            to_=to_,
            color="blue",
            alpha=0.5,
            label=label1,
        )
        self._plot(
            cross_corr2,
            ax,
            from_=from_,
            to_=to_,
            color="red",
            alpha=0.5,
            label=label2,
        )

    def _plot(
        self,
        cross_corr: np.ndarray,
        ax,
        from_: str = "",
        to_: str = "",
        color=None,
        alpha=1.0,
        label=None,
    ):
        ax.bar(
            np.arange(-self.deltat, self.deltat + self.bin_size, self.bin_size),
            cross_corr,
            width=self.bin_size,
            color=color,
            alpha=alpha,
            label=label,
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Count")
        title = "Cross-correlogram"
        if from_:
            title += f" from {from_}"
            if to_:
                title += f" to {to_}"
        ax.set_title(title)
        ax.axvline(
            x=0,
            color="black",
            linestyle="--",
            alpha=0.5,
        )

    # def compute_from_dict(self, df_dict):
    #     """Compute all pairwise cross-correlograms from a dictionary of dataframes."""
    #     ccs = {}
    #     for channel, df in tqdm(
    #         df_dict.items(), desc="Computing cross-correlograms"
    #     ):
    #         cross_corr = {}
    #         for other_channel, other_df in df_dict.items():
    #             if channel != other_channel:
    #                 cross_corr[other_channel] = self.compute(df, other_df)
    #         ccs[channel] = cross_corr

    #     return ccs

    # def plot_from_dict(self, ccs, channels=None):
    #     """Plot all pairwise cross-correlograms from a dictionary of cross-correlograms."""
    #     for i, (channel, cross_corrs) in enumerate(ccs.items()):
    #         if channels is not None and i not in channels:
    #             continue
    #         n_cols = 4
    #         n_rows = int(np.ceil(len(cross_corrs) / n_cols))
    #         fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    #         axes = axes.flatten()
    #         for j, (other_channel, cross_corr) in enumerate(
    #             cross_corrs.items()
    #         ):
    #             self._plot(
    #                 cross_corr,
    #                 from_=str(channel),
    #                 to_=str(other_channel),
    #                 ax=axes[j],
    #             )
    #         plt.tight_layout()
    #         plt.show()

    @staticmethod
    def _plot_cross_corr_3d(cross_corrs, bin_sizes, deltat, reverse=False):
        colors = plt.get_cmap("tab10").colors
        yticks = range(len(bin_sizes))

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Bin size (ms)")
        ax.set_zlabel("Count")
        ax.set_yticks(yticks)

        min_bin_size = min(bin_sizes)
        xs = np.arange(-deltat, deltat + min_bin_size, min_bin_size)
        if reverse:
            cross_corrs = list(reversed(cross_corrs))
        for i, (c, k) in enumerate(zip(colors, yticks)):
            cross_corr_data = cross_corrs[i]
            cross_corr_data = (cross_corr_data - cross_corr_data.min()) / (
                cross_corr_data.max() - cross_corr_data.min()
            )
            ys = np.repeat(cross_corr_data, len(xs) // len(cross_corr_data))
            padding = (len(xs) - len(ys)) // 2
            ys = np.pad(ys, (padding, len(xs) - len(ys) - padding), "constant")

            cs = [c] * len(xs)

            ax.bar(
                xs,
                ys,
                zs=k,
                zdir="y",
                color=cs,
                alpha=1.0,
            )
            for x, y in zip(xs, ys):
                ax.plot([x, x], [k, k], [0, y], color="black", alpha=0.5)

        ax.set_yticklabels(bin_sizes if not reverse else list(reversed(bin_sizes)))
        plt.show()

    def compute_cc_grid_from_db(
        self,
        sources,
        targets,
        fs_id,
        start_time,
        end_time,
        check_for_triggers=False,
    ):
        """Compute a grid of cross-correlograms between sources and targets from the database.

        Note : requires neuroplatform access to be used.

        Args:
            sources (list): List of source channels.
            targets (list): List of target channels.
            fs_id (str): ID of the fileset.
            start_time (pd.Timestamp): Start time for the analysis.
            end_time (pd.Timestamp): End time for the analysis.
            check_for_triggers (bool, optional): Check for triggers in the provided times for the channels. Defaults to False.

        Returns:
            np.array: 3D array of cross-correlograms. Shape is n_sources x n_targets x n_bins.
        """
        results = np.zeros(
            (len(sources), len(targets), 2 * self.deltat // self.bin_size + 1)
        )

        for i, source in enumerate(
            tqdm(sources, desc="Computing cross-correlograms...")
        ):
            for j, target in enumerate(targets):
                try:
                    cross_corr = self.compute_from_db(
                        fs_id,
                        source,
                        target,
                        start_time,
                        end_time,
                        check_for_triggers=check_for_triggers,
                        keep_dfs=False,
                    )
                    results[i, j] = cross_corr
                except KeyboardInterrupt:
                    return
                except Exception as e:
                    print(f"Error from {source} to {target} : {e}")
                    continue

        return results

    def plot_cc_grid(
        self,
        cross_corrs: np.array,
        channels_i=None,
        channels_j=None,
        n_rows=8,
        n_cols=4,
        avoid_symmetry=True,
    ):
        """PLot a grid of cross-correlograms.

        Args:
            cross_corrs (np.array): 3D array of cross-correlograms. Shape should be n_sources x n_targets x n_bins. If a cross-correlogram is not available, it should be set to all zeros.
            channels_i (list, optional): List of channels to plot. If None, all channels are plotted. Defaults to None.
            channels_j (list, optional): List of channels to plot. If None, all channels are plotted. Defaults to None.
            n_rows (int, optional): Number of rows in the grid. Defaults to 8.
            n_cols (int, optional): Number of columns in the grid. Defaults to 4.
            avoid_symmetry (bool, optional): If True, only the upper triangle of the grid is plotted. Defaults to True.
        """

        if not isinstance(cross_corrs, np.ndarray) or len(cross_corrs.shape) != 3:
            raise ValueError(
                "cross_corrs should be a 3D numnpy array of size : n_sources x n_targets x n_bins"
            )

        channels_i = range(cross_corrs.shape[0]) if channels_i is None else sorted(channels_i)
        channels_j = range(cross_corrs.shape[1]) if channels_j is None else sorted(channels_j)

        sources = range(cross_corrs.shape[0])
        targets = range(cross_corrs.shape[1])
        
        for i in tqdm(sources, desc="Plotting cross-correlograms..."):
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 40))
            axes = axes.flatten()
            tot = 0
            for j in targets:
                if avoid_symmetry and j < i:
                    continue
                try:
                    cc = cross_corrs[i, j]
                    if cc.sum() == 0:
                        continue
                    ax = axes[tot]
                    self.plot(cc, from_=str(channels_i[i]), to_=str(channels_j[j]), ax=ax)
                except KeyboardInterrupt:
                    return
                except Exception as e:
                    print(f"Error from {i} to {j} : {e}")
                    continue
                tot += 1

            # Hide any unused subplots
            for k in range(tot, len(axes)):
                fig.delaxes(axes[k])

            plt.tight_layout()
            plt.show()
