from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import libfmp.b

from chord_rec.constants import LEGEND_COLORS
from chord_rec.chromagram import ChromagramConfig, compute_chromagram
from chord_rec.recognition import ChordRecognizer, chord_recognition_template, get_chord_labels
from chord_rec.evaluation import EvaluationResult, compute_eval_measures, convert_chord_ann_matrix
from chord_rec.data_utils import Song


# ---------------------------------------------------------------------------
# Chromagram plots
# ---------------------------------------------------------------------------

def plot_chromagram_annotation(
    ax: list,
    X: np.ndarray,
    Fs_X: float,
    ann: list,
    color_ann: dict,
    x_dur: float,
    cmap: str = 'gray_r',
    title: str = '',
) -> None:
    """Plot a chromagram with a chord annotation overlay."""
    libfmp.b.plot_chromagram(
        X, Fs=Fs_X, ax=ax,
        chroma_yticks=[0, 4, 7, 11], clim=[0, 1], cmap=cmap,
        title=title, ylabel='Chroma', colorbar=True,
    )
    libfmp.b.plot_segments_overlay(
        ann, ax=ax[0], time_max=x_dur,
        print_labels=False, colors=color_ann, alpha=0.1,
    )


def plot_chromagrams(
    songs: list[Song],
    indices: list[int],
    Fs_X_dict_STFT: dict[int, float],
    X_dict_STFT: dict[int, np.ndarray],
    Fs_X_dict_CQT: dict[int, float],
    X_dict_CQT: dict[int, np.ndarray],
    Fs_X_dict_IIR: dict[int, float],
    X_dict_IIR: dict[int, np.ndarray],
    cmap: str = 'gray_r',
    xlim: list[float] | None = None,
) -> None:
    """Plot STFT, CQT and IIR chromagrams side by side for each selected song."""
    for i in indices:
        fig, ax = plt.subplots(
            1, 3,
            gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [2]},
            figsize=(10, 2.5),
        )
        name = songs[i].name
        for ax_i, (version, X_dict, Fs_X_dict) in enumerate([
            ('STFT', X_dict_STFT, Fs_X_dict_STFT),
            ('CQT', X_dict_CQT, Fs_X_dict_CQT),
            ('IIR', X_dict_IIR, Fs_X_dict_IIR),
        ]):
            libfmp.b.plot_chromagram(
                X_dict[i], Fs=Fs_X_dict[i], ax=[ax[ax_i]],
                chroma_yticks=[0, 4, 7, 11], clim=[0, 1], cmap=cmap,
                title=f'{name}, {version}',
                ylabel='Chroma', xlabel='Time (seconds)',
                colorbar=True, xlim=xlim,
            )
        plt.tight_layout()


def plot_chromagram(
    songs: list[Song],
    indices: list[int],
    Fs_X_dict: dict[int, float],
    X_dict: dict[int, np.ndarray],
    version: Literal['STFT', 'CQT', 'IIR'] = 'STFT',
    cmap: str = 'gray_r',
    xlim: list[float] | None = None,
    figsize: tuple[float, float] = (3.5, 2.5),
) -> None:
    """Plot a single chromagram version for each selected song."""
    for i in indices:
        fig, ax = plt.subplots(
            1, 1,
            gridspec_kw={'width_ratios': [1], 'height_ratios': [2]},
            figsize=figsize,
        )
        title = f'{songs[i].name}, {version} ({Fs_X_dict[i]:.1f} Hz)'
        libfmp.b.plot_chromagram(
            X_dict[i], Fs=Fs_X_dict[i], ax=[ax],
            chroma_yticks=list(range(12)), clim=[0, 1], cmap=cmap,
            title=title, ylabel='Chroma', xlabel='Time (seconds)',
            colorbar=True, xlim=xlim,
        )
        plt.tight_layout()


def plot_hmm_likelihood_matrix(
    audio_path: Path | str,
    annotation_path: Path | str,
    color_ann: dict,
    chord_labels: list[str],
    config: ChromagramConfig | None = None,
) -> None:
    """Plot the chromagram (observation sequence) and HMM likelihood matrix
    for a given audio file side by side with annotation overlays."""
    if config is None:
        config = ChromagramConfig(window_size=4096, hop_size=1024, gamma=0.1)

    X, Fs_X, _, _, x_dur = compute_chromagram(audio_path, config)
    N_X = X.shape[1]
    chord_sim, _ = chord_recognition_template(X, norm_sim='1')
    ann_matrix, _, _, _, ann_seg_sec = convert_chord_ann_matrix(
        annotation_path, chord_labels, Fs=Fs_X, N=N_X, last=True,
    )

    fig, ax = plt.subplots(
        3, 2,
        gridspec_kw={'width_ratios': [1, 0.03], 'height_ratios': [1.5, 3, 0.2]},
        figsize=(9, 7),
    )
    libfmp.b.plot_chromagram(
        X, ax=[ax[0, 0], ax[0, 1]], Fs=Fs_X, clim=[0, 1], xlabel='',
        title=f'Observation sequence ({config.version}-based chromagram, {Fs_X:.1f} Hz)',
    )
    libfmp.b.plot_segments_overlay(
        ann_seg_sec, ax=ax[0, 0], time_max=x_dur,
        print_labels=False, colors=color_ann, alpha=0.1,
    )
    libfmp.b.plot_matrix(
        chord_sim, ax=[ax[1, 0], ax[1, 1]], Fs=Fs_X,
        clim=[0, np.max(chord_sim)],
        title='Likelihood matrix (time–chord representation)',
        ylabel='Chord', xlabel='',
    )
    ax[1, 0].set_yticks(np.arange(len(chord_labels)))
    ax[1, 0].set_yticklabels(chord_labels)
    libfmp.b.plot_segments_overlay(
        ann_seg_sec, ax=ax[1, 0], time_max=x_dur,
        print_labels=False, colors=color_ann, alpha=0.1,
    )
    libfmp.b.plot_segments(
        ann_seg_sec, ax=ax[2, 0], time_max=x_dur, time_label='Time (seconds)',
        colors=color_ann, alpha=0.3,
    )
    ax[2, 1].axis('off')
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def plot_matrix_chord_eval(
    I_ref: np.ndarray,
    I_est: np.ndarray,
    Fs: float = 1,
    xlabel: str = 'Time (seconds)',
    ylabel: str = 'Chord',
    title: str = '',
    chord_labels: list[str] | None = None,
    ax: list | None = None,
    grid: bool = True,
    figsize: tuple[float, float] = (9, 3.5),
):
    """Plot TP, FP, and FN items in a colour-coded time–chord grid."""
    fig = None
    if ax is None:
        fig, ax_single = plt.subplots(1, 1, figsize=figsize)
        ax = [ax_single]

    I_TP = np.logical_and(I_ref, I_est).astype(int)
    I_FP = I_est - I_TP
    I_FN = I_ref - I_TP
    I_vis = 3 * I_TP + 2 * I_FN + 1 * I_FP

    eval_cmap = colors.ListedColormap([[1, 1, 1], [1, 0.3, 0.3], [1, 0.7, 0.7], [0, 0, 0]])
    eval_bounds = np.array([0, 1, 2, 3, 4]) - 0.5
    eval_norm = colors.BoundaryNorm(eval_bounds, 4)

    T_coef = np.arange(I_vis.shape[1]) / Fs
    F_coef = np.arange(I_vis.shape[0])
    x_ext = (T_coef[1] - T_coef[0]) / 2
    y_ext = (F_coef[1] - F_coef[0]) / 2
    extent = [T_coef[0] - x_ext, T_coef[-1] + x_ext, F_coef[0] - y_ext, F_coef[-1] + y_ext]

    im = ax[0].imshow(
        I_vis, origin='lower', aspect='auto',
        cmap=eval_cmap, norm=eval_norm, extent=extent, interpolation='nearest',
    )
    if len(ax) == 2:
        cbar = plt.colorbar(im, cax=ax[1], cmap=eval_cmap, norm=eval_norm,
                            boundaries=eval_bounds, ticks=[0, 1, 2, 3])
    else:
        plt.sca(ax[0])
        cbar = plt.colorbar(im, cmap=eval_cmap, norm=eval_norm,
                            boundaries=eval_bounds, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['TN', 'FP', 'FN', 'TP'])

    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)
    if chord_labels is not None:
        ax[0].set_yticks(np.arange(len(chord_labels)))
        ax[0].set_yticklabels(chord_labels)
    if grid:
        ax[0].grid()

    return fig, ax, im


def plot_chord_recognition_result(
    ann_matrix: np.ndarray,
    result: EvaluationResult,
    chord_matrix: np.ndarray,
    chord_labels: list[str],
    title: str = '',
    matrix_label: str = '',
    p: float = 0.15,
    xlim: list[float] | None = None,
    Fs_X: float = 1,
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot the chord recognition evaluation result (TP/FP/FN colour grid)."""
    suffix = f' (TP={result.true_positives}, FP={result.false_positives}, FN={result.false_negatives}, ' \
             f'P={result.precision:.3f}, R={result.recall:.3f}, F={result.f_measure:.3f})'
    if matrix_label:
        suffix += f' [{matrix_label}, p={p:.2f}]'
    full_title = title + suffix

    fig, ax, im = plot_matrix_chord_eval(
        ann_matrix, chord_matrix, Fs=Fs_X, figsize=figsize,
        title=full_title, ylabel='Chord', xlabel='Time (frames)',
        chord_labels=chord_labels,
    )
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Transition matrix plot
# ---------------------------------------------------------------------------

def plot_transition_matrix(
    A: np.ndarray,
    log: bool = True,
    ax: list | None = None,
    figsize: tuple[float, float] = (6, 5),
    title: str = '',
    xlabel: str = 'State',
    ylabel: str = 'State',
    cmap: str = 'gray_r',
    quadrant: bool = False,
) -> tuple:
    """Plot a 24-state chord transition matrix."""
    fig = None
    if ax is None:
        fig, ax_single = plt.subplots(1, 1, figsize=figsize)
        ax = [ax_single]

    if log:
        A_plot = np.log(A)
        cbar_label = 'Log probability'
        clim = [-6, 0]
    else:
        A_plot = A
        cbar_label = 'Probability'
        clim = [0, 1]

    im = ax[0].imshow(A_plot, origin='lower', aspect='equal', cmap=cmap, interpolation='nearest')
    im.set_clim(clim)
    plt.sca(ax[0])
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel(cbar_label)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)

    chord_labels = get_chord_labels()
    squeezed = chord_labels.copy()
    for k in [1, 3, 6, 8, 10, 11, 13, 15, 17, 18, 20, 22]:
        squeezed[k] = ''
    ax[0].set_xticks(np.arange(24))
    ax[0].set_yticks(np.arange(24))
    ax[0].set_xticklabels(squeezed)
    ax[0].set_yticklabels(chord_labels)

    if quadrant:
        ax[0].axvline(x=11.5, ymin=0, ymax=24, linewidth=2, color='r')
        ax[0].axhline(y=11.5, xmin=0, xmax=24, linewidth=2, color='r')

    return fig, ax, im


# ---------------------------------------------------------------------------
# Experiment / statistics plots
# ---------------------------------------------------------------------------

def plot_statistics(
    para_list: list,
    songs: list[Song],
    indices: list[int],
    result_dict: dict[int, np.ndarray],
    ax,
    ylim: list[float] | None = None,
    title: str = '',
    xlabel: str = '',
    ylabel: str = 'F-measure',
    legend: bool = True,
) -> None:
    """Plot per-song F-measure curves over a parameter sweep, plus the mean."""
    from chord_rec.evaluation import compute_mean_f_measure

    for idx, i in enumerate(indices):
        color = LEGEND_COLORS[idx % len(LEGEND_COLORS)]
        ax.plot(
            para_list, result_dict[i],
            color=color, linestyle=':', linewidth=2, label=songs[i].name,
        )
    ax.plot(
        para_list, compute_mean_f_measure(result_dict, indices),
        color='k', linestyle='-', linewidth=2, label='Mean',
    )
    if legend:
        ax.legend(loc='upper right', fontsize=6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlim([para_list[0], para_list[-1]])


def plot_recognition_results(
    songs: list[Song],
    indices: list[int],
    X_dict_STFT: dict[int, np.ndarray],
    ann_dict_STFT: dict[int, tuple],
    X_dict_CQT: dict[int, np.ndarray],
    ann_dict_CQT: dict[int, tuple],
    X_dict_IIR: dict[int, np.ndarray],
    ann_dict_IIR: dict[int, tuple],
    chord_labels: list[str],
    recognizer: ChordRecognizer | None = None,
    matrix_label: str = '',
) -> None:
    """Run recognition and plot evaluation results for all three chromagram
    versions (STFT, CQT, IIR) for each selected song.

    Args:
        recognizer: ChordRecognizer to use. Defaults to a basic uniform recognizer.
        matrix_label: Human-readable label for the transition matrix used in titles.
    """
    if recognizer is None:
        recognizer = ChordRecognizer()

    for i in indices:
        for version, X_dict, ann_dict in [
            ('STFT', X_dict_STFT, ann_dict_STFT),
            ('CQT', X_dict_CQT, ann_dict_CQT),
            ('IIR', X_dict_IIR, ann_dict_IIR),
        ]:
            chord_hmm, _, _ = recognizer.recognize(X_dict[i])
            result = compute_eval_measures(ann_dict[i][0], chord_hmm)
            title = f'{songs[i].name} [{version}; HMM]'
            plot_chord_recognition_result(
                ann_dict[i][0], result, chord_hmm, chord_labels,
                title=title, matrix_label=matrix_label, p=recognizer.config.p,
            )
