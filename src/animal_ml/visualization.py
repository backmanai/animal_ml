"""
Visualization helper functions for audio classification project.

This module provides reusable visualization functions to keep notebooks clean
and focused on analysis rather than plotting code.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', cmap='Blues', figsize=(8, 6)):
    """
    Plot a confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels (e.g., ['birds', 'cats', 'dogs'])
        title: Plot title
        cmap: Colormap name
        figsize: Figure size tuple

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=cmap, values_format='d')
    plt.title(title)
    plt.tight_layout()

    return fig, ax


def plot_model_comparison(model_names, default_scores, tuned_scores, figsize=(10, 6)):
    """
    Create a bar chart comparing default vs tuned model performance.

    Args:
        model_names: List of model names
        default_scores: List of default model accuracies
        tuned_scores: List of tuned model accuracies
        figsize: Figure size tuple

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, default_scores, width, label='Default',
                   color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, tuned_scores, width, label='Tuned',
                   color='lightgreen', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Model Performance: Default vs Tuned Hyperparameters',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim([0.85, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    return fig, ax


def plot_cv_score_distribution(default_scores, tuned_scores,
                                default_test_acc, tuned_test_acc,
                                model_name='SVM', default_params='C=1.0',
                                tuned_params='C=100', figsize=(10, 6)):
    """
    Plot boxplot showing distribution of cross-validation scores.

    Args:
        default_scores: Array of CV scores for default model
        tuned_scores: Array of CV scores for tuned model
        default_test_acc: Single test set accuracy for default model
        tuned_test_acc: Single test set accuracy for tuned model
        model_name: Name of the model
        default_params: String describing default parameters
        tuned_params: String describing tuned parameters
        figsize: Figure size tuple

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    positions = [1, 2]
    data_to_plot = [default_scores, tuned_scores]

    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True, meanline=True)

    # Color the boxes
    colors = ['skyblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add our single test set result as red dots
    ax.plot(1, default_test_acc, 'ro', markersize=10,
            label='Our test set (random_state=42)')
    ax.plot(2, tuned_test_acc, 'ro', markersize=10)

    ax.set_xticklabels([f'Default {model_name}\n({default_params})',
                        f'Tuned {model_name}\n({tuned_params})'])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{model_name} Performance Distribution Across Multiple Random Splits',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    return fig, ax


def plot_waveforms(audio_data_list, labels, sr=22050, figsize=(12, 8)):
    """
    Plot waveforms for multiple audio samples.

    Args:
        audio_data_list: List of audio arrays
        labels: List of labels for each audio sample
        sr: Sample rate
        figsize: Figure size tuple

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    import librosa.display

    n_samples = len(audio_data_list)
    fig, axes = plt.subplots(n_samples, 1, figsize=figsize)

    if n_samples == 1:
        axes = [axes]

    for idx, (audio, label) in enumerate(zip(audio_data_list, labels)):
        librosa.display.waveshow(audio, sr=sr, ax=axes[idx])
        axes[idx].set_title(f'{label} Waveform')

    plt.tight_layout()

    return fig, axes


def plot_spectrograms(audio_data_list, labels, sr=22050, figsize=(12, 8)):
    """
    Plot spectrograms for multiple audio samples.

    Args:
        audio_data_list: List of audio arrays
        labels: List of labels for each audio sample
        sr: Sample rate
        figsize: Figure size tuple

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    import librosa
    import librosa.display

    n_samples = len(audio_data_list)
    fig, axes = plt.subplots(n_samples, 1, figsize=figsize)

    if n_samples == 1:
        axes = [axes]

    for idx, (audio, label) in enumerate(zip(audio_data_list, labels)):
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[idx])
        axes[idx].set_title(f'{label} Spectrogram')

    plt.tight_layout()

    return fig, axes
