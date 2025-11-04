"""
Source code for animal audio classification project.

This package contains helper modules for data loading and visualization.
"""

# Re-export from subpackage for convenience
from .animal_ml import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_cv_score_distribution,
    plot_waveforms,
    plot_spectrograms,
    load_animal_audio_stats
)

__all__ = [
    'plot_confusion_matrix',
    'plot_model_comparison',
    'plot_cv_score_distribution',
    'plot_waveforms',
    'plot_spectrograms',
    'load_animal_audio_stats'
]

__version__ = '0.1.0'
