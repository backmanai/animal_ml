"""
Animal ML - Audio classification utilities.
"""

from .visualization import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_cv_score_distribution,
    plot_waveforms,
    plot_spectrograms
)

from .data_loading import (
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
