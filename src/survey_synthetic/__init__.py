"""Synthetic ESS/WVS-like survey microdata for pipeline testing."""

from survey_synthetic.generator import generate_survey_dataframe
from survey_synthetic.schema import DEFAULT_N_COLS, DEFAULT_N_ROWS

__all__ = ["generate_survey_dataframe", "DEFAULT_N_COLS", "DEFAULT_N_ROWS"]
