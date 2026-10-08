"""The cross encoder scoring stage.

This subpackage deliberately imports nothing. ``prompt.py`` is pure string
handling and runs anywhere; ``model.py`` and ``pair_scores.py`` need torch and
transformers, which are an optional extra. An ``__init__`` that re-exported
them would pull torch into every caller that only wanted to read a prompt
template, and would make the CPU-only parts of the package unusable on a
machine without torch or a GPU.

Reach for the modules by name:

    from cruxes.reranker.prompt import render_reranker_prompt   # no torch
    from cruxes.reranker.pair_scores import score_pair_matrices # needs torch
"""
