"""
The paper's figures: one style, saved numbers, dedicated drawing code.

    style.py        the one figure style (font, sizes, colours, the rules)
    store.py        the numbers behind each figure, saved and read back
    inputs.py       where the pipeline's outputs are and how they are read
    fig_*.py        the figures, one module per section of the paper
    registry.py     the list of all paper figures

Every figure is made in two stages:

    compute   reads what the pipeline wrote (no training, no inference) and
              saves the numbers that are drawn      -> output/paper/data/<name>.npz
    draw      reads ONLY that file and draws        -> output/paper/figures/<dir>/<name>.pdf

so a figure can be redrawn in a second after a change of style or of a label.
The program is scripts/figures.py.
"""
