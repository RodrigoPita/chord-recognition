from typing import Final

CHORDAL_TYPES: Final[dict[str, str]] = {
    'Z': 'M7',
    'Y': '7',
    'X': '(b5)7',
    'W': '(#5)7',
    'V': '',
    'z': 'm7',
    'y': 'ø',
    'x': '°7',
    'w': 'm(M7)',
    'v': 'm',
}

FUNDAMENTALS: Final[list[str]] = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
]

TEMPLATES_IN_C: Final[dict[str, list[int]]] = {
    'M7':    [0, 4, 7, 11],
    '7':     [0, 4, 7, 10],
    '(b5)7': [0, 4, 6, 10],
    '(#5)7': [0, 4, 8, 10],
    '':      [0, 4, 7],
    'm7':    [0, 3, 7, 10],
    'ø':     [0, 3, 6, 10],
    '°7':    [0, 3, 6, 9],
    'm(M7)': [0, 3, 7, 11],
    'm':     [0, 3, 7],
}

LEGEND_COLORS: Final[list[str]] = ['r', 'b', 'c', 'g', 'm', 'y', 'k']
