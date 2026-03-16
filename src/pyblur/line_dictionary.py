class LineDictionary:
    """Lookup table mapping kernel size and angle to line endpoint coordinates.

    The ``lines`` attribute has the shape
    ``{kernel_dim: {angle_degrees: [r0, c0, r1, c1]}}``.
    """

    lines: dict[int, dict[float, list[int]]]

    def __init__(self) -> None:
        self.lines = {}
        self._create_3x3_lines()
        self._create_5x5_lines()
        self._create_7x7_lines()
        self._create_9x9_lines()

    def _create_3x3_lines(self) -> None:
        lines: dict[float, list[int]] = {}
        lines[0] = [1, 0, 1, 2]
        lines[45] = [2, 0, 0, 2]
        lines[90] = [0, 1, 2, 1]
        lines[135] = [0, 0, 2, 2]
        self.lines[3] = lines

    def _create_5x5_lines(self) -> None:
        lines: dict[float, list[int]] = {}
        lines[0] = [2, 0, 2, 4]
        lines[22.5] = [3, 0, 1, 4]
        lines[45] = [0, 4, 4, 0]
        lines[67.5] = [0, 3, 4, 1]
        lines[90] = [0, 2, 4, 2]
        lines[112.5] = [0, 1, 4, 3]
        lines[135] = [0, 0, 4, 4]
        lines[157.5] = [1, 0, 3, 4]
        self.lines[5] = lines

    def _create_7x7_lines(self) -> None:
        lines: dict[float, list[int]] = {}
        lines[0] = [3, 0, 3, 6]
        lines[15] = [4, 0, 2, 6]
        lines[30] = [5, 0, 1, 6]
        lines[45] = [6, 0, 0, 6]
        lines[60] = [6, 1, 0, 5]
        lines[75] = [6, 2, 0, 4]
        lines[90] = [0, 3, 6, 3]
        lines[105] = [0, 2, 6, 4]
        lines[120] = [0, 1, 6, 5]
        lines[135] = [0, 0, 6, 6]
        lines[150] = [1, 0, 5, 6]
        lines[165] = [2, 0, 4, 6]
        self.lines[7] = lines

    def _create_9x9_lines(self) -> None:
        lines: dict[float, list[int]] = {}
        lines[0] = [4, 0, 4, 8]
        lines[11.25] = [5, 0, 3, 8]
        lines[22.5] = [6, 0, 2, 8]
        lines[33.75] = [7, 0, 1, 8]
        lines[45] = [8, 0, 0, 8]
        lines[56.25] = [8, 1, 0, 7]
        lines[67.5] = [8, 2, 0, 6]
        lines[78.75] = [8, 3, 0, 5]
        lines[90] = [8, 4, 0, 4]
        lines[101.25] = [0, 3, 8, 5]
        lines[112.5] = [0, 2, 8, 6]
        lines[123.75] = [0, 1, 8, 7]
        lines[135] = [0, 0, 8, 8]
        lines[146.25] = [1, 0, 7, 8]
        lines[157.5] = [2, 0, 6, 8]
        lines[168.75] = [3, 0, 5, 8]
        self.lines[9] = lines
