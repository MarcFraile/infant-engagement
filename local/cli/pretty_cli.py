from typing import Any, List, Optional

class PrettyCli:
    """
    Ordered pretty printing in the terminal.

    Formalizes my style choices for outputting data in the terminal into a cleaner system.
    """

    def __init__(self):
        self.previous_line_blank = True # Used to decide if whitespace should be added above.
        self.indent = " " * 4

    def blank(self) -> None:
        """
        Add a blank line, IF the previous line was not blank as well.
        """
        if not self.previous_line_blank:
            print()
            self.previous_line_blank = True

    def print(self, obj: Any, end: Optional[str] = None) -> None:
        """
        Base block for CLI pretty-printing.

        * Manages state for blank lines.
        * For dicts: calls self.print_dict()
        * For others: casts to str and strips trailing whitespace.
        * end keyword is NOT respected for dicts. Otherwise works like print().
        """
        if type(obj) is dict:
            self._print_dict(obj)
            self.blank()
        else:
            text = str(obj)
            print(text.rstrip(), end=end)
        self.previous_line_blank = False

    def main_title(self, text: str) -> None:
        """
        Use this to make a title at the beginning of the run.

        * Strips whitespace and casts to uppercase.
        * Encases in a big box.
        """
        side_padding : int =  24
        lines : List[str] = [line.strip().upper() for line in text.strip().split("\n")]

        max_len : int = 0
        for line in lines:
            max_len = max(max_len, len(line))

        # main_line = f"{side_padding} {text.strip().upper()} {side_padding}"
        # secondary_line = "=" * len(main_line) # len() doesn't seem to behave well with Unicode (tried  你好！)
        cap_line = "=" * (max_len + 2 * (side_padding + 1))

        self.blank()
        self.print(cap_line)
        # self.print(main_line)
        for line in lines:
            overflow = len(cap_line) - len(line) - 2
            left_pad = "=" * (overflow // 2)
            right_pad = "=" * (overflow - len(left_pad))
            self.print(f"{left_pad} {line} {right_pad}")
        self.print(cap_line)
        self.blank()

    def chapter(self, text: str) -> None:
        """
        Use this to separate major parts in the script.

        * Strips whitespace and capitalizes.
        * Adds = to the sides.
        """
        capitalized = " ".join([word.capitalize() for word in text.split()])
        side_padding = "=" * 16
        line = f"{side_padding} {capitalized} {side_padding}"

        self.blank()
        self.print(line)
        self.blank()

    def subchapter(self, text: str) -> None:
        """
        Use this if you need a division bigger than a section but smaller than a chapter.

        * Strips whitespace and capitalizes.
        * Adds - to the sides.
        """
        capitalized = " ".join([word.capitalize() for word in text.split()])
        side_padding = "-" * 8
        line = f"{side_padding} {capitalized} {side_padding}"

        self.blank()
        self.print(line)
        self.blank()

    def section(self, text: str) -> None:
        """
        Use this to separate minor parts in the script.

        * Strips whitespace and capitalizes.
        * Encases in [].
        """
        capitalized = " ".join([word.capitalize() for word in text.split()])

        self.blank()
        self.print(f"[{capitalized}]")
        self.blank()

    def big_divisor(self) -> None:
        """
        Adds a horizontal line (32 '=') surrounded by blanks.
        """
        self.blank()
        self.print("=" * 32)
        self.blank()

    def small_divisor(self) -> None:
        """
        Adds a horizontal line (16 '-') surrounded by blanks.
        """
        self.blank()
        self.print("-" * 16)
        self.blank()

    def _size_dict(self, d: dict, depth: int) -> int:
        """
        Used internally to align all values when pretty-printing a dict.
        """
        prefix = self.indent * depth

        max_len = 0
        for (key, value) in d.items():
            key_len = len(str(key)) + len(prefix) + 2
            max_len = max(max_len, key_len)

            if type(value) is dict:
                child_len = self._size_dict(value, depth + 1)
                max_len = max(max_len, child_len)

        return max_len

    def _print_dict(self, d: dict, depth: int = 0, max_len: Optional[int] = None) -> None:
        """
        Used internally to pretty-print dicts.

        * Prints one "Key: Value" pair per line.
        * Prints sub-dicts hierarchically, with indenting.
        * Pre-calculates the space taken by all printed keys (including sub-dicts) and left-aligns all values to the same column.
        """
        prefix = self.indent * depth
        if max_len is None:
            max_len = self._size_dict(d, depth)

        for (key, value) in d.items():
            if type(value) is dict:
                self.print(f"{prefix}{key}:")
                self._print_dict(value, depth + 1, max_len)
            else:
                self.print(f"{prefix}{key}:".ljust(max_len) + f"{value}")
