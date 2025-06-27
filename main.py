#! /usr/bin/env python3

import os
import sys
import numpy
from PIL import Image
from posix import terminal_size

RESET: str = "\033[0m"
CHAR: str = " "

class Screen():
    def __init__(self):
        self.refresh_dimensions()
        return

    def __str__(self) -> str:
        return f"Screen size (columns={self.columns}, lines={self.lines})"

    def refresh_dimensions(self) -> None:
        size: terminal_size = os.get_terminal_size()
        self.columns: int = size.columns
        self.lines: int = size.lines
        return

    def get_dimensions(self) -> tuple[int, int]:
        return (self.columns, self.lines)

    def display_image(self, img) -> None:
        self.refresh_dimensions()
        resized = img.resize(self.get_dimensions())
        if resized.mode not in ('RGBA', 'RGB'):
            resized = resized.convert('RGB')
        pixel_array = numpy.array(resized)
        for y in range(self.lines):
            for x in range(self.columns):
                print(f"\033[48;2;{pixel_array[y, x][0]};{pixel_array[y, x][1]};{pixel_array[y, x][2]}m{CHAR}{RESET}", end="")
            if y != self.lines - 1:
                print()
        return

def print_usage(prog: str) -> None:
    print("USAGE:")
    print(f"\t{prog} [-h] filename")
    print()
    print("DESCRIPTION:")
    print("\tfilename\tThe filepath to the image you want to display on your terminal")
    print("\t-h\t\tPrints this help")
    return

def main() -> int:
    argv: list[str] = sys.argv
    argc: int = len(argv)
    if argc != 2:
        sys.stderr.write("Invalid number of arguments!\n\n")
        sys.stderr.flush()
        print_usage(argv[0])
        return 1
    if argv[1] == "-h":
        print_usage(argv[0])
        return 0
    screen: Screen = Screen()
    img = Image.open(argv[1])
    screen.display_image(img)
    return 0

if __name__ == "__main__":
    sys.exit(main())
