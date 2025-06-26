#! /usr/bin/env python3

import os
import sys
import numpy
from PIL import Image
from posix import terminal_size

RESET: str = "\033[0m"
CHAR: str = " "

def resize_image(img, dimensions):
    new_image = img.resize(dimensions)
    return new_image

def display_image(img, dimensions: tuple):
    rgb_img = img.convert('RGB')
    pixel_array = numpy.array(rgb_img)
    for y in range(dimensions[1]):
        for x in range(dimensions[0]):
            print(f"\033[48;2;{pixel_array[y, x][0]};{pixel_array[y, x][1]};{pixel_array[y, x][2]}m{CHAR}{RESET}", end="")
        if y != dimensions[1] - 1:
            print()
    return

def main() -> int:
    argv = sys.argv
    argc = len(argv)
    assert argc == 2, "Invalid number of arguments!"
    img = Image.open(argv[1])
    size: terminal_size = os.get_terminal_size()
    dimensions = (size.columns, size.lines)
    new_image = resize_image(img, dimensions)
    display_image(new_image, dimensions)
    return 0

if __name__ == "__main__":
    sys.exit(main())
