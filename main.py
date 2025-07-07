#! /usr/bin/env python3

import os
import sys
import numpy
from PIL import Image
from posix import terminal_size

RESET: str = "\033[0m"
CHAR: str = " "
CHAR_RATIO = 2.25 / 1

def get_avg_rgb(original_array, coord: tuple[int, int], ratio: tuple[float, float]) -> tuple[int, int, int]:
    red = 0
    green = 0
    blue = 0
    nb_pixels = 0
    for x_ratio in range(int(ratio[1])):
        for y_ratio in range(int(ratio[0])):
            nb_pixels += 1
            red += int(original_array[coord[1] * int(ratio[1]) + x_ratio, coord[0] * int(ratio[0]) + y_ratio][0])
            green += int(original_array[coord[1] * int(ratio[1]) + x_ratio, coord[0] * int(ratio[0]) + y_ratio][1])
            blue += int(original_array[coord[1] * int(ratio[1]) + x_ratio, coord[0] * int(ratio[0]) + y_ratio][2])
    if nb_pixels == 0:
        return (0, 0, 0)
    return (red // nb_pixels, green // nb_pixels, blue // nb_pixels)

def resize_image(img, screen_dimensions: tuple[int, int]):
    np_array = numpy.array(img)
    original_size = img.size
    img_ratio = original_size[0] / original_size[1]
    if img_ratio > (screen_dimensions[0] / screen_dimensions[1]) * CHAR_RATIO:
        print("First")
        screen_dimensions = (screen_dimensions[0], int(screen_dimensions[0] / (img_ratio / CHAR_RATIO)))
    else:
        print("Second")
        screen_dimensions = (int(screen_dimensions[1] * img_ratio * CHAR_RATIO), screen_dimensions[1])
    rgb_array = numpy.zeros((screen_dimensions[1], screen_dimensions[0], 3), dtype=numpy.uint8)
    print(screen_dimensions)
    ratio = (original_size[0] / screen_dimensions[0], original_size[1] / screen_dimensions[1])
    for x in range(screen_dimensions[0]):
        for y in range(screen_dimensions[1]):
            rgb_array[y, x] = get_avg_rgb(np_array, (x, y), ratio)
    resized = Image.fromarray(rgb_array)
    return resized

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
        if img.mode not in ('RGBA', 'RGB'):
            img = img.convert('RGB')
        if self.columns == 0 or self.lines == 0:
            return
        resized = resize_image(img, self.get_dimensions())
        pixel_array = numpy.array(resized)
        for x in range(self.lines):
            for y in range(self.columns):
                print(f"\033[48;2;{pixel_array[x, y][0]};{pixel_array[x, y][1]};{pixel_array[x, y][2]}m{CHAR}{RESET}", end="")
            if x != self.lines - 1:
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
