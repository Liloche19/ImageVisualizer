#include "../include/visualizer.h"

void print_pixel(unsigned char r, unsigned char g, unsigned char b)
{
    char pixel[] = "\033[48;2;000;000;000m ";

    pixel[7] = (r / 100) + 48;
    pixel[8] = ((r / 10) % 10) + 48;
    pixel[9] = (r % 10) + 48;
    pixel[11] = (g / 100) + 48;
    pixel[12] = ((g / 10) % 10) + 48;
    pixel[13] = (g % 10) + 48;
    pixel[15] = (b / 100) + 48;
    pixel[16] = ((b / 10) % 10) + 48;
    pixel[17] = (b % 10) + 48;
    pixel[19] = CHAR;
    write(1, pixel, strlen(pixel));
    return;
}

void display_image(Image *image, Screen *screen)
{
    int index = 0;

    screen->resized = resize_image(image, screen);
    for (int i = 0; i < screen->rows; i++) {
        for (int j = 0; j < screen->cols; j++) {
            index = (i * screen->cols + j) * 3;
            print_pixel(screen->resized[index + 0], screen->resized[index + 1], screen->resized[index + 2]);
        }
        write(1, RESET, strlen(RESET));
        write(1, "\n", 1);
    }
    free(screen->resized);
    return;
}
