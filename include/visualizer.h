#ifndef _VISUALIZER_H_
    #define _VISUALIZER_H_

    #include <stdlib.h>
    #include <stdio.h>
    #include <unistd.h>
    #include <string.h>
    #include <pthread.h>
    #include <time.h>
    #include <sys/ioctl.h>

    #include "stb_image.h"

    #define RESET "\033[0m"
    #define CHAR ' '
    #define DEFAULT_CHAR_RATIO 2.29

typedef struct {
    unsigned char *pixels;
    char *filename;
    int height;
    int width;
    int channels;
} Image;

typedef struct {
    int cols;
    int rows;
    unsigned char *resized;
    float char_ratio;
} Screen;

typedef union rgb_u {
    char rgb[3];
} rgb_t;

// Display functions
void display_image(Image *image, Screen *screen);

// Image functions
unsigned char *resize_image(Image *image, Screen *screen);
void load_image(char *filename, Image *settings);

// Terminal functions
void get_screen_informations(Screen *settings);

// Threading functions
long get_nb_threads(void);

typedef struct {
    Image *image;
    Screen * screen;
    int start_index;
    int end_index;
    float ratio_x;
    float ratio_y;
} ThreadData;

#endif //_VISUALIZER_H_
