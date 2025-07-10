#ifndef _VISUALIZER_H_
    #define _VISUALIZER_H_

    #include <stdlib.h>
    #include <stdio.h>
    #include <unistd.h>
    #include <string.h>
    #include <pthread.h>
    #include <time.h>
    #include <stdbool.h>
    #include <sys/ioctl.h>

    #define RESET "\033[0m"
    #define CHAR ' '
    #define DEFAULT_CHAR_RATIO 2.29

typedef struct {
    unsigned char *pixels;
    unsigned char *gpu_pixels;
    char *filename;
    int height;
    int width;
    int channels;
} Image;

typedef struct {
    int cols;
    int rows;
    char *print_buffer;
    int buffer_size;
    float char_ratio;
    char *gpu_print_buffer;
} Screen;

typedef union rgb_u {
    unsigned char rgb[3];
} rgb_t;

// Display functions
void apply_color_at_coord_on_buffer(Screen *screen, int x, int y, rgb_t color);
void display_image(Image *image, Screen *screen);

// Image functions
rgb_t avg_rgb(unsigned char *img, float ratio_x, float ratio_y, int x, int y, int width, int height, int channels);
char *resize_image(Image *image, Screen *screen);
void load_image(char *filename, Image *settings);

// Image loaders
void open_jpeg(char *filename, Image *settings);
void open_png(char *filename, Image *settings);
void open_gif(char *filename, Image *settings);

// Terminal functions
void get_screen_informations(Screen *settings);

// Threading functions
long get_nb_threads(void);

typedef struct {
    Image *image;
    Screen *screen;
    int start_index;
    int end_index;
    float ratio_x;
    float ratio_y;
} ThreadData;

#endif //_VISUALIZER_H_
