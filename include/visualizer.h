#ifndef _VISUALIZER_H_
    #define _VISUALIZER_H_

    #include <stdlib.h>
    #include <stdio.h>
    #include <unistd.h>
    #include <string.h>
    #include <pthread.h>
    #include <time.h>
    #include <stdbool.h>
    #include <pthread.h>
    #include <gif_lib.h>
    #include <sys/ioctl.h>
    #include <webp/demux.h>

    #define RESET "\033[0m"
    #define CHAR ' '
    #define PIXEL_TEMPLATE "\033[48;2;000;000;000m "
    #define DEFAULT_CHAR_RATIO 2.29

typedef enum image_type_e {
    BMP,
    GIF,
    JPEG,
    PNG,
    WEBP,
} ImageType;

typedef struct {
    unsigned char *pixels;
    unsigned char *previous_pixels;
    unsigned char *gpu_pixels;
    ImageType type;
    int height;
    int width;
    int channels;
    int nb_frames;
    int actual_frame;
    int ms_to_wait;
    /* fields for pecific types */
    GifFileType *gif;
    WebPIterator webp;
    WebPDemuxer *demux;
    unsigned char *webp_data;
} Image;

typedef struct {
    char *print_buffer;
    char *gpu_print_buffer;
    int cols;
    int rows;
    int buffer_size;
    float char_ratio;
    bool use_gpu;
    pthread_t gpu_loader;
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
void open_webp(char *filename, Image *settings);
void open_bmp(char *filename, Image *settings);

void get_pixels_from_frame_gif(Image *settings, int frame_to_load);
void get_pixels_from_next_frame_webp(Image *settings);


typedef struct {
    int offset_left;
    int offset_top;
    int width;
    int height;
} FrameZone;

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

// Destroy functions
void destroy_image(Image *image);

void destroy_gif(Image *image);
void destroy_webp(Image *image);

#endif //_VISUALIZER_H_
