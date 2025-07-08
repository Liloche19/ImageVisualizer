#ifndef _VISUALIZER_H_
    #define _VISUALIZER_H_

    #include <stdlib.h>
    #include <stdio.h>
    #include <unistd.h>
    #include <string.h>
    #include <sys/ioctl.h>

    #define RESET "\033[0m"
    #define CHAR ' '
    #define DEFAULT_CHAR_RATIO 2.29

typedef struct {
    unsigned char *pixels;
    int height;
    int width;
    int channels;
} Image;

typedef struct {
    int cols;
    int rows;
    float char_ratio;
} Screen;

#endif //_VISUALIZER_H_
