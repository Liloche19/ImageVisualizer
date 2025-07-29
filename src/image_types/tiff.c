#include "../../include/visualizer.h"
#include <tiff.h>
#include <tiffio.h>

void open_tiff(char *filename, Image *settings)
{
    TIFF* tif = TIFFOpen(filename, "r");
    unsigned char *buf = NULL;
    tsize_t scanline_size;
    int bits = 0;
    short channels = 0;
    short photometric = 0;

    if (tif == NULL) {
        fprintf(stderr, "Error while opening tiff\n");
        exit(1);
    }
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &settings->width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &settings->height);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);
    if (TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &channels) != 1) {
        channels = 1;
    }
    settings->channels = (int) channels;
    scanline_size = TIFFScanlineSize(tif);
    buf = malloc((settings->height) * scanline_size);
    if (buf == NULL) {
        TIFFClose(tif);
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }
    for (int row = 0; row < settings->height; row++) {
        if (TIFFReadScanline(tif, buf + row * scanline_size, row, 0) < 0) {
            fprintf(stderr, "Error while reading tiff file!\n");
            free(buf);
            TIFFClose(tif);
            exit(1);
        }
    }
    if (settings->channels == 2) {
        settings->channels = 3;
        settings->pixels = malloc(sizeof(unsigned char) * settings->height * settings->width * 3);
        if (settings->pixels == NULL) {
            fprintf(stderr, "Malloc failed!\n");
            free(buf);
            exit(1);
        }
        for (int y = 0; y < settings->height; y++) {
            unsigned char *src = buf + y * scanline_size;
            for (int x = 0; x < settings->width; x++) {
                int in_idx = x * 2;
                int out_idx = (y * settings->width + x) * 3;
                unsigned char gray = src[in_idx];
                settings->pixels[out_idx    ] = gray;
                settings->pixels[out_idx + 1] = gray;
                settings->pixels[out_idx + 2] = gray;
            }
        }
        free(buf);
    } else if (settings->channels == 1) {
        settings->channels = 3;
        settings->pixels = malloc(sizeof(unsigned char) * settings->height * settings->width * settings->channels);
        if (settings->pixels == NULL) {
            fprintf(stderr, "Malloc failed!\n");
            exit(1);
        }
        for (int y = 0; y < settings->height; y++) {
            unsigned char *src = buf + y * scanline_size;
            for (int x = 0; x < settings->width; x++) {
                int in_idx = x * 4;
                int out_idx = (y * settings->width + x) * 3;
                settings->pixels[out_idx    ] = src[in_idx    ];
                settings->pixels[out_idx + 1] = src[in_idx + 1];
                settings->pixels[out_idx + 2] = src[in_idx + 2];
            }
        }
        free(buf);
    } else if (bits == 32 && settings->channels == 3) {
        settings->pixels = malloc(sizeof(unsigned char) * settings->height * settings->width * 3);
        if (settings->pixels == NULL) {
            fprintf(stderr, "Malloc failed!\n");
            free(buf);
            exit(1);
        }
        for (int y = 0; y < settings->height; y++) {
            float *src = (float *)(buf + y * scanline_size);
            for (int x = 0; x < settings->width; x++) {
                int in_idx = x * 3;
                int out_idx = (y * settings->width + x) * 3;
                float r = src[in_idx];
                float g = src[in_idx + 1];
                float b = src[in_idx + 2];
                settings->pixels[out_idx    ] = (unsigned char)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f);
                settings->pixels[out_idx + 1] = (unsigned char)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f);
                settings->pixels[out_idx + 2] = (unsigned char)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f);
            }
        }
        free(buf);
    } else {
        settings->pixels = buf;
    }
    TIFFClose(tif);
    settings->nb_frames = 1;
    settings->actual_frame = 0;
    settings->ms_to_wait = 0;
    settings->type = TIF;
    return;
}
