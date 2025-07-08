#include "../include/visualizer.h"
#include <png.h>

void open_png(char *filename, Image *settings)
{
    FILE *fp = NULL;
    png_structp png;
    png_infop info;
    int color_type;
    int bit_depth;
    png_bytep *rows;

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file!\n");
        exit(1);
    }
    png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info = png_create_info_struct(png);
    if (png == NULL || info == NULL) {
        fprintf(stderr, "Error while initialising PNG image!\n");
        exit(1);
    }
    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error while reading PNG image!\n");
        exit(1);
    }
    png_init_io(png, fp);
    png_read_info(png, info);
    settings->width  = png_get_image_width(png, info);
    settings->height = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth  = png_get_bit_depth(png, info);
    if (bit_depth == 16)
        png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    png_read_update_info(png, info);
    settings->channels = 4;
    settings->pixels = malloc(settings->width * settings->height * settings->channels);
    if (settings->pixels == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }
    rows = malloc(sizeof(png_bytep) * settings->height);
    for (int y = 0; y < settings->height; y++)
        rows[y] = settings->pixels + y * settings->width * settings->channels;
    png_read_image(png, rows);
    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
    free(rows);
    return;
}

char *get_extension(char *filename)
{
    int dot = 0;

    for (int i = 0; filename[i] != '\0'; i++)
        if (filename[i] == '.')
            dot = i;
    return filename + dot + 1;
}

void load_image(char *filename, Image *settings)
{
    char *extension = get_extension(filename);

    if (strcmp(extension, "png") == 0 || strcmp(extension, "PNG") == 0)
        return open_png(filename, settings);
    fprintf(stderr, "Uknown format detected!\n");
    exit(1);
}
