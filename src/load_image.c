#include "../include/visualizer.h"

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
    if (strcmp(extension, "jpg") == 0 || strcmp(extension, "JPG") == 0 || strcmp(extension, "jpeg") == 0 || strcmp(extension, "JPEG") == 0)
        return open_jpeg(filename, settings);
    if (strcmp(extension, "gif") == 0 || strcmp(extension, "GIF") == 0)
        return open_gif(filename, settings);
    if (strcmp(extension, "webp") == 0 || strcmp(extension, "webp") == 0)
        return open_webp(filename, settings);
    fprintf(stderr, "Uknown format detected!\n");
    exit(1);
}
