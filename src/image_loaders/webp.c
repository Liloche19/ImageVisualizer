#include "../../include/visualizer.h"
#include <webp/decode.h>

void open_webp(char *filename, Image *settings)
{
    FILE *file = fopen(filename, "rb");
    size_t file_size = 0;
    uint8_t *webp_data = NULL;

    if (file == NULL) {
        fprintf(stderr, "Error opening image file!\n");
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    rewind(file);
    webp_data = malloc(file_size);
    if (!webp_data) {
        fprintf(stderr, "Malloc failed!\n");
        fclose(file);
        exit(1);
    }
    if (fread(webp_data, 1, file_size, file) != file_size) {
        fprintf(stderr, "Erreur reading image file!\n");
        free(webp_data);
        fclose(file);
        exit(1);
    }
    fclose(file);
    settings->channels = 4;
    settings->pixels = WebPDecodeRGBA(webp_data, file_size, &settings->width, &settings->height);
    free(webp_data);
    if (settings->pixels == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }
    settings->actual_frame = 0;
    settings->nb_frames = 1;
    return;
}
