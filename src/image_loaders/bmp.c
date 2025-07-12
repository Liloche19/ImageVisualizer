#include "../../include/visualizer.h"
#include "bmp.h"

void open_bmp(char *filename, Image *settings)
{
    FILE *file = fopen(filename, "rb");
    BmpFileHeader fileHeader;
    BmpInfoHeader infoHeader;
    int row_padded = 0;
    int src_idx = 0;
    int dst_idx = 0;
    unsigned char *row = NULL;

    if (file == NULL) {
        fprintf(stderr, "Error opening image file!\n");
        exit(1);
    }
    fread(&fileHeader, sizeof(fileHeader), 1, file);
    if (fileHeader.bfType != 0x4D42) {
        fprintf(stderr, "Not a valid BMP file!\n");
        fclose(file);
        exit(1);
    }
    fread(&infoHeader, sizeof(infoHeader), 1, file);
    if (infoHeader.biBitCount != 24 || infoHeader.biCompression != 0) {
        settings->channels = 3;
        fprintf(stderr, "This BMP file is not yet supported!\n");
        fclose(file);
        exit(1);
    }
    settings->width = infoHeader.biWidth;
    settings->height = infoHeader.biHeight;
    row_padded = ((settings->width * 3 + 3) & ~3);
    settings->pixels = malloc(3 * (settings->width) * (settings->height));
    if (settings->pixels == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        fclose(file);
        exit(1);
    }
    fseek(file, fileHeader.bfOffBits, SEEK_SET);
    row = malloc(sizeof(unsigned char) * row_padded);
    if (row == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        free(settings->pixels);
        fclose(file);
        exit(1);
    }
    for (int y = settings->height - 1; y >= 0; y--) {
        fread(row, sizeof(unsigned char), row_padded, file);
        for (int x = 0; x < settings->width; x++) {
            src_idx = x * 3;
            dst_idx = (y * (settings->width) + x) * 3;
            settings->pixels[dst_idx] = row[src_idx + 2];
            settings->pixels[dst_idx + 1] = row[src_idx + 1];
            settings->pixels[dst_idx + 2] = row[src_idx];
        }
    }
    free(row);
    fclose(file);
    settings->actual_frame = 0;
    settings->nb_frames = 1;
    return;
}
