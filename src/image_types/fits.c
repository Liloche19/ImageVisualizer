#include "../../include/visualizer.h"
#include "fitsio.h"

void read_black_and_white(fitsfile *fptr, long *naxes, Image *settings)
{
    long fpixels[3] = {1, 1, 1};
    long npixels = 0;
    int anynul = 0;
    int status = 0;
    float min = 0.0;
    float max = 0.0;
    float val = 0.0;
    unsigned char scaled = 0;
    float *gray_color = NULL;

    npixels = naxes[0] * naxes[1];
    settings->width = naxes[0];
    settings->height = naxes[1];
    gray_color = malloc(sizeof(float) * npixels);
    fits_read_pix(fptr, TFLOAT, fpixels, npixels, NULL, gray_color, &anynul, &status);
    settings->pixels = malloc(sizeof(unsigned char) * npixels * 3);
    if (settings->pixels == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }
    min = gray_color[0];
    max = gray_color[0];
    for (int i = 1; i < npixels; i++){
        if (gray_color[i] < min)
            min = gray_color[i];
        if (gray_color[i] > max)
            max = gray_color[i];
    }
    for(long i = 0; i < npixels; i++) {
        val = (gray_color[i] - min) / (max - min);
        if (val < 0)
            val = 0;
        if (val > 1)
            val = 1;
        scaled = val * 255;
        settings->pixels[3 * i] = scaled;
        settings->pixels[3 * i + 1] = scaled;
        settings->pixels[3 * i + 2] = scaled;
    }
    free(gray_color);
    return;
}

void read_rgb(fitsfile *fptr, long *naxes, Image *settings)
{
    long fpixels[3] = {1, 1, 1};
    long npixels = 0;
    int anynul = 0;
    int status = 0;
    unsigned char *red = NULL;
    unsigned char *green = NULL;
    unsigned char *blue = NULL;

    npixels = naxes[0] * naxes[1];
    settings->width = naxes[0];
    settings->height = naxes[1];
    red = malloc(sizeof(unsigned char) * npixels);
    fpixels[2] = 1;
    fits_read_pix(fptr, TBYTE, fpixels, npixels, NULL, red, &anynul, &status);
    green = malloc(sizeof(unsigned char) * npixels);
    fpixels[2] = 2;
    fits_read_pix(fptr, TBYTE, fpixels, npixels, NULL, green, &anynul, &status);
    blue = malloc(sizeof(unsigned char) * npixels);
    fpixels[2] = 3;
    fits_read_pix(fptr, TBYTE, fpixels, npixels, NULL, blue, &anynul, &status);
    settings->pixels = malloc(sizeof(unsigned char) * npixels * 3);
    if (settings->pixels == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }
    for(long i = 0; i < npixels; i++){
        settings->pixels[3 * i] = red[i];
        settings->pixels[3 * i + 1] = green[i];
        settings->pixels[3 * i + 2] = blue[i];
    }
    free(red);
    free(green);
    free(blue);
    return;
}

void open_fits(char *filename, Image *settings)
{
    int bitpix = 0;
    int naxis = 0;
    int status = 0;
    long naxes[3] = {1, 1, 1};
    fitsfile *fptr = NULL;

    if(fits_open_file(&fptr, filename, READONLY, &status)) {
        fprintf(stderr, "Error while opening image!\n");
        exit(1);
    }
    fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
    if (naxis == 2) {
        if (bitpix != -32) {
            fprintf(stderr, "exotic fits format!\n");
            exit(1);
        }
        read_black_and_white(fptr, naxes, settings);
    } else if (naxis == 3) {
        read_rgb(fptr, naxes, settings);
    } else {
        fprintf(stderr, "exotic fits format!\n");
        exit(1);
    }
    fits_close_file(fptr, &status);
    settings->type = FITS;
    settings->actual_frame = 0;
    settings->nb_frames = 1;
    settings->channels = 3;
    return;
}
