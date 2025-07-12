#include <gif_lib.h>
#ifdef USE_CUDA
#include "../include/visualizer_cuda.cuh"
#else
#include "../include/visualizer.h"
#endif

int help(char *prog, int status)
{
    printf("USAGE:\n");
    printf("\t%s [-h] filemane\n\n", prog);
    printf("DESCRIPTION:\n");
    printf("\tfilename\tThe filepath to the image you want to display on your terminal\n");
    printf("\t-h\t\tPrints this help\n");
    return status;
}

int main(int argc, char **argv)
{
    char *filename = NULL;
    Screen screen;
    Image image;
    int err = 0;

    if (argc != 2)
        return help(argv[0], 1);
    if (strcmp(argv[1], "-h") == 0)
        return help(argv[0], 0);
    filename = argv[1];
    if (filename == NULL)
        return 0;
    #ifdef USE_CUDA
        init_gpu(&screen.gpu_loader);
    #endif
    get_screen_informations(&screen);
    load_image(filename, &image);
    display_image(&image, &screen);
    if (image.gif != NULL)
        DGifCloseFile(image.gif, &err);
    free(image.pixels);
    return 0;
}
