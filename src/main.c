#include "../include/visualizer.h"

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
    float char_ratio = CHAR_RATIO;
    char *image = NULL;
    struct winsize w;

    if (argc != 2)
        return help(argv[0], 1);
    if (strcmp(argv[1], "-h") == 0)
        return help(argv[0], 0);
    image = argv[1];
    if (image == NULL || char_ratio == 0)
        return 0;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    char_ratio = (float) w.ws_ypixel / w.ws_row / ((float) w.ws_xpixel / w.ws_col);
    printf("xpixel: %d\n", w.ws_xpixel);
    printf("cols: %d\n", w.ws_col);
    printf("ypixel: %d\n", w.ws_ypixel);
    printf("rows: %d\n", w.ws_row);
    printf("ratio: %f\n", char_ratio);
    return 0;
}
