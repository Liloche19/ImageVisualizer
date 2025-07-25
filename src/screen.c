#include "../include/visualizer.h"

void get_screen_informations(Screen *settings)
{
    struct winsize win;

    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &win) == -1) {
        settings->cols = DEFAULT_NB_COLS;
        settings->rows = DEFAULT_NB_ROWS;
        settings->char_ratio = DEFAULT_CHAR_RATIO;
        settings->print_buffer = NULL;
        return;
    }
    settings->cols = win.ws_col;
    settings->rows = win.ws_row;
    if (win.ws_xpixel != 0 && win.ws_ypixel != 0)
        settings->char_ratio = (float) win.ws_ypixel / win.ws_row / ((float) win.ws_xpixel / win.ws_col);
    else
        settings->char_ratio = DEFAULT_CHAR_RATIO;
    settings->print_buffer = NULL;
    return;
}
