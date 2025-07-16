# Source files
SRC_C	=	src/main.c\
			src/image.c\
			src/print.c\
			src/thread.c\
			src/terminal.c\
			src/load_image.c\
			src/image_loaders/png.c\
			src/image_loaders/gif.c\
			src/image_loaders/bmp.c\
			src/image_loaders/jpeg.c\
			src/image_loaders/webp.c\

SRC_CU	=	src/init_gpu.cu\
			src/resize_cuda.cu\
			src/cuda_kernels.cu\

OBJ_C	=	$(SRC_C:.c=.o)
OBJ_CU	=	$(SRC_CU:.cu=.o)

# Compilation parameters
GPU_COMPILER	=	nvcc
C_COMPILER	=	gcc
LIBS	=	-lm -lpthread -lpng -ljpeg -lgif -lwebp -lwebpdemux
GPU_COMPILER_FLAGS	=	-DUSE_CUDA -O2 -Wno-deprecated-gpu-targets -g -Xcompiler -fPIC $(LIBS)
C_COMPILER_FLAGS	=	-O2 -Wall -Wextra -W -g $(LIBS)
MAKEFLAGS	=	-j$(shell nproc) --silent --no-print-directory

NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
	COMPILER    =   $(C_COMPILER)
	COMPILER_FLAGS  = $(C_COMPILER_FLAGS)
	OBJ =   $(OBJ_C)
	TOTAL_FILES =   $(SRC_C)
else
	COMPILER    =   $(GPU_COMPILER)
	COMPILER_FLAGS  =   $(GPU_COMPILER_FLAGS)
	OBJ =   $(OBJ_CU) $(OBJ_C)
	TOTAL_FILES =   $(SRC_C) $(SRC_CU)
endif

N_FILES	:=	$(words $(TOTAL_FILES))
CURRENT_FILE	:=	0
NAME	=	ImageVisualizer

# Colors
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
BLUE=\033[0;34m
PURPLE=\033[0;35m
CYAN=\033[0;36m
WHITE=\033[0;37m
RESET=\033[0m

# Main rule
all:	$(NAME)

# Compilation of the library binary
$(NAME):
	@echo -ne "[$(YELLOW)$(NAME)$(RESET)] "
	@echo -ne "Compiling all objects files...\n"
	@echo ""
	@make $(OBJ)
	@echo -ne "\n"
	@echo ""
	@echo -ne "--------------------------------------------\n"
	@echo ""
	@echo -ne "[$(YELLOW)$(NAME)$(RESET)] "
	@echo -ne "Compiling the main program...\n"
	@$(COMPILER) -o $(NAME) $(OBJ) $(COMPILER_FLAGS)
	@echo ""
	@echo -ne "[$(YELLOW)$(NAME)$(RESET)] "
	@echo -ne "Main program successfully compiled !\n"

# Rule to compile objects files
$(OBJ_C): %.o: %.c
	@echo -ne "[$(BLUE)COMPILATION$(RESET)] "
	@echo -ne "($(shell expr $(CURRENT_FILE) + 1)/$(N_FILES)) $@\r"
	@$(COMPILER) -c $< -o $@ $(COMPILER_FLAGS)
	@$(eval CURRENT_FILE := $(shell expr $(CURRENT_FILE) + 1))

$(OBJ_CU): %.o: %.cu
	@echo -ne "[$(BLUE)COMPILATION$(RESET)] "
	@echo -ne "($(shell expr $(CURRENT_FILE) + 1)/$(N_FILES)) $@\r"
	@$(COMPILER) -c $< -o $@ $(COMPILER_FLAGS)
	@$(eval CURRENT_FILE := $(shell expr $(CURRENT_FILE) + 1))

# Rule to remove all object files
clean:
	@echo -ne "[$(RED)REMOVE$(RESET)] "
	@echo -ne "Removing all object files...\n"
	@rm -f $(OBJ_C)
	@rm -f $(OBJ_CU)
	@echo -ne "[$(RED)REMOVE$(RESET)] "
	@echo -ne "All objects files successfully removed !\n"

# Rule to remove all binary files
fclean:
	@make clean
	@echo -ne "[$(RED)REMOVE$(RESET)] "
	@echo -ne "Removing Main program binary...\n"
	@rm -f $(NAME)
	@echo -ne "[$(RED)REMOVE$(RESET)] "
	@echo -ne "Main program binary successfully removed !\n"
	@echo ""
	@echo -ne "[$(RED)REMOVE$(RESET)] "
	@echo -ne "Repository successfully cleaned !\n"

# Rule to fully recompile the prgram
re:
	@make fclean
	@echo ""
	@echo -ne "--------------------------------------------\n"
	@echo ""
	@echo -ne "[$(YELLOW)$(NAME)$(RESET)] "
	@echo -ne "Recompiling the main program\n"
	@echo ""
	@echo -ne "--------------------------------------------\n"
	@echo ""
	@make
