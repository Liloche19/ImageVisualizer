# Source files
SRC	=	src/main.c

OBJ	=	$(SRC:.c=.o)
N_FILES	:=	$(words $(SRC))
CURRENT_FILE	:=	0
NAME	=	ImageVisualizer

# Compilation flags
CFLAGS	=	-Wall -Wextra -W -g -lm
MAKEFLAGS	=	-j$(shell nproc) --silent --no-print-directory

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
	@gcc -o $(NAME) $(OBJ) $(CFLAGS)
	@echo ""
	@echo -ne "[$(YELLOW)$(NAME)$(RESET)] "
	@echo -ne "Main program successfully compiled !\n"

# Rule to compile objects files of the library
$(OBJ): %.o: %.c
	@echo -ne "[$(BLUE)COMPILATION$(RESET)] "
	@echo -ne "($(shell expr $(CURRENT_FILE) + 1)/$(N_FILES)) $@\r"
	@gcc -c $< -o $@ $(CFLAGS)
	@$(eval CURRENT_FILE := $(shell expr $(CURRENT_FILE) + 1))

# Rule to remove all object files
clean:
	@echo -ne "[$(RED)REMOVE$(RESET)] "
	@echo -ne "Removing all object files...\n"
	@rm -f $(OBJ)
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
