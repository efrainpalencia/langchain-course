from datetime import datetime


class Colors:
    RESET = "\033[0m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(message: str) -> None:
    print(
        f"{Colors.CYAN}[INFO]{Colors.RESET} "
        f"{Colors.WHITE}{_timestamp()} - {message}{Colors.RESET}"
    )


def log_success(message: str) -> None:
    print(
        f"{Colors.GREEN}[SUCCESS]{Colors.RESET} "
        f"{Colors.WHITE}{_timestamp()} - {message}{Colors.RESET}"
    )


def log_warning(message: str) -> None:
    print(
        f"{Colors.YELLOW}[WARNING]{Colors.RESET} "
        f"{Colors.WHITE}{_timestamp()} - {message}{Colors.RESET}"
    )


def log_error(message: str) -> None:
    print(
        f"{Colors.RED}[ERROR]{Colors.RESET} "
        f"{Colors.WHITE}{_timestamp()} - {message}{Colors.RESET}"
    )


def log_header(title: str) -> None:
    line = "=" * (len(title) + 8)
    print(
        f"\n{Colors.MAGENTA}{Colors.BOLD}{line}{Colors.RESET}\n"
        f"{Colors.MAGENTA}{Colors.BOLD}===  {title}  ==={Colors.RESET}\n"
        f"{Colors.MAGENTA}{Colors.BOLD}{line}{Colors.RESET}\n"
    )
