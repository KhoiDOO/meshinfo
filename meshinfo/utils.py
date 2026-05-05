from colorama import Fore, Style
from .constants import FORMAT_PRECISION_FLOAT

def format_bool(value):
    """Format boolean values with color."""
    if value is True:
        return f"{Fore.GREEN}True{Style.RESET_ALL}"
    elif value is False:
        return f"{Fore.RED}False{Style.RESET_ALL}"
    return str(value)

def format_value(value):
    """Format values with appropriate color based on type."""
    if isinstance(value, bool):
        return format_bool(value)
    elif isinstance(value, (int, float, complex)):
        if isinstance(value, int):
            return f"{Fore.YELLOW}{value:,}{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}{value:.{FORMAT_PRECISION_FLOAT}f}{Style.RESET_ALL}"
    else:
        return f"{Fore.WHITE}{value}{Style.RESET_ALL}"
