import trimesh
import numpy as np
import fcl
from colorama import Fore, Style, init

from ..constants import FORMAT_LABEL_WIDTH, FORMAT_PRECISION_COORD
from . import format_value, format_bool

# Initialize colorama for Windows compatibility
init(autoreset=True)


class PointInfo:
    def __init__(self, points: np.ndarray, normals: np.ndarray=None, name:str=""):
        self.points = points
        self.normals = normals
        self.name = name

        self.stats = {
            'num_points': len(points),
            'has_normals': normals is not None,
        }

        self.analysis = {
            'bounds': [np.min(points, axis=0), np.max(points, axis=0)],
        }
    
    def __str__(self):
        info_str = f"{Fore.CYAN}{Style.BRIGHT}╔═══ Point Cloud Information [{self.name}] ═══╗{Style.RESET_ALL}\n"

        # Statistics
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Statistics:{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'#points':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL}"
        info_str += f"  {Fore.CYAN}{format_value(self.stats['num_points'])}{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'has_normals':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL}"
        info_str += f"  {Fore.CYAN}{format_bool(self.stats['has_normals'])}{Style.RESET_ALL}\n"

        # Analysis
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Analysis:{Style.RESET_ALL}\n"
        for key, value in self.analysis.items():
            if key == "bounds":
                value_str = f"{Fore.YELLOW}[[{value[0][0]:.{FORMAT_PRECISION_COORD}f}, {value[0][1]:.{FORMAT_PRECISION_COORD}f}, {value[0][2]:.{FORMAT_PRECISION_COORD}f}], "
                value_str += f"[{value[1][0]:.{FORMAT_PRECISION_COORD}f}, {value[1][1]:.{FORMAT_PRECISION_COORD}f}, {value[1][2]:.{FORMAT_PRECISION_COORD}f}]]{Style.RESET_ALL}"
                info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {value_str}\n"

        return info_str