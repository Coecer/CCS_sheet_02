import logging
import re
from pathlib import Path

def create_plots_directory():
    """
    Create a plots directory if it does not already exist and returns it.
    """
    # Define the root output directory
    HERE = Path(__file__).resolve().parent.parent
    PLOTS_DIR = HERE / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)
    logging.info(f'Created plots directory: {PLOTS_DIR}')

    return PLOTS_DIR

def create_output_directory():
    """
    Create an outputs directory with an auto-incremented name based on existing directories.
    The directory will be named 'run_<number>', where <number> is the next available integer.
    """

    # Define the root output directory
    HERE = Path(__file__).resolve().parent.parent
    ROOT_OUT = HERE / "outputs"
    ROOT_OUT.mkdir(exist_ok=True)

    # Find existing run directories and determine the next index
    existing = [p for p in ROOT_OUT.iterdir() if p.is_dir() and p.name.startswith("run_")]
    run_ids = []
    
    for p in existing:
        m = re.match(r"run_(\d+)$", p.name)
        if m:
            run_ids.append(int(m.group(1)))

    run_idx = max(run_ids, default=0) + 1
    OUTDIR = ROOT_OUT / f"run_{run_idx}"
    OUTDIR.mkdir()
    
    print(f"[INFO] Writing all files to {OUTDIR.relative_to(HERE)}")
    return OUTDIR

def setup_logging(OUTDIR):
    
    # Define a consistent log message format
    FMT = "%(asctime)s  [%(levelname)s]  %(message)s"
    # Configure the root logger
    logging.basicConfig(level=logging.INFO, format=FMT, datefmt="%H:%M:%S")
    fh = logging.FileHandler(OUTDIR / "simulation.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter(FMT, datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

# def setup_logging(outdir: Path, level=logging.INFO):
#     """
#     Set up logging to both the console and a log file inside the output directory.

#     Parameters
#     ----------
#     outdir : Path
#         Directory where the log file ('simulation.log') will be stored.
#     level : int, optional
#         Logging level (default: logging.INFO).
#     """

#     # Define a consistent log message format
#     log_format = "%(asctime)s  [%(levelname)s]  %(message)s"
#     date_format = "%H:%M:%S"

#     # Configure the root logger
#     logging.basicConfig(
#         level=level,
#         format=log_format,
#         datefmt=date_format,
#         handlers=[
#             logging.StreamHandler(),  # Print to console
#             logging.FileHandler(outdir / "simulation.log", mode="w", encoding="utf-8")  # Save to file
#         ]
#     )

#     logging.info("Logging initialized.")
#     logging.info(f"All logs will be saved to: {outdir / 'simulation.log'}")