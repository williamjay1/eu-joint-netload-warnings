"""Explicit clone/work/raw locations for optional full-pipeline source."""
from pathlib import Path
import os
WORK_ROOT = Path(os.environ.get('EU_NETLOAD_WORK_ROOT', Path(__file__).resolve().parents[1])).resolve()
DEFAULT_RAW_ROOT = Path('F:/AcademicData/eu_joint_netload/raw') if os.name == 'nt' else WORK_ROOT / 'raw'
RAW_ROOT = Path(os.environ.get('EU_NETLOAD_RAW_ROOT', DEFAULT_RAW_ROOT)).resolve()
def work_path(relative=''):
    return WORK_ROOT / relative
def raw_path(relative=''):
    return RAW_ROOT / relative
