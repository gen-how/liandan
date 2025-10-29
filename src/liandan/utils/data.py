from hashlib import md5
from pathlib import Path
from zipfile import ZipFile

import requests


def calculate_md5(path: str | Path, chunk_size=1024 * 1024) -> str:
    """計算指定檔案的 MD5 雜湊值。

    此函式由`torchvision.datasets.utils.calculate_md5`修改而來。

    Args:
        path (str | Path): 檔案路徑。
        chunk_size (int, optional): 每次讀取的區塊大小。

    Returns:
        out (str): MD5 雜湊值字串。
    """
    hasher = md5(usedforsecurity=False)
    path = Path(path).expanduser()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str, filepath: str | Path):
    """下載`url`檔案到指定的檔案路徑。

    Args:
        url (str): 檔案下載網址。
        filepath (str | Path): 目標儲存路徑。
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    filepath = Path(filepath).expanduser()
    with filepath.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def extract_zip(
    path: str | Path,
    target_dir: str | Path,
    password=None,
    with_dir=False,
    ignore_cache=True,
):
    """解壓縮指定的檔案到指定的資料夾內。

    Args:
        path (str | Path): 壓縮檔路徑。
        target_dir (str | Path): 解壓縮目標資料夾。
        password (str, optional): 壓縮檔密碼。
        with_dir (bool, optional): 是否以壓縮檔檔名新增資料夾並解壓縮。
        ignore_cache (bool, optional): 是否忽略系統快取。
    """
    path = Path(path).expanduser()
    target_dir = Path(target_dir).expanduser()
    if with_dir:
        target_dir = target_dir / path.stem

    target_dir.mkdir(parents=True, exist_ok=True)

    def is_cache(name):
        return (
            name.startswith("__MACOSX/")
            or name.endswith(".DS_Store")
            or name.endswith("Thumbs.db")
        )

    with ZipFile(path, "r") as zf:
        if ignore_cache:
            members = (name for name in zf.namelist() if not is_cache(name))
        else:
            members = None
        zf.extractall(target_dir, members, pwd=password)
