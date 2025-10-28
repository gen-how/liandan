from pathlib import Path
from zipfile import ZipFile

import requests


def download_file(url: str, target_dir: str | Path):
    """下載`url`檔案到指定的資料夾內。

    Args:
        url (str): 檔案下載網址。
        target_dir (str | Path): 目標儲存路徑。
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    filename = url.split("/")[-1]
    filepath = Path(target_dir).expanduser() / filename
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
        password (bytes | None): 壓縮檔密碼，預設值為`None`。
        with_dir (bool): 是否以壓縮檔檔名新增資料夾並解壓縮，預設值為`False`。
        ignore_cache (bool): 是否忽略系統快取，預設值為`True`。
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
