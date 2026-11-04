"""Finding the File Explorer window that shows a folder, so it can be brought to the front."""
from pathlib import Path

from editp.explorer import shows

FOLDER = Path(r"C:\Users\someone\Pictures\EditP")


def test_a_title_naming_the_folder_matches():
    assert shows("EditP - 檔案總管", FOLDER)
    assert shows("EditP - File Explorer", FOLDER)
    assert shows(r"C:\Users\someone\Pictures\EditP - 檔案總管", FOLDER)  # full path in the title bar
    assert shows("EditP", FOLDER)


def test_other_folders_do_not_match():
    assert not shows("EditP old - 檔案總管", FOLDER)
    assert not shows("Pictures - 檔案總管", FOLDER)
    assert not shows("my EditP - 檔案總管", FOLDER)
    assert not shows("", FOLDER)
