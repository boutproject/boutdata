import difflib


def yes_or_no(question: str) -> bool:
    """Convert user input from yes/no variations to True/False"""
    while True:
        reply = input(f"{question} [y/N] ").lower().strip()
        if not reply or reply[0] == "n":
            return False
        if reply[0] == "y":
            return True


def create_patch(filename: str, original: str, modified: str) -> str:
    """Create a unified diff between original and modified"""

    patch = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            modified.splitlines(),
            fromfile=filename,
            tofile=filename,
            lineterm="",
        )
    )

    return patch


def apply_or_display_patch(
    filename: str,
    original: str,
    modified: str,
    patch_only: bool,
    quiet: bool,
    force: bool,
):
    """Given the original and modified versions of a file, display and/or apply it

    Parameters
    ----------
    filename : str
        Name of file
    original : str
        Original text of file
    modified : str
        Modified text of file
    patch_only : bool
        If ``True``, only print the patch
    quiet : bool
        If ``True``, don't print to screen, unless ``patch_only`` is
        ``True``
    force : bool
        If ``True``, always apply modifications to file

    """

    patch = create_patch(filename, original, modified)

    if patch_only:
        print(patch)
        return

    if not patch:
        if not quiet:
            print(f"No changes to make to {filename}")
        return

    if not quiet:
        print("\n******************************************")
        print(f"Changes to {filename}\n{patch}")
        print("\n******************************************")

    make_change = force or yes_or_no(f"Make changes to {filename}?")

    if make_change:
        with open(filename, "w") as f:
            f.write(modified)
