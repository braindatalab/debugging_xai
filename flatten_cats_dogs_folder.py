import os
import shutil
import hashlib

# ====== CONFIG ======
base_dir = "./images"  # top-level folder
categories = ["cats", "dogs"]
rename_map = {"cats": "cat", "dogs": "dog"}
allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
chunk_size = 1024 * 1024  # 1MB for hashing large files
# ====================

def is_image_file(name: str) -> bool:
    if not name or name.startswith("."):
        return False
    ext = os.path.splitext(name)[1].lower()
    return ext in allowed_ext

def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def unique_destination(dst_dir: str, filename: str) -> str:
    """Return a non-colliding destination path by appending _1, _2, ... if needed."""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dst_dir, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dst_dir, f"{base}_{counter}{ext}")
        counter += 1
    return candidate

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def remove_empty_dirs(root: str, keep: set) -> None:
    """Remove empty directories under root, except those in keep (by absolute path)."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        # Skip protected dirs
        if os.path.abspath(dirpath) in keep:
            continue
        # If empty, remove
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass

def main():
    # Source folders (double nested)
    source_roots = [
        os.path.join(base_dir, "training_set", "training_set"),
        os.path.join(base_dir, "test_set", "test_set"),
    ]

    # Targets
    target_dirs = {old: os.path.join(base_dir, new) for old, new in rename_map.items()}
    for td in target_dirs.values():
        ensure_dir(td)

    # Track duplicates by content hash (global across both categories & sets)
    seen_hashes = {}

    stats = {
        "moved": 0,
        "duplicates_deleted": 0,
        "skipped_non_images": 0,
        "missing_sources": [],
        "errors": 0,
    }

    for category in categories:
        dst_dir = target_dirs[category]
        # Gather possible source directories for this category
        cat_sources = [os.path.join(sr, category) for sr in source_roots]

        for src_dir in cat_sources:
            if not os.path.isdir(src_dir):
                stats["missing_sources"].append(src_dir)
                continue

            for name in os.listdir(src_dir):
                src_path = os.path.join(src_dir, name)

                # Skip subdirectories; we only expect files here
                if not os.path.isfile(src_path):
                    continue

                # Skip junk/non-images
                if not is_image_file(name):
                    stats["skipped_non_images"] += 1
                    continue

                try:
                    # Content-based deduplication
                    h = hash_file(src_path)
                    if h in seen_hashes:
                        # Duplicate content found: delete the later copy
                        try:
                            os.remove(src_path)
                            stats["duplicates_deleted"] += 1
                        except OSError:
                            stats["errors"] += 1
                        continue

                    # Move to destination (resolve name collisions)
                    dst_path = unique_destination(dst_dir, name)
                    shutil.move(src_path, dst_path)
                    seen_hashes[h] = dst_path
                    stats["moved"] += 1

                except Exception:
                    stats["errors"] += 1
                    # Continue with the rest

    # Clean up empty directories under base_dir
    keep_dirs = {os.path.abspath(target_dirs["cats"]), os.path.abspath(target_dirs["dogs"]), os.path.abspath(base_dir)}
    remove_empty_dirs(base_dir, keep=keep_dirs)

    # Summary
    print("Done.")
    print(f"Moved images: {stats['moved']}")
    print(f"Duplicates removed (by content): {stats['duplicates_deleted']}")
    print(f"Non-images skipped: {stats['skipped_non_images']}")
    if stats["missing_sources"]:
        print("Missing source dirs (ok if some sets weren't present):")
        for p in stats["missing_sources"]:
            print("  -", p)
    if stats["errors"]:
        print(f"Errors encountered: {stats['errors']}")

if __name__ == "__main__":
    main()
