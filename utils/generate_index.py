import os
import sys


def get_n_vertices(file_path):
    """Read a file and return the N_VERTICES value if found."""
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("N_VERTICES"):
                parts = line.strip().split()
                return int(parts[1])
    return 0


def generate_index(output_files, index_file):
    max_vertices = -1
    max_file = None

    with open(index_file, "w") as f:
        for file_path in output_files:
            basename = os.path.basename(file_path)
            track_name = os.path.splitext(basename)[0]
            display_name = track_name.replace("_", " ").title()
            # Keep hyphens nicely capitalised
            parts = display_name.split("-")
            display_name = "-".join(p.strip().title() for p in parts)
            f.write(f"{track_name}|{display_name}\n")

            n_vertices = get_n_vertices(file_path)
            if n_vertices > max_vertices:
                max_vertices = n_vertices
                max_file = file_path

    if max_file:
        print(f"Track with most vertices ({max_vertices}): {max_file}")


def main():
    index_file = sys.argv[1]
    output_files = sys.argv[2:]
    generate_index(output_files, index_file)


if __name__ == "__main__":
    main()
