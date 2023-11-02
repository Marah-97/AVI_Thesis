# Function to count JPG files in a directory
import os
import glob

good_vials_dirs = [r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\Good\KR40424\CAM4",
                   r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\Good\KR40433\CAM4",
                   r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\Good\KR40597\CAM4",
                   r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\Good\KR40655\CAM4"]
bad_vials_dirs = [r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\ChipB\A\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\ChipB\B\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\ChipB\C\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\ChipB\D\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\ChipB\E\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\ChipB\F\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\ChipB\G\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\ChipB\MW63546\CAM4"]

def count_jpg_files(directory):
    jpg_files = glob.glob(os.path.join(directory, "*.jpg"))
    return len(jpg_files)


total_good_files = 0
total_bad_files = 0

for directory in good_vials_dirs:
    num_jpg_files = count_jpg_files(directory)
    total_good_files += num_jpg_files
    print(f"Good Directory: {directory}, JPG files: {num_jpg_files}")

for directory in bad_vials_dirs:
    num_jpg_files = count_jpg_files(directory)
    total_bad_files += num_jpg_files
    print(f"Bad Directory: {directory}, JPG files: {num_jpg_files}")

print(f"Total Good JPG files: {total_good_files}")
print(f"Total Bad JPG files: {total_bad_files}")