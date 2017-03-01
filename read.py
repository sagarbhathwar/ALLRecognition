import glob
import os

import pandas as pd


# Read image path and the co-ordinates of blast cells given in the xyc file
# into a pandas dataframe for easier processing
def read_data(img_path, xyc_path):
    # Read the names/paths of all necessary files
    img_files = glob.glob(os.path.join(img_path, '*'))
    xyc_files = glob.glob(os.path.join(xyc_path, '*'))

    row_list = []
    for i in range(len(xyc_files)):
        img_file = img_files[i]
        has_blasts = bool(int(img_file[-5]))

        f = open(xyc_files[i])
        points = []
        for line in f.readlines():
            # Check if the line is not just line termination to avoid
            # spurious empty tuple as a point
            line = line.strip()
            if line:
                points.append(tuple(map(int, line.split())))

        d = {"id": i + 1, "img_path": img_file, "has_blasts": has_blasts,
             "blast_xy": points}
        row_list.append(d)

    df = pd.DataFrame(row_list)
    return df[["id", "img_path", "has_blasts", "blast_xy"]]
