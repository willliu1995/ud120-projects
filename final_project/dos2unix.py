#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py 
"""
import os
os.getcwd()
os.chdir(r"C:\\Users\\willl\\OneDrive - WillfortheFuture\\My Documents\\Study\\Python\Udacity\\Introduction to Machine Learning\\ud120-projects\\final_project")

# original = ["final_project_dataset.pkl", ]
# destination = "final_project_dataset_unix.pkl"

dict_files = {
    "final_project_dataset.pkl":        "final_project_dataset_unix.pkl",
    "final_project_dataset_modified.pkl": "final_project_dataset_modified_unix.pkl"
}

content = ''
outsize = 0

for (original, destination) in dict_files.items():
    with open(original, 'rb') as infile:
        content = infile.read()
    with open(destination, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))
