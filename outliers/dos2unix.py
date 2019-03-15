# !/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py 
"""
import os
os.getcwd()
os.chdir(r"C:\\Users\\willl\\OneDrive - WillfortheFuture\\My Documents\\Study\\Python\\Udacity\\Introduction to Machine Learning\\ud120-projects\\outliers")

dict_files={
    "practice_outliers_ages.pkl":"practice_outliers_ages_unix.pkl",
    "practice_outliers_net_worths.pkl":"practice_outliers_net_worths_unix.pkl"
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