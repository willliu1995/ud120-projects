#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py 
"""
import os
os.getcwd()
os.chdir(r"C:\\Users\\willl\\OneDrive - WillfortheFuture\\My Documents\\Study\\Python\Udacity\\Introduction to Machine Learning\\ud120-projects\\tools")

dict_files={
    "word_data.pkl":"word_data_unix.pkl",
    "python2_lesson06_keys.pkl":"python2_lesson06_keys_unix.pkl",
    "python2_lesson13_keys.pkl":"python2_lesson13_keys_unix.pkl",
    "python2_lesson14_keys.pkl":"python2_lesson14_keys_unix.pkl",
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