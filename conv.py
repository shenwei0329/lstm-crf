# coding: utf-8
#

import sys

f = open(sys.argv[1], "r", encoding='utf-8')
o = open("out_data", "w", encoding='utf-8')

while True:
    line = f.readline()
    line = line.replace("\r", '')
    # print(line, end="")
    o.write(line)
    if len(line) < 1:
        break

f.close()
o.close()

