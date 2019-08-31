

import pickle

with open("fce-train.p", "rb") as fp:
    data = pickle.load(fp)

xx = data[0]
yy = data[1]

debug = 1
