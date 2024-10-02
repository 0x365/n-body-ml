import csv
import numpy as np

# from scipy.fft import fft, fftfreq
import math
import numpy as np

from tqdm import tqdm
import itertools

def csv_input(file_name):
    with open(file_name, "r") as f:
        read_obj = csv.reader(f)
        output = []
        for row in read_obj:
            output.append(row)
    f.close()
    return output

total_num = 50

combos = list(itertools.product(range(-total_num, total_num+1), repeat=2))
for ii in tqdm(combos):
    i = ii[0]
    j = ii[1]
    v1 = 1.5 * i / total_num
    v2 = 1.5 * j / total_num
    try:
        data = np.array(csv_input("data/"+str(v1)+"_"+str(v2)+".csv"),dtype=float)
        np.save("data-np/"+str(v1)+"_"+str(v2), data)
        del data
    except:
        pass