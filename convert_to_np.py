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

total_num = 100

combos = list(itertools.product([0, *range(-total_num, total_num+1)], repeat=2))
for ii in tqdm(combos):
    i = ii[0]
    j = ii[1]
    try:
        v1 = 2.0 * i / total_num
        v2 = 2.0 * j / total_num
        if round(v1) == v1:
            v1 = round(v1)
        if round(v2) == v2:
            v2 = round(v2)
        data = np.array(csv_input("data/"+str(v1)+"_"+str(v2)+".csv"),dtype=float)
        np.save("data-np/"+str(i)+"_"+str(j), data)
        del data
    except:
        print(v1,v2, i,j)
        pass


# # for i in range(-100,101):
# #     j = 100
# #     v1 = 2.0 * i / total_num
# #     v2 = 2.0 * j / total_num
# #     if round(v1) == v1:
# #         v1 = round(v1)
# #     if round(v2) == v2:
# #         v2 = round(v2)

# #     data = np.array(csv_input("data/"+str(v1)+"_"+str(v2)+".csv"),dtype=float)
# #     np.save("data-np/"+str(i)+"_"+str(j), data)
# #     del data

# for i in range(-100,101):
#     j = 0
#     v1 = 2.0 * i / total_num
#     v2 = 2.0 * j / total_num
#     if round(v1) == v1:
#         v1 = round(v1)
#     if round(v2) == v2:
#         v2 = round(v2)

#     data = np.array(csv_input("data/"+str(v1)+"_"+str(v2)+".csv"),dtype=float)
#     np.save("data-np/"+str(i)+"_"+str(-100), data)
#     del data