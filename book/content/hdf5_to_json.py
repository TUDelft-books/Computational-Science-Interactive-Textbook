import numpy as np
import os
import h5py
import json

path = "./"
dirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
all_h5s = []
for dir in dirs:
    files = os.listdir(dir)
    h5s = [file for file in files if file[-4:] == 'hdf5']
    h5s = os.path.join(dir, h5s[0]) # There will only be one per dir
    all_h5s.append(h5s)
print(all_h5s)

out_data = {}
for file in all_h5s:
    f = h5py.File(file, 'r+') 
    for key in f.keys():
        out_data[key] = np.array(f[key]).tolist()
    print(type(out_data))
    with open(f'{file[:-4]}json', 'w+') as f:
        json.dump(out_data, f, indent=3)
