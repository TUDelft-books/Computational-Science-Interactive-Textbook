import numpy as np
import os
import h5py
import json

def main():
    path = "./"
    dirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    all_h5s = []
    for dir in dirs:
        files = os.listdir(dir)
        h5s = [file for file in files if file[-4:] == 'hdf5']
        if h5s != []:
            h5s = os.path.join(dir, h5s[0]) # There will only be one per dir
            all_h5s.append(h5s)
    print(all_h5s)
    
    out_data = {}
    for file in all_h5s:
        f = h5py.File(file, 'r+') 
        for key in f.keys():
            datum = np.array(f[key]).tolist()
            try:
                if type(datum[0]) == complex:
                    print("Needed to use complex function!")
                    datum = complex_list_to_real_and_imag_dict(datum)
                    print(key)
                else:
                    pass
            except TypeError:  
                pass
            out_data[key] = datum
        with open(f'{file[:-4]}json', 'w+') as f:
            json.dump(out_data, f, indent=3)
        print(f"Converted file {file}")
    return 

def complex_list_to_real_and_imag_dict(data):
    reals = []
    imags = []
    for dat in data:
        reals = dat.real
        imags = dat.imag
    out_dict = {}
    out_dict['real'] = reals
    out_dict['imag'] = imags
    return out_dict


main()
