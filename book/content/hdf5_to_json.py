import sys
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
            print(f'{key} is size {sys.getsizeof(datum)}')
            print(np.array(datum).shape)
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


def main_for_big_files():
    path = "./"
    dirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    all_h5s = []
    for dir in dirs:
        files = os.listdir(dir)
        h5s = [file for file in files if file[-4:] == 'hdf5']
        if h5s != []:
            h5s = os.path.join(dir, h5s[0]) # There will only be one per dir
            all_h5s.append(h5s)
    
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
            long_key = 'answer_8_2c_1'
            if key == long_key:
                out_array = np.array(datum) # Too big to be saved as a json, MUST be compressed
            else:
                out_data[key] = datum
        with open(f'{file[:-5]}_part_1.json', 'w+') as f:
            json.dump(out_data, f, indent=3)
        np.save(f'{file[:-5]}_{long_key}.npy', out_array)
        print(f"Converted file {file}")
    return 

def complex_list_to_real_and_imag_dict(data):
    reals = []
    imags = []
    for dat in data:
        reals.append(dat.real)
        imags.append(dat.imag)
    out_dict = {}
    out_dict['real'] = reals
    out_dict['imag'] = imags
    return out_dict


main_for_big_files()
#main()
