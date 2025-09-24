"""
This one is hard-coded for FFT2 because the values.json was too big for git.
It was split across values_part_1.json and values_answer_8_2c_1.json
The user should not notice the difference but the check_answer function
reads both files and combines them into a single dict.
The answer 8 couldn't even be stored as a json, it had to be compressed into a npy.
"""
import json
import os.path
import numpy as np

# A library for saving and checking answers for notebooks

location = "./values_part_1.json"
location2 = "./values_answer_8_2c_1.json"

# For outputting matrices in a somewhat readable way...
def format_difference(A,B,atol):
    A = A.astype(float) ## the solution
    B = B.astype(float) ## the answer to check
    msg = ""
    n = 0
    nmax = 10
    if (len(A.shape) == 0):
        msg += "Sol = %e Yours = %e Diff %e" % (A, B, A-B)
        return msg
    elif (len(A.shape) == 1):
        for i in range(A.shape[0]):
            if not np.isclose(A[i], B[i], atol=atol):
                msg += "i = %d  Sol = %e  Yours = %e  Diff = %e\n" % (i, A[i], B[i], A[i]-B[i])
                n += 1
                if (n > nmax): 
                    return msg
    else:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if not np.isclose(A[i,j], B[i,j], atol=atol):
                    msg += "i,j = %d,%d  Sol = %e  Yours = %e  Diff = %e\n" % (i, j, A[i,j], B[i,j], A[i,j]-B[i,j])
                    n += 1
                    if (n > nmax): 
                        return msg
    return msg    

""" Depricated but may be needed in future
def save_answer(value, key):
    if not os.path.isfile(location):
        open(location, "x")
    with h5py.File(location, "r+") as f:
        try: 
            del f[key]
        except: 
            print("Creating new key %s" % key)
        f.create_dataset(key, data=value)
"""

def check_answer(value, key, atol=None):
    if not os.path.isfile(location):
        print("(NEW) Missing answers file, instructor must run all cells then run cell at end of notebook to generate file")
        return(True, "")
    # Load the two partial files and combine them 
    with open(location) as f:
        d1 = json.load(f)
    array_for_answer_8_2c_1 = np.load(location2)
    d2 = {'answer_8_2c_1': list(array_for_answer_8_2c_1)}
    d = d1|d2 # Combine the partial dicts into one dict

    try:
        sol_answer = np.array(d[key]) 
    except: 
        print("Missing key, instructor must run all cells to generate and save answers")
        return(True, "")
        
    student_answer = np.array(value)

    # The atol argument was a flawed idea. We should pick an absolute tolerance that is 
    # related to the MAX value of the correct solution answer. Let's say 1 part in 10^4. 
    
    if not atol:
        atol = 1e-3*np.max(sol_answer[~np.isnan(sol_answer)])
    
    print("\nTolerance set to %e" % atol)
    
    msg = "\nFeedback %s:\n" % key
    msg += "="*(len(msg)-2)
    msg += "\n\n"

    # First check that the answer array is the correct shape:
    if student_answer.shape != sol_answer.shape:
        msg += "Your answer has the wrong dimensions: its shape should be: " + str(sol_answer.shape)
        return (False, msg)
    # Then check if values are correct
    elif np.allclose(student_answer, sol_answer,atol=atol):
        msg += "Your answer is correct"
        return (True, msg)
    else:
        msg += "Your answer has the correct shape but contains incorrect values. \n\n" + \
        "The first 10 non-matching entries are:\n" + \
        str(format_difference(sol_answer, student_answer, atol))
        return (False, msg)

"""
student_answer = np.array([1.0,2.0,3.0])
sol_answer = np.array([1,np.nan,np.nan])

check_answer(sol_answer, "answer_3_01_1")
"""
