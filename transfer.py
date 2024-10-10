import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf

class frac:
    """
    Class for storing faction equations for use
    by the functions in the program
    """
    def __init__(self, num, den, k=1):
        """
        num - numberator, array
        den - denominator, array
        k - is the gain, floating point

        numerator and denominator do not need to be equal size,
        if not, they will be made equal by adding 0s to one
        """

        # apply k value to numerator
        for i in range(len(num)):
            num[i] *= k
        self.num = num
        self.den = den
        self.k = k
        
        # make numerator and denominator equal
        while len(num) != len(den):
            if len(num) < len(den):
                self.num.insert(0,0)
            elif len(num) > len(den):
                self.den.insert(0,0)
          
    def set_k(self,k):
        # apply k value to numerator
        for i in range(len(self.num)):
            self.num[i] *= k
        self.k = k
    def print(self):
        num_str = ""
        den_str = ""
        for i in range(len(self.num)):
            if self.num[i] != 0:
                if (len(self.num)-1) - i == 1:
                    num_str += f"{self.num[i]}s + " 
                elif (len(self.num)-1) - i != 0:
                    num_str += f"{self.num[i]}s^{len(self.num) - i - 1} + "
                else:
                    num_str += f"{self.num[i]} + "
            if self.den[i] != 0:
                if (len(self.num)-1) - i == 1:
                    den_str += f"{self.den[i]}s + "
                elif (len(self.num)-1) - i != 0:
                    den_str += f"{self.den[i]}s^{len(self.num) - i - 1} + "
                    
                else:
                    den_str += f"{self.den[i]} + "
        den_str = den_str[:-3]
        num_str = num_str[:-3]
        return f"$\\frac{{{num_str}}}{{{den_str}}}$ k = {self.k}"
        

def convolve_equations(*args):
    arg_list = list(args)
    first_arg = arg_list.pop(0)
    arg_total = frac(first_arg.num, first_arg.den)
    
    for arg in arg_list:
        arg_total.num = np.convolve(arg.num, arg_total.num)
        arg_total.den = np.convolve(arg.den, arg_total.den)
    
    return arg_total 

def db_to_freq(Hs, search_val, bode_start = 1, bode_stop = 1E3, dt=.1, fuzz = .1):
    w = np.arange(bode_start, bode_stop+dt, dt)
    system = sig.lti(Hs.num, Hs.den)
    w, Hmag, Hphase = sig.bode(system, w)
    possible_values = []
    index = 0
    in_range = False
    for value in Hmag:
        if search_val+fuzz >= value and search_val-fuzz <= value: 
            if in_range == False:
                possible_values.append([])
            in_range = True
            possible_values[-1].append((value, w[index]))
        else:
            in_range = False
        index += 1

    ret_vals = []
    for arr in possible_values:
        ret_val = arr[0]
        for val, w in arr:
            ret_val_dist = np.abs(ret_val[0] - search_val)
            cur_val_dist = np.abs(val - search_val)
            if ret_val_dist > cur_val_dist:
                ret_val = (val, w)
        ret_vals.append(ret_val)

    return ret_vals

def deg_to_freq(Hs, search_val, bode_start = 1, bode_stop = 1E3, dt=.1, fuzz = .1):
    w = np.arange(bode_start, bode_stop+dt, dt)
    system = sig.lti(Hs.num, Hs.den)
    w, Hmag, Hphase = sig.bode(system, w)
    possible_values = []
    index = 0
    in_range = False
    for value in Hphase:
        if search_val+fuzz >= value and search_val-fuzz <= value: 
            if in_range == False:
                possible_values.append([])
            in_range = True
            possible_values[-1].append((value, w[index]))
        else:
            in_range = False
        index += 1

    ret_vals = []
    for arr in possible_values:
        ret_val = arr[0]
        for val, w in arr:
            ret_val_dist = np.abs(ret_val[0] - search_val)
            cur_val_dist = np.abs(val - search_val)
            if ret_val_dist > cur_val_dist:
                ret_val = (val, w)
        ret_vals.append(ret_val)

    return ret_vals

def _get_y_from_w(w, H, value):

    start_ind = 0
    end_ind = len(w) - 1
    mid = int((end_ind - start_ind)/2)

    while w[mid] != value and (end_ind - start_ind > 1):
        if w[mid] < value:
            start_ind = mid
        else:
            end_ind = mid
        mid = int(start_ind + (end_ind - start_ind)/2)
    
    return (H[mid], w[mid])

def get_db_from_w(Hs, search_val, bode_start = 1, bode_stop = 1E3, dt=.1):
    w, Hmag, Hphase = make_bode(Hs, bode_start, bode_stop, dt)
    return _get_y_from_w(w, Hmag, search_val)

def get_phase_from_w(Hs, search_val, bode_start = 1, bode_stop = 1E3, dt=.1):
    w, Hmag, Hphase = make_bode(Hs, bode_start, bode_stop, dt)
    return _get_y_from_w(w, Hphase, search_val)

def get_margins(Hs,bode_start = 1, bode_stop = 1E3, dt=.1):
    w = np.arange(bode_start, bode_stop+dt, dt)
    system = sig.lti(Hs.num, Hs.den)
    w, Hmag, Hphase = sig.bode(system, w)
    gm, pm, wg, wp = margin(Hmag, Hphase, w)
    print(f"gm: {gm}")
    print(f"pm: {pm}")
    print(f"wg: {wg}")
    print(f"wp: {wp}")
    return gm, pm, wg, wp

def make_bode(Hs,bode_start = 1, bode_stop = 1E3, dt=.1):
    w = np.arange(bode_start, bode_stop+dt, dt)
    system = sig.lti(Hs.num, Hs.den)
    w, Hmag, Hphase = sig.bode(system, w)
    return w, Hmag, Hphase
