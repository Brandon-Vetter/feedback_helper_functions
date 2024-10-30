from transfer import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt, ceil


def lowpass_filter(n, wc = 1, Debug=False, return_list = False):
    itier = int(ceil(n)/2)
    offset_start = 180/(2*n)
    frac_list = []
    filt = frac([1], [1])
    for i in range(itier):
        #print(offset_start + i*(180/n))
        cur_frac = frac([0, 0, wc**2],
                        [1, 2*np.cos(np.deg2rad(offset_start + i*(180/n)))*wc*1, wc**2])
        filt = convolve_equations(filt, cur_frac)
        frac_list.append(cur_frac)
        if Debug:
            print(cur_frac.print())

    if ceil(n)%2 == 1:
        cur_frac = frac([0, wc], [1, wc])
        filt = convolve_equations(filt, cur_frac)
        frac_list.append(cur_frac)
        if Debug:
            print(cur_frac.print())
    
    if not return_list:
        return filt
    else:
        return frac_list

def highpass_filter(n, wc = 1, Debug=False):
    itier = int(ceil(n)/2)
    offset_start = 180/(2*n)

    filt = frac([1], [1])
    for i in range(itier):
        #print(offset_start + i*(180/n))
        cur_frac = frac([1, 0, 0],
                        [1, 2*np.cos(np.deg2rad(offset_start + i*(180/n)))*wc*-1, wc**-2])
        filt = convolve_equations(filt, cur_frac)
        if Debug:
            print(cur_frac.print())

    if ceil(n)%2 == 1:
        cur_frac = frac([1, 0], [1, wc*-1])
        filt = convolve_equations(filt, cur_frac)
        if Debug:
            print(cur_frac.print())
    
    return filt

def bandpass_filter(n, wc=1, passband=0, Debug=False, **kargs):
    pb_start = None
    pb_stop = None
    for key, arg in kargs.items():
        if key == "pb_start":
            pb_start = arg
        if key == "pb_stop":
            pb_start = arg
        
    if pb_start and pb_stop != None:
        passband = pb_stop - pb_start

    n = int(ceil(n)/2)
    corner = passband
    filters = lowpass_filter(n, corner, return_list=True)
    for filt in filters:
        if len(filt.num) == 3:
            filt.den = [1, filt.den[1], filt.den[2]+2*wc**2, filt.den[1]*wc**2, wc**4]
            filt.num = [0, 0, filt.num[-1], 0, 0]
        else:
            filt.den = [1, filt.den[1], wc**2]
            filt.num = [0, filt.num[-1], 0]
        if Debug:
            print(filt.print())

    return convolve_equations(*filters)


def find_np(wp, hp, wc=1):
    wp = wp/wc
    Hp2 = (hp)**2
    npp = np.log10( 1/Hp2 - 1)/(2*np.log10(wp) )
    return npp

def find_ns(ws, hs, wc=1):
    ws = ws/wc
    Hs2 = (hs)**2
    ns = np.log10( 1/Hs2 - 1)/(2*np.log10(ws) )
    return ns

def to_time_domain(transfer, f, NN, dt):
    A,B,C,D = sig.tf2ss(transfer.num,transfer.den)
    x = np.zeros(np.shape(B))
    y = np.zeros(len(NN))

    for m in range(len(NN)):
        x = x + dt*A.dot(x) + dt*B*f[m]
        y[m] = C.dot(x) + D*f[m]
    return y

def pole_sweep(start, stop, jump, wp, hp, ws, hs, Debug=False):
    ret_val = None
    last_np = -1
    last_ns = -1
    while start <= stop:
        try:
            npp = find_np(wp, hp, wc = start)
            ns = find_ns(ws, hs, wc=start)
            if Debug:
                print(f"np: {npp}, ns: {ns}, wc: {start}")
            if ceil(npp) != ceil(ns) or ns < 0 or npp < 0:
                start+=jump
                continue
        except OverflowError:
            start+=jump
            continue

        if ceil(last_np) > ceil(npp) or last_np == -1:
            last_np = npp
            ret_val = start
            last_ns = ns
        start+=jump
    return ret_val, last_np, last_ns

def find_chebyshev_ab(n, hp):
    epsilon = np.sqrt(1/np.abs(hp)**2 - 1)
    alpha = 1/epsilon + np.sqrt(1 + 1/epsilon**2)

    a = .5*(alpha**(1/n) - alpha**(-1/n))
    b = .5*(alpha**(1/n) + alpha**(-1/n))
    return a, b

def find_n_chebyshev(wp, hp, hs, wc=1):
    ws = ws/wc
    epsilon = np.sqrt(1/((hp**2) - 1))
    part1 = np.arccosh((1/epsilon**2)*np.sqrt((1/(1 - np.abs(hp)**2))))
    part2 = 1/(np.arccosh(ws))
    return part1*part2

def find_n_chebyshevII(wp, hp, hs, wc=1):
    wp = wp/wc
    epsilon = np.sqrt(hp**2/((hp**2) - 1))
    part1 = np.arccosh((1/epsilon**2)*np.sqrt((1/(1-np.abs(hp)**2))))
    part2 = 1/(np.arccosh(wp))
    return part1*part2


def find_chebyshev2_ab(wp, ws, hp, hs):
    epsilon = np.sqrt(hs**2/(1 - hs**2))

    a = np.arccosh(np.sqrt(1/(epsilon**2*(1-hp**2))))
    b = (1/np.arccosh(1/wp))
    return a, b

def chebyshev_filter(n, hp, wc=1, return_list=False, calulate_K=True):
    a, b = find_chebyshev_ab(n, .9)
    itier = int(ceil(n)/2)
    offset_start = 180/(2*n)
    frac_arr = frac([1], [1])
    frac_list = []
    for i in range(itier):
        cur_fraca = frac([wc**2], [1, (a*np.cos(np.deg2rad(offset_start + i*(180/n))) + 1j*b*np.sin(np.deg2rad(offset_start + i*(180/n))))])
        cur_fracb = frac([1], [1, (a*np.cos(np.deg2rad(offset_start + i*(180/n))) - 1j*b*np.sin(np.deg2rad(offset_start + i*(180/n))))])
        cur_frac = convolve_equations(cur_fraca, cur_fracb)
        cur_frac.den[1] *= wc
        cur_frac.den[2] *= wc**2
        cur_frac.apply_function(lambda x : np.real(x))
        frac_arr = convolve_equations(frac_arr, cur_frac)
        frac_list.append(cur_frac)
        print(cur_frac.print())

    if ceil(n)%2 == 1:
        cur_frac = frac([0, wc], [1, wc])
        filt = convolve_equations(filt, cur_frac)
        frac_list.append(cur_frac)
        print(cur_frac.print())

    if calulate_K:
        K = (frac_arr.den[-1] * hp)/wc**n
        frac_arr.set_k(float(K))
    if not return_list:
        return cur_frac
    else:
        return frac_list

def get_angles(n, a=1, b=1, Debug=False):
    offset_start = 180/(2*n)
    angles = []
    for i in range(n):
        angles.append((offset_start + i*(180/n,(a*np.cos(np.deg2rad(offset_start + i*(180/n))) + 1j*b*np.sin(np.deg2rad(offset_start + i*(180/n)))))))
        if Debug:
            print(f"angle = {angles[i][0]} value = {angles[i][1]}")

if __name__ == "__main__":
    print(bandpass_filter(8, wc=500, passband=100, Debug=True).print())