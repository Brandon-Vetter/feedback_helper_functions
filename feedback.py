import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf
from  transfer import *
"""
@Author Brandon Vetter
@Date September 30th, 2024

This file has some helper functions used for feedback systems.

This was made for the ECE 450 - signals and systems 2 class
for the Unversity of Idaho.
"""



class wz_gt_wp(Exception):
    """
    Execption if wz is > wp
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return(repr(self.value))
    
    
class wp_gt_wz(Exception):
    """
    Execption if wz is < wp
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return(repr(self.value))



def calulate_alpha_db(sigm):
    alf = (1+ np.sin(np.deg2rad(sigm)))/(1-np.sin(np.deg2rad(sigm)))
    return -10*np.log(alf) 

def phase_lead_compensation(wm, sigm):
    alf = (1+ np.sin(np.deg2rad(sigm)))/(1-np.sin(np.deg2rad(sigm)))
    wp = np.sqrt(alf)*wm
    wz = wm/np.sqrt(alf)
    if wz >= wp:
        raise wz_gt_wp("wz can not be greater then wp for a phase lead filter")
    lead_filter = frac([1, float(wz)], [1, float(wp)], k=float(wp/wz))
    print(f"phase wp: {wp}")
    print(f"phase wz: {wz}")
    print(f"phase alf: {alf}")
    return lead_filter

def phase_lag_compensation(wz, wp):
    if wp >= wz:
        raise wp_gt_wz("wp can not be greater then wz for a phase lag filter")
    lead_filter = frac([1, float(wz)], [1, float(wp)])
    print(f"phase wp: {wp}")
    print(f"phase wz: {wz}")
    return lead_filter

def make_errors(Hs, error_start = 0, error_end = 1, error_dt = .01):
    """
    takes Hs, error_start, error_end, and error_dt as input
    returns (step, y1, errS), (ramp, y2, errR), (parabola, y3, errP), TT
    """
    TT = np.arange(error_start, error_end+error_dt, error_dt)
    NN = len(TT)
    step = np.zeros(NN)
    ramp = np.zeros(NN)
    parabola = np.zeros(NN)
    errS = np.zeros(NN)
    errR = np.zeros(NN)
    errP = np.zeros(NN)

    for i in range(NN):
        step[i] = 1.0
        ramp[i] = (error_dt*i)
        parabola[i] = (error_dt*i)**(2)

    denCL = np.add(Hs.num, Hs.den)

    t1, y1, x1 = sig.lsim((Hs.num, denCL), step, TT)
    t2, y2, x2 = sig.lsim((Hs.num, denCL), ramp, TT)
    t3, y3, x3 = sig.lsim((Hs.num, denCL), parabola, TT)

    for i in range(NN):
        errS[i] = step[i] - y1[i]
        errR[i] = ramp[i] - y2[i]
        errP[i] = parabola[i] - y3[i]
    
    return TT, (step, y1, errS), (ramp, y2, errR), (parabola, y3, errP)

def graph_bode(w, Hmag, Hphase, disable_text = False, save=False, save_bode="bode.png", bode_start = 1, bode_end = 1E3, bode_y_min = -60, bode_y_max = 20, bode_x_ticks = None, phase_y_min = -180, phase_y_max = 180, phase_bode_y_ticks = None, wp=None, pm=None, size=(10,5)):
    plt.figure(figsize=size)
    plt.subplot(211)
    plt.semilogx(w, Hmag, 'k')
    plt.ylim(bode_y_min,  bode_y_max)
    plt.xlim(bode_start, bode_end)
    if bode_x_ticks != None:
        plt.xticks(bode_x_ticks)
    plt.ylabel('|H| dB', size=12)
    if not disable_text and wp != None:
        plt.text(0.3 + bode_start, -20, '$\\omega$p = {}'.format(round(wp, 1)), fontsize=12)
    plt.title('Bode Comp')
    plt.grid(which='both')

    for n in range(100):
        if Hphase[n] > 0:
            Hphase[n] = Hphase[n] - 360

    plt.subplot(212)
    plt.semilogx(w, Hphase, 'k')
    plt.ylim(phase_y_min, phase_y_max)
    plt.xlim(bode_start, bode_end)
    plt.yticks(phase_bode_y_ticks)
    plt.ylabel('$\\angle$ H (degs)', size=12)
    plt.xlabel('$\\omega$ (rad/s)')
    if not disable_text and pm != None:
        plt.text(0.3 + bode_start, -150, 'pm = {}'.format(round(pm, 0)), fontsize=12)
    plt.grid(which='both')
    if save:
        plt.savefig(save_bode)
    plt.show()

def graph_error(TT, eqt, y1, error_eqt, save=False, error_name = "error", save_name = "graph.png", axis = None, error_axis = None, xticks = None, yticks = None, error_xticks = None, error_yticks = None, size=(8,3)):
        plt.figure(figsize=size)
        plt.subplot(121)
        plt.plot(TT, y1, 'k--', label='y1(t)')
        plt.plot(TT, eqt, 'k', label='u(t)')
        if axis != None:
            plt.axis(axis)
        if xticks != None:
            plt.xticks(xticks)
        if yticks != None:
            plt.yticks(yticks)
        plt.ylabel(error_name)
        plt.xlabel('t (sec)')
        plt.legend()
        plt.grid()

        plt.subplot(122)
        plt.plot(TT, error_eqt, 'k', label='error')
        plt.legend()
        if error_axis != None:
            plt.axis(error_axis)
        if error_xticks != None:
            plt.xticks(error_xticks)
        if error_yticks != None:
            plt.yticks(error_yticks)
        plt.grid()
        if save:
            plt.savefig(save_name)
        plt.show()

def ds_bode(Hs, save=True, disable_text = False, disable_bode = False, disable_ramp = False, disable_parab = False, disable_step = False, **kargs):
    bode_start =1
    bode_end = 1E3
    bode_step = .1
    bode_y_max = 20
    bode_y_min = -60
    phase_y_max = 180
    phase_y_min = -180
    error_dt = 0.002
    bode_x_ticks = None
    phase_bode_y_ticks = [-270, -180, -90, 0]
    error_start = 0
    error_end = 5
    step_axis = None
    ramp_axis = None
    parab_axis = None
    error_step_axis = None
    error_ramp_axis = None
    error_parab_axis = None
    step_xticks = None
    ramp_xticks = None
    parab_xticks = None
    error_step_xticks = None
    error_ramp_xticks = None
    error_parab_xticks = None
    step_yticks = None
    ramp_yticks = None
    parab_yticks = None
    error_step_yticks = None
    error_ramp_yticks = None
    error_parab_yticks = None
    step_graph_size = (10,2)
    ramp_graph_size = (10,2)
    parab_graph_size = (10,2)
    bode_graph_size = (10,5)
    save_bode = 'bode.png'
    save_step = 'step.png'
    save_ramp = 'ramp.png'
    save_parab = 'parab.png'
    
    for key,arg in kargs.items():
        if 'bode_start' == key:
            bode_start = arg
        if 'bode_end' == key:
            bode_end = arg
        if 'bode_step' == key:
            bode_step = arg
        if 'bode_y_max' == key:
            bode_y_max = arg
        if 'bode_y_min' == key:
            bode_y_min = arg
        if 'phase_y_max' == key:
            phase_y_max = arg
        if 'phase_y_min' == key:
            phase_y_min = arg
        if 'error_dt' == key:
            error_dt = arg
        if 'bode_x_ticks' == key:
            bode_x_ticks = arg
        if 'phase_bode_y_ticks' == key:
            phase_bode_y_ticks = arg
        if 'error_start' == key:
            error_start = arg
        if 'error_end' == key:
            error_end = arg
        if 'save_bode' == key:
            save_bode = arg
        if 'save_step' == key:
            save_step = arg
        if 'save_ramp' == key:
            save_ramp = arg
        if 'save_parab' == key:
            save_parab = arg
        if 'step_axis' == key:
            step_axis = arg
        if 'ramp_axis' == key:
            ramp_axis = arg
        if 'parab_axis' == key:
            parab_axis = arg
        if 'error_step_axis' == key:
            error_step_axis = arg
        if 'error_ramp_axis' == key:
            error_ramp_axis = arg
        if 'error_parab_axis' == key:
            error_parab_axis = arg
        if 'step_xticks' == key:
            step_xticks = arg
        if 'ramp_xticks' == key:
            ramp_xticks = arg
        if 'parab_xticks' == key:
            parab_xticks = arg
        if 'error_step_xticks' == key:
            error_step_xticks = arg
        if 'error_ramp_xticks' == key:
            error_ramp_xticks = arg
        if 'error_parab_xticks' == key:
            error_parab_xticks = arg
        if 'step_yticks' == key:
            step_yticks = arg
        if 'ramp_yticks' == key:
            ramp_yticks = arg
        if 'parab_yticks' == key:
            parab_yticks = arg
        if 'error_step_yticks' == key:
            error_step_yticks = arg
        if 'error_ramp_yticks' == key:
            error_ramp_yticks = arg
        if 'error_parab_yticks' == key:
            error_parab_yticks = arg
        if 'step_graph_size' == key:
            step_graph_size = arg
        if 'ramp_graph_size' == key:
            ramp_graph_size = arg
        if 'parab_graph_size' == key:
            parab_graph_size = arg
        if 'bode_graph_size' == key:
            bode_graph_size = arg

    
    w, Hmag, Hphase = make_bode(Hs, bode_start, bode_end, bode_step)
    if not disable_text:
        gm, pm, wg, wp = margin(Hmag, Hphase, w)
        print(f"gm: {gm}")
        print(f"wg: {wg}")
    # wp freq for phase margin at gain crossover (gain = 1)
    # pm phase maring

    if not disable_bode:
        graph_bode(w, Hmag, Hphase, disable_text, save,
                    save_bode, bode_start, bode_end,
                    bode_y_min, bode_y_max, bode_x_ticks,
                    phase_y_min, phase_y_max, phase_bode_y_ticks, wp, pm, size = bode_graph_size)

    # =============================================================================
    # ---- TIME PORTION ----
    # =============================================================================

    TT, (step, y1, errS), (ramp, y2, errR), (parabola, y3, errP) = make_errors(Hs, error_start, error_end, error_dt)

    if not disable_step:
        graph_error(TT,step, y1, errS, save=save, error_name = "step",
                    save_name=save_step, axis=step_axis, error_axis=error_step_axis,
                    yticks=step_yticks, xticks=step_xticks, error_xticks=error_step_xticks,
                    error_yticks=error_step_yticks, size=step_graph_size)

    if not disable_ramp:
        graph_error(TT,ramp, y2, errR, save=save, error_name = "ramp",
                    save_name=save_ramp, axis=ramp_axis, error_axis=error_ramp_axis,
                    yticks=ramp_yticks, xticks=ramp_xticks, error_xticks=error_ramp_xticks,
                    error_yticks=error_ramp_yticks, size=ramp_graph_size)

    if not disable_parab:
        graph_error(TT,parabola, y3, errP, save=save, error_name = "parabola",
                    save_name=save_parab, axis=parab_axis, error_axis=error_parab_axis,
                    yticks=parab_yticks, xticks=parab_xticks, error_xticks=error_parab_xticks,
                    error_yticks=error_parab_yticks, size=parab_graph_size)

    

if __name__ == '__main__':
    test_frac = phase_lead_compensation(8.4, 30)
    print(test_frac.print())
