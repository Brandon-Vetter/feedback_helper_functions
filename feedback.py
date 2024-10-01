import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt
from control import margin
from control import tf
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
          
    def print(self):
        num_str = ""
        den_str = ""
        for i in range(len(self.num)):
            if self.num[i] != 0:
                if (len(self.num)-1) - i == 1:
                    num_str += f"{self.num[i]}s + " 
                elif (len(self.num)-1) - i != 0:
                    num_str += f"{self.num[i]}s^{len(self.num) - i + 1} + "
                else:
                    num_str += f"{self.num[i]} + "
            if self.den[i] != 0:
                if (len(self.num)-1) - i == 1:
                    den_str += f"{self.den[i]}s + "
                elif (len(self.num)-1) - i != 0:
                    den_str += f"{self.den[i]}s^{len(self.num) - i + 1} + "
                    
                else:
                    den_str += f"{self.den[i]} + "
        den_str = den_str[:-3]
        num_str = num_str[:-3]
        return f"$\\frac{{{num_str}}}{{{den_str}}}$ k = {self.k}"
        

def convolve_equations(*args):
    arg_list = list(args)
    arg_total = arg_list.pop(0)
    for arg in arg_list:
        arg_total.num = np.convolve(arg.num, arg_total.num)
        arg_total.den = np.convolve(arg.den, arg_total.den)
    
    return arg_total 

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
    return lead_filter

def phase_lag_compensation(wm, sigm):
    alf = (1+ np.sin(np.deg2rad(sigm)))/(1-np.sin(np.deg2rad(sigm)))
    wp = np.sqrt(alf)*wm
    wz = wm/np.sqrt(alf)
    if wp >= wp:
        raise wp_gt_wz("wp can not be greater then wp for a phase lag filter")
    lead_filter = frac([1, float(wz)], [1, float(wp)])
    return lead_filter
    
def ds_bode(Hs, save=True, disable_text = False, **kargs):
    bode_start =1
    bode_end = 100
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
    
    w = np.arange(bode_start, bode_end+bode_step, bode_step)
    system = sig.lti(Hs.num, Hs.den)
    w, Hmag, Hphase = sig.bode(system, w)
    if not disable_text:
        gm, pm, wg, wp = margin(Hmag, Hphase, w)
    # wp freq for phase margin at gain crossover (gain = 1)
    # pm phase maring

    plt.subplot(211)
    plt.semilogx(w, Hmag, 'k')
    plt.ylim(bode_y_min,  bode_y_max)
    plt.xlim(bode_start, bode_end)
    if bode_x_ticks != None:
        plt.xticks(bode_x_ticks)
    plt.ylabel('|H| dB', size=12)
    if not disable_text:
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
    plt.xlabel('$\\omega$ (rad/s)')
    if not disable_text:
        plt.text(0.3 + bode_start, -150, 'pm = {}'.format(round(pm, 0)), fontsize=12)
    plt.grid(which='both')
    if save:
        plt.savefig(save_bode)
    plt.show()

    # =============================================================================
    # ---- TIME PORTION ----
    # =============================================================================


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

    plt.subplot(321)
    plt.plot(TT, y1, 'k--', label='y1(t)')
    plt.plot(TT, step, 'k', label='u(t)')
    if step_axis != None:
        plt.axis(step_axis)
    plt.ylabel('step')
    plt.xlabel('t (sec)')
    plt.legend()
    plt.grid()

    plt.subplot(322)
    plt.plot(TT, errS, 'k', label='error')
    plt.legend()
    if error_step_axis != None:
        plt.axis(error_step_axis)
    plt.grid()
    if save:
        plt.savefig(save_step)
    plt.show()

    plt.subplot(321)
    plt.plot(TT, y2, 'k--', label='y2(t)')
    plt.plot(TT, ramp, 'k', label='r(t)')
    if ramp_axis != None:
        plt.axis(ramp_axis)
    plt.ylabel('ramp')
    plt.xlabel('t (sec)')
    plt.legend()
    plt.grid()

    plt.subplot(322)
    plt.plot(TT, errR, 'k', label='error')
    plt.legend(loc=4)
    plt.xlabel('t (sec)')
    if error_ramp_axis != None:
        plt.axis(error_ramp_axis)
 #   plt.yticks([-0.5, 0, 0.3, 0.5])
    plt.grid()
    if save:
        plt.savefig(save_ramp)
    plt.show()

    plt.subplot(321)
    plt.plot(TT, y3, 'k--', label='y3(t)')
    plt.plot(TT, parabola, 'k', label='parab(t)')
    if parab_axis != None:
        plt.axis(parab_axis)
    plt.ylabel('parabola')
    plt.xlabel('t (sec)')
    plt.legend()
    plt.grid()

    plt.subplot(322)
    plt.plot(TT, errP, 'k', label='error')
    plt.xlabel('t (sec)')
    plt.legend()
    if error_parab_axis != None:
        plt.axis(error_parab_axis)
    plt.grid()
    if save:
        plt.savefig(save_parab)
    plt.show()
    

if __name__ == '__main__':
    test_frac = phase_lead_compensation(8.4, 30)
    print(test_frac.print())
    find_good_phase_values(50, test_frac)
