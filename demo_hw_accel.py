#Author: Tan H. Nguyen and Enyu Luo
#The gold of this file is removing all stuffs relating to classes, simply all unecessary stuffs
# so that we can easily see what part can be ported into hardware


import numpy as np
import scipy as sp
import scipy.io as sio
import math
import scipy.stats as sps
import string #Needed for writting/reading files

from matplotlib.pyplot import *

print "----Demo code for ECE 527----"
#----Global setting parameters----
#---GPS CONSTANTS----
#These constants are obtaind from libgnss/constants.py
mu = 3.986005e14           # earth's universal gravitation parameter
F = -4.442807633e-10       # relativistic correction term
OEDot = 7.2921151467e-5    # earth's sidereal rotation rate
c = 299792458.0            # speed of light in meters per second
pi = 3.1415926535898       # pi
radToDeg = 180.0/pi
degToRad = pi/180.0
L1 = 1.57542e9             # the GPS L1 frequency
L2 = 1.22760e9             # the GPS L2 frequency
ds = 1.0   #-1.0       # -ve for flipped doppler (high-LO), +ve for regular doppler (low-LO) - see _init_ in settings.py
f = L1 		# nominal carrier frequency in Hz
fc0 = 1.023e6 #Code frequency in Hz (defined in settings.py)
Lc = 1023.0 #Number of chips in C/A code (have to be float) (defined in settings.py)
fs = 2.0e6  #5.456e6 
fi0 = 0.0    #1.3364e6 - Defined in settings class
Tc = 0.001    # period of a C/A code (Tc = Lc/fc), also basic tracking block --See definition in class instantiation in settings.py

fc = fc0 #fc is a variables that is defined for each channel, not necessary fc0
fi = fi0 
fi_bias = 0
fc_bias = 0
_fi = fi + fi_bias
_fc = fc + fc_bias

# INITIALIZE THE GLOBAL TRACKING SETTINGS ------------------------------------
t_elspacing = 0.5  # early/late spacing from prompt in code chips
t_filterIntegrator = 'BILINEAR' # 'BILINEAR' or 'BOXCAR'

#loopfilter constants
N = 1   #also used for discriminator
T = Tc*N

# dll loop filter N=1, order=2, Bnp = 3.0, Bnf = 0.0, integrator = 'BILINEAR'
dll_Bnp = 3.0
dll_w0p = dll_Bnp/0.53
dll_Kvp = dll_w0p**2.0
dll_Kpp = 1.414*dll_w0p
dll_Kvf = dll_Bnp/0.25
dll_Bnf = 0.0

# pll loop filter N=1, order=2, Bnp = 40.0, Bnf = 0.0, integrator = 'BILINEAR'
pll_Bnp = 40.0
pll_w0p = pll_Bnp/0.53
pll_Kvp = pll_w0p**2.0
pll_Kpp = 1.414*pll_w0p
pll_Kvf = pll_Bnp/0.25
pll_Bnf = 0.0

# lock detector
lockd_k = 1.5
lockd_lossthreshold = 50
lockd_lockthreshold = 240
lockd_N = 20
#lock detector low pass filter k value
lockd_lpf_k = 0.0247


# fileobject to raw samples
fileobject = None
#initialize empty dictionary to store multiple channel variables
rx_channel = {}
#-----------------------------------------------------------------

# One global file object for the whole code
class Channel():
    __slots__ = ['prn', 'fi_bias', 'fc_bias', '_fi', '_fc', '_mscount', '_cpcount', \
                '_mscount0', '_cpcount0', 'chips', 'e_a', 'p_a', 'l_a', \
                'dpi', 'dpc', 'dfi', 'dfc', 'di', 'dc', 'efi', 'efc', \
                'iE', 'iP', 'iL', 'qE', 'qP', 'qL', 'rc', 'ri', \
                'pll_h', 'dll_h', 'losscount', 'lockcount', 'lock', \
                'lockd_iFilter_h', 'lockd_qFilter_h']
    
    def __init__(self,  prn):#Use as a replacement for overload constructor
        self.prn  = prn
        #self._DATA_LOG_ON = False
        #self.set_block_size(msblock=datalog_msblock, buff=datalog_buffer)

        # Initialize helper variables.
        global fi, fc
        self.fi_bias = 0 # carrier NCO frequency bias, will be determined during acquisition
        self.fc_bias = 0 # code    NCO frequency bias, will be determined during acquisition
        self._fi = fi+self.fi_bias  # internal carrier frequency tracker
        self._fc = fc+self.fc_bias  # internal code    frequency tracker
        #something to think about: might wanna make mscount global if we are
        #processing all channels at the same time for each ms
        self._mscount = 0 # internal millisecond counter
        self._cpcount = 0 # internal codeperiod counter
        self._mscount0 = 0 # internal millisecond counter, start value
        self._cpcount0 = 0 # internal codeperiod counter, start value
        
        #for correlator
        self.chips = _make_gps_code_chips(self.prn)
        self.e_a = 0
        self.p_a = 0
        self.l_a = 0
        
        self.rc  = None # code phase at the zero sample
        self.ri  = None # (intermediate) carrier phase at the zero sample
        
        self.losscount = 0
        self.lockcount = 0
        self.lock = 0
        self.lockd_iFilter_h = 0
        self.lockd_qFilter_h = 0

        #for loop filter
        self.pll_h = 0
        self.dll_h = 0
        
	
    def update_found_params(self,settings):
    #This method update found paramsters from the searching results
	self.settings = settings
	self.fi_bias = self.settings['fi_bias']
	self.fc_bias = self.settings['fc_bias']
	self._fi = self.settings['_fi']
	self._fc = self.settings['_fc']
	self.rc = self.settings['rc']
	self.ri = self.settings['ri']
		
    def update(self):
   #the following function is from channel.update() of the channel class
	"""
	Perform a 1 ms update of the tracking channel
	This consist of 
	1) Computing the signal correlations
	2) Producing code and carrier frequency update	
	"""
	ms = self._mscount
	cp = self._cpcount
	#Record the number of completed code periods by the start of the correlation window
	self.cp[ms] = cp

	"""	
	Outputs 
	rc: zero sample code phase
	ri: zero sample carrier phase
	c_r: receiver synchronous correlation (at S samples)
	c_s1: signal synchronous correlation (at xs1 samples)
	c_s2: second synchronous correlation (at xs2 samples)
	Note that:
		(1): c_s1 and c_s2 may be both None-> no code signal boundary, code signal extends entire block
		(2): only c_s2 may be None, and -> one code signal boundary
		(3): neither c_s1 nor c_s2 may be none -> two code signal boundary
	This is due to different signal boundaries between scenarios.
	"""

def _get_1ms_correlations(prn,ms):
    """
    Helper function that returns the correlations
    """
    global rx_channel
    iE = rx_channel[prn].iE[ms]
    iP = rx_channel[prn].channel.iP[ms]
    iL = rx_channel[prn].channel.iL[ms]
    qE = rx_channel[prn].channel.qE[ms]
    qP = rx_channel[prn].channel.qP[ms]
    qL = rx_channel[prn].channel.qL[ms]

    return iE, iP, iL, qE, qP, qL

def _pll_update(prn,ms):
    """

    Helper function that returns the carrier discriminator output

    Note: Script assumes that N will not be set  > n unless ms > n - 1 
    Note: Script assumes that N will only be set > 1 when
          there is a good lock (value of iP is accurate)

    @rtype : tuple
    @return: (xpn, xfn) where xpn and xfn are the discriminator 
             phase and frequency outputs respectively
    """

    xp = xf = 0.0

    iEc, iPc, iLc, qEc, qPc, qLc = _get_1ms_correlations(prn,ms)

    # only copied the code related to PLL  
    iP = np.sum(iPc)
    qP = np.sum(qPc)

    if  (iP!=0) :
        xp = np.arctan(qP/iP) / (2.0*pi)
    else:
        print("Warning: very low correlation values")

    return xp, xf

def _dll_update(prn,ms):
    """
    Helper function that returns the code discriminator output

    Note: Script assumes that N will not be set  > n unless ms > n - 1 
    Note: Script assumes that N will only be set > 1 when
          there is a good lock (value of iP is accurate)

    @rtype : tuple
    @return: (xpn, xfn) where xpn and xfn are the discriminator 
             phase and frequency outputs respectively
    """
    xp = xf = 0.0
    iEc, iPc, iLc, qEc, qPc, qLc = _get_1ms_correlations(prn,ms)
    iE = np.sum(iEc)
    qE = np.sum(qEc)
    iL = np.sum(iLc)
    qL = np.sum(qLc)
    # Use the Normalized Early Minus Late Envelope discriminator.
    E = np.sqrt(iE**2.0 + qE**2.0)
    L = np.sqrt(iL**2.0 + qL**2.0)
    if ((E+L)!=0) :
        xp = (E - L) / (2.0*(E + L))
    else :
        print("Warning: very low correlation values")
    return xp, xf

def dll_BilinearIntegrator_update(prn, xn):
    # A(n-1) and A(n) are represented here by h0 and self.h, respectively.
    # The time between samples (often denoted T) is represented as self.k.
    global rx_channel, T
    h0 = rx_channel[prn].dll_h
    rx_channel[prn].dll_h = rx_channel[prn].dll_h + T*xn
    return (rx_channel[prn].dll_h + h0) / 2.0

def pll_BilinearIntegrator_update(prn, xn):
    # A(n-1) and A(n) are represented here by h0 and self.h, respectively.
    # The time between samples (often denoted T) is represented as self.k.
    global rx_channel, T
    h0 = rx_channel[prn].pll_h
    rx_channel[prn].pll_h = rx_channel[prn].pll_h + T*xn
    return (rx_channel[prn].pll_h + h0) / 2.0


# dll loop filter N=1, order=2, Bnp = 3.0, Bnf = 0.0, integrator = 'BILINEAR'
def dll_loopfilter_update(prn, xp = 0, xf = 0):
    global dll_Kvp, dll_Kvf, dll_Kpp
    yn = dll_BilinearIntegrator_update(prn, dll_Kvp+xf*dll_Kvf) + xp*dll_Kpp
    return yn

# pll loop filter N=1, order=2, Bnp = 40.0, Bnf = 0.0, integrator = 'BILINEAR'
def pll_loopfilter_update(prn, xp = 0, xf = 0):
    global pll_Kvp, pll_Kvf, pll_Kpp
    yn = pll_BilinearIntegrator_update(prn, pll_Kvp+xf*pll_Kvf) + xp*pll_Kpp
    return yn


def scalar_track(prn):

    global rx_channel, fi, fc, fcaid

    ms = rx_channel[prn]._mscount-1

    # Get the current carrier and code phase
    # self.correlator.ri = self.correlator.ri + self._fi*self.settings.Tc
    # self.correlator.rc = self.correlator.rc + self._fc*self.settings.Tc

    # TRACK using correlation output, get discriminator outputs
    # Output of Discriminator: normalized discriminator
    # dpi, dfi: carrier phase and frequency error
    # dpc, dfc: code    phase and frequency error

    #need to convert discriminator code
    dpi, dfi = _pll_update(prn,ms) # carrier and
    dpc, dfc = _dll_update(prn,ms) # code update

    # save current discriminator error and NCO frequency bias error values
    rx_channel[prn].dpi[ms], rx_channel[prn].dpc[ms], rx_channel[prn].dfi[ms], \
        rx_channel[prn].dfc[ms] = dpi, dpc, dfi, dfc

    # Get filtered NCO corrections
    rx_channel[prn].di[ms] = di = pll_loopfilter_update(prn=prn, xp = dpi, xf = dfi)
    rx_channel[prn].dc[ms] = dc = dll_loopfilter_update(prn=prn, xp = dpc, xf = dfc)

    # TRACK using code and carrier frequency.
    rx_channel[prn].efi[ms] =  (fi + rx_channel[prn].fi_bias + di) - rx_channel[prn]._fi
    rx_channel[prn].efc[ms] = ((fc + rx_channel[prn].fc_bias + dc) + fcaid*(rx_channel[prn].fi_bias+di)) - rx_channel[prn]._fc

    # Corrects carrier and code frequency based on frequency errors
    rx_channel[prn]._fi = rx_channel[prn]._fi+ rx_channel[prn].efi[ms]
    rx_channel[prn]._fc = rx_channel[prn]._fc+ rx_channel[prn].efc[ms]

def correlate(prn, fi, fc):
    """
    Correlate one millisecond of raw data against the replica signal.
    The replica signal is produced based on the input code and carrier frequencies
    (and internally saved code and carrier phase offsets).

    @type  fi: float
    @param fi: The current (intermediate) carrier frequency in cycles per second.
    @type  fc: float
    @param fc: The current code frequency in chips per second.
    @rtype: tuple
    @return: (self.rc, self.ri, c_r, c_s1, c_s2)
        self.rc : zero sample code phase,            
        self.ri : zero sample carrier phase,            
        c_r  : receiver synchronous correlations (from 0:S samples),
        c_s1 : signal synchronous correlations (from prev:xs1 samples),
        c_s2 : signal synchronous correlations (from xs1:xs2 samples).
    """

    # note on the variables that are used in this section of the code
    # idc: unpermuted sample index
    # ri : intermediate carrier phase (cycles)
    # rc : code phase (chips)
    # Lc : ideal number of code chips per window (1023.0 chips)
    # offset: 0.5 chips, such that the different between E and L is 1 chip
    # chips : 1023 code chips from the function _make_gps_code_chips()

    # Read in the next set of samples and convert to baseband.
    # samples are complex signal amplitude values

    # initialize commonly used parameters
    global rx_channel, fileobject, datatype, datacomplex, Tc, Lc, t_elspacing, fs, S, idc
    offset = t_elspacing

    rc = rx_channel[prn].rc
    ri = rx_channel[prn].ri        

    raw = np.fromfile(fileobject, datatype, S)
    if datacomplex:
        raw = raw['i']+1j*raw['q']

    trigarg = ((2.0*pi*fi)*(idc/fs)) + (2.0*pi*ri) 
    baseband = np.exp(-1j*trigarg)*raw

    print "raw['i'].datatype: ", raw['i'].dtype
    """
    baseband_f = np.fft.fft(baseband,self.settings.S)
    fftbins    = np.fft.fftfreq(n = self.settings.S, d = 1.0/self.settings.fs)
    baseband_f[np.where((fftbins<-self.settings.fc))] = 0
    baseband_f[np.where((fftbins>self.settings.fc))]  = 0
    baseband   = np.fft.ifft(baseband_f, self.settings.S)
    """

    # fi is f_IF + f_doppler
    #   the remaining carrier signal after carrier wipeoff is:
    #     np.exp(1j*(2*pi*df*t+dri))
    #   an update can then be performed after discrimination as:
    #     dri = arctan(qP/iP)
    #     ri = ri + dri = ri + fi*dt

    # Compute the expected code boundary locations (xs1, xs2) in samples.
    # fc is included in here to address the code doppler
    xs1 = (Lc - rc) * (fs/fc)
    xs2 = xs1 + (Lc * (fs/fc))

    # Produce the code replica indices.
    # fidc is the prompt replica's indices 
    # because it is shifted by xs1 to match the received signal
    fidc = (idc - xs1) * (fc/fs) # float code indices by sample
    #pictorial representation in page 177 of Kaplan and Hegarty
    #data   0 1 2 3 4 5 6 7 fidc
    #early  1 2 3 4 5 6 7 0 fidc + offset
    #prompt 0 1 2 3 4 5 6 7 fidc
    #late   7 0 1 2 3 4 5 6 fidc - offset
    eidc = np.mod(np.floor(fidc + offset),Lc).astype(np.int16) # early indices
    pidc = np.mod(np.floor(fidc         ),Lc).astype(np.int16) # prompt indices
    lidc = np.mod(np.floor(fidc - offset),Lc).astype(np.int16) # late indices

    # Produce the shifted replica vectors. 
    # numpy.take returns array values from the given indexes
    early  = rx_channel[prn].chips.take(eidc)
    prompt = rx_channel[prn].chips.take(pidc)
    late   = rx_channel[prn].chips.take(lidc)

    # initialize for the next iteration
    rx_channel[prn].rc = (rx_channel[prn].rc +fc*Tc)% Lc
    rx_channel[prn].ri =  rx_channel[prn].ri +fi*Tc

    # initialize the sample boundaries 
    # (+1 is added to include the sample during array indexing)
    idxs1 = np.mod(np.floor(xs1),S).astype(np.int16) + 1 

    # The normal case: the first boundary is within this window, and the
    # second boundary is outside of this window.
    if  xs1 <= S < xs2:
        # The flow of this case is as follows:
        # First correlate part B and form signal   synchronous outputs.
        # Next  correlate part A and form receiver synchronous outputs.
        # |bbb|aaaaaa| |bbb|aaaaaa|

        # Segment the baseband vector into B and A parts.
        baseband_b = baseband[:idxs1]
        baseband_a = baseband[idxs1:]

        # Get the correlations for segments B and A. 
        e_b = np.inner(baseband_b, early[:idxs1])  # early  part B
        p_b = np.inner(baseband_b, prompt[:idxs1]) # prompt part B
        l_b = np.inner(baseband_b, late[:idxs1])   # late   part B
        e_a = np.inner(baseband_a, early[idxs1:])  # early  part A
        p_a = np.inner(baseband_a, prompt[idxs1:]) # prompt part A
        l_a = np.inner(baseband_a, late[idxs1:])   # late   part A

        # Prepare the signal synchronous correlation outputs.
        e_s = rx_channel[prn].e_a + e_b # early  signal synchronous
        p_s = rx_channel[prn].p_a + p_b # prompt signal synchronous
        l_s = rx_channel[prn].l_a + l_b # late   signal synchronous

        # Record the part A correlations for use in the next update.
        rx_channel[prn].e_a, rx_channel[prn].p_a, rx_channel[prn].l_a = e_a, p_a, l_a

        # c_s1 = i-early, q-early, i-prompt, q-prompt, i-late, q-late.
        c_s1 = e_s.real, e_s.imag, p_s.real, p_s.imag, l_s.real, l_s.imag

        # Prepare the receiver synchronous correlation outputs.
        pos = np.abs(e_b + p_b + l_b + e_a + p_a + l_a)
        neg = np.abs(e_b + p_b + l_b - e_a - p_a - l_a)

        e_r, p_r, l_r = 0, 0, 0

        if pos > neg: # there was no polarity change from B to A
            e_r = e_b + e_a # early  receiver synchronous
            p_r = p_b + p_a # prompt receiver synchronous
            l_r = l_b + l_a # late   receiver synchronous
        else: # there was a polarity change from B to A
            e_r = e_b - e_a # early  receiver synchronous
            p_r = p_b - p_a # prompt receiver synchronous
            l_r = l_b - l_a # late   receiver synchronous

        # c_r = i-early, q-early, i-prompt, q-prompt, i-late, q-late.
        c_r = e_r.real, e_r.imag, p_r.real, p_r.imag, l_r.real, l_r.imag

        # Returns:
        # rc   : zero sample code phase (for the start of this correlation window),
        # c_r  : receiver synchronous correlations (at S samples),
        # c_s1 : signal synchronous correlations (at xs1 samples),
        # None : second signal synchronous correlations (there were none).
        return rc, ri, c_r, c_s1, None

    if  xs1 < xs2 <= S:

        # The flow of this case is as follows:
        # First correlate part B and form signal synchronous outputs.
        # Next correlate an A & B pair and form signal synchronous outputs.
        # Finally correlate part A and form receiver synchronous outputs.

        # Compute the zero sample index of the second code.
        # (+1 is added to include the sample during array indexing)
        idxs2 = np.mod(np.floor(xs2),S).astype(np.int16)+1

        # Segment the baseband vector into B, AB, and A parts.
        baseband_b = baseband[:idxs1]
        baseband_s = baseband[idxs1:idxs2]
        baseband_a = baseband[idxs2:]

        # Get the correlations for the first B segment. 
        e_b = np.inner(baseband_b, early[:idxs1])  # early  part B
        p_b = np.inner(baseband_b, prompt[:idxs1]) # prompt part B
        l_b = np.inner(baseband_b, late[:idxs1])   # late   part B

        # Prepare the signal synchronous correlation outputs.
        e_s = rx_channel[prn].e_a + e_b
        p_s = rx_channel[prn].p_a + p_b
        l_s = rx_channel[prn].l_a + l_b

        # c_s1 = i-early, q-early, i-prompt, q-prompt, i-late, q-late.
        c_s1 = e_s.real, e_s.imag, p_s.real, p_s.imag, l_s.real, l_s.imag

        # Get the correlations for the complete A and B segment pair. 
        e_s = np.inner(baseband_s, early[idxs1:idxs2])  # early  part S
        p_s = np.inner(baseband_s, prompt[idxs1:idxs2]) # prompt part S
        l_s = np.inner(baseband_s, late[idxs1:idxs2])   # late   part S

        # c_s2 = i-early, q-early, i-prompt, q-prompt, i-late, q-late.
        c_s2 = e_s.real, e_s.imag, p_s.real, p_s.imag, l_s.real, l_s.imag                

        # Get the correlations for the final A segment.
        e_a = np.inner(baseband_a, early[idxs2:])  # early  part A
        p_a = np.inner(baseband_a, prompt[idxs2:]) # prompt part A
        l_a = np.inner(baseband_a, late[idxs2:])   # late   part A

        # Record the part A correlations for use in the next update.
        rx_channel[prn].e_a, rx_channel[prn].p_a, rx_channel[prn].l_a = e_a, p_a, l_a

        # Prepare the receiver synchronous correlation outputs.            
        pos = np.abs(e_b + p_b + l_b + e_s + p_s + l_s)
        neg = np.abs(e_b + p_b + l_b - e_s - p_s - l_s)

        e_r, p_r, l_r = 0, 0, 0

        if pos > neg: # there was no polarity change from B to AB

            pos = np.abs(e_s + p_s + l_s + e_a + p_a + l_a)
            neg = np.abs(e_s + p_s + l_s - e_a - p_a - l_a)

            if pos > neg: # there was no polarity change from AB to A
                e_r = e_b + e_s + e_a
                p_r = p_b + p_s + p_a
                l_r = l_b + l_s + l_a
            else: # there was a polarity change from AB to A
                e_r = e_b + e_s - e_a
                p_r = p_b + p_s - p_a
                l_r = l_b + l_s - l_a

        else: # there was a polarity change from B to AB (thus not AB to A)

            e_r = e_b - e_s - e_a
            p_r = p_b - p_s - p_a
            l_r = l_b - l_s - l_a

        # c_r = i-early, q-early, i-prompt, q-prompt, i-late, q-late
        c_r = e_r.real, e_r.imag, p_r.real, p_r.imag, l_r.real, l_r.imag 

        # Returns:
        # rc   : zero sample code phase (for the start of this correlation window)
        # c_r  : receiver synchronous correlations (at S samples),
        # c_s1 : signal syncrhonous correlations (at xs1 samples),
        # c_s2 : signal synchronous correlations (at xs2 samples).
        return rc, ri, c_r, c_s1, c_s2

    if  S < xs1:
        # The flow of this case is as follows:
        # First correlate part B, actually a continuation of A from the
        # last correlation window.  Form the receiver synchronous outputs.

        # Get the correlations for the continuous B segment. 
        e_b = np.inner(baseband, early)
        p_b = np.inner(baseband, prompt)
        l_b = np.inner(baseband, late)

        # Update the part A correlations for use in the next update.
        rx_channel[prn].e_a = rx_channel[prn].e_a + e_b
        rx_channel[prn].p_a = rx_channel[prn].p_a + p_b
        rx_channel[prn].l_a = rx_channel[prn].l_a + l_b

        # Prepare the receiver synchronous correlation outputs.
        # c_r = i-early, q-early, i-prompt, q-prompt, i-late, q-late.
        c_r = e_b.real, e_b.imag, p_b.real, p_b.imag, l_b.real, l_b.imag 

        # Returns:
        # rc   : zero sample code phase (for the start of this correlation window)
        # c_r  : receiver synchronous correlations (at S samples),
        # None : first signal synchronous correlations (there were none),
        # None : second signal synchronous correlations (there were none).
        return rc, ri, c_r, None, None

def lockd_iFilter_update(prn, xn):
    """
    The LowPassFilter, unlike the LoopFilter, is a simple general purpose filter
    for smoothing outputs, such as those used in signal to noise measurements and
    lock detection.  
    
    For more information, see
     - Kaplan and Hegarty, page 234, Fig. 5.43

    Returns the filtered inputs given the most recent input xn.

    @type xn: float
    @param xn: input
    @rtype: float
    @return: Filtered inputs, given most recent input xn.
    """
    # The most recent output y(n-1) is represented here by self.h.
    global rx_channel
    rx_channel[prn].lockd_iFilter_h = lockd_lpf_k*xn + (1-lockd_lpf_k)*rx_channel[prn].lockd_iFilter_h
    return rx_channel[prn].lockd_iFilter_h

def lockd_qFilter_update(prn, xn):
    """
    The LowPassFilter, unlike the LoopFilter, is a simple general purpose filter
    for smoothing outputs, such as those used in signal to noise measurements and
    lock detection.  
    
    For more information, see
     - Kaplan and Hegarty, page 234, Fig. 5.43

    Returns the filtered inputs given the most recent input xn.

    @type xn: float
    @param xn: input
    @rtype: float
    @return: Filtered inputs, given most recent input xn.
    """
    # The most recent output y(n-1) is represented here by self.h.
    global rx_channel
    rx_channel[prn].lockd_qFilter_h = lockd_lpf_k*xn + (1-lockd_lpf_k)*rx_channel[prn].lockd_qFilter_h
    return rx_channel[prn].lockd_qFilter_h

def lockdetector_update(prn, iP, qP):
    """
    Update the lock detector with the latest prompt correlator outputs,
    and determine the locking status.

    @type iP : float
    @param iP : This is the latest inphase prompt correlator output.
    @type qP : float
    @param qP : This is the latest quadraphase prompt correlator output.
    @rtype : tuple
    @return : (int, float), (self.lock, iP-qP)
    This tuple carries locking status (self.lock - either 1 for True or
    0 for False) and the current difference between the filtered
    magnitudes of the (scaled) inphase prompt value and the quadraphase
    prompt value.  The latter can be used as a diagnostic aid when
    reviewing data.
    """

    global rx_channel

    iP = lockd_iFilter_update(prn, iP.__abs__()) / lockd_k
    qP = lockd_qFilter_update(prn, qP.__abs__())

    if iP > qP:
        rx_channel[prn].losscount = 0
        if rx_channel[prn].lockcount > lockd_lockthreshold:
            rx_channel[prn].lock = 1

            # Development Notes: consider implementing false phase lock
            # detection logic here.
        else:
            rx_channel[prn].lockcount = rx_channel[prn].lockcount + 1
    else:
        rx_channel[prn].lockcount = 0
        if rx_channel[prn].losscount > lockd_lossthreshold:
            rx_channel[prn].lock = 0
        else:
            rx_channel[prn].losscount = rx_channel[prn].losscount + 1

    return rx_channel[prn].lock, iP-qP

def channel_update(prn):
    """
    Performs a 1ms update of the tracking channel.  
    This consists of:
     1) computing signal correlations, 
     2) producing code and carrier frequency updates, 
    """

    global rx_channel

#    assert self._DATA_LOG_ON, 'Data logging attributes must be available for update.'

    ms = rx_channel[prn]._mscount
    cp = rx_channel[prn]._cpcount

    #record the number of completed code periods by the start of the correlation window
    rx_channel[prn].cp[ms] = cp 

    # CORRELATE 1ms with the following outputs:
    # rc   : zero sample code phase,
    # ri   : zero sample carrier phase,
    # c_r  : receiver synchronous correlations (at S samples),
    # c_s1 : signal synchronous correlations (at xs1 samples),
    # c_s2 : second signal synchronous correlations (at xs2 samples).
    # Note that (1) c_s1 and c_s2 may both be None,    -> no code signal boundary, code signal extends entire block
    #           (2) only c_s2 may be None, and         -> one code signal boundary
    #           (3) neither c_s1 nor c_s2 may be None. -> two code signal boundaries
    # This is due to different signal boundaries scenarios.
    rc, ri, c_r, c_s1, c_s2 = correlate(prn, rx_channel[prn]._fi, rx_channel[prn]._fc) #current code and carrier frequency
    """
    print "Output of the correlation calculation: "
    print "rc: ", rc
    print "ri: ", ri
    print "c_r: ", c_r
    print "c_s1: ", c_s1
    print "c_s2: ", c_s2
    """


    rx_channel[prn].rc[ms] = rc       # records the current zero-sample code phase, at the start of this correlation window
    rx_channel[prn].ri[ms] = ri       # records the current zero-sample carrier phase, at the start of this correlation window
    rx_channel[prn].fi[ms] = rx_channel[prn]._fi # records the current intermediate frequency fi
    rx_channel[prn].fc[ms] = rx_channel[prn]._fc # records the current code frequency fc
    rx_channel[prn].iE[ms], rx_channel[prn].iP[ms], rx_channel[prn].iL[ms] = c_r[0], c_r[2], c_r[4] #saved using receiver synchronous correlations
    rx_channel[prn].qE[ms], rx_channel[prn].qP[ms], rx_channel[prn].qL[ms] = c_r[1], c_r[3], c_r[5]

    # Calculate the lock and SNR using prompt receiver synchronous correlations.
    lock, lockval = lockdetector_update(prn, c_r[2], c_r[3])
    rx_channel[prn].lock[ms], rx_channel[prn].lockval[ms] = lock, lockval
#    rx_channel[prn].snr[ms] = self.snrmeter.update(c_r[2], c_r[3])

    # because we do 1ms correlations, it is possible for i = 0,1,2 code correlations
    # it is usually i = 1 but we might get code boundaries        
    i = 0 if c_s1 is None else 1     # 0: first code signal boundary not found,  1: first code signal boundary found
    i = i if c_s2 is None else i + 1 # i: second code signal boundary not found, i+1: second code signal boundary found

    if i > 0: 
        # save current signal synchronous correlations
        rx_channel[prn].ss_iE[cp], rx_channel[prn].ss_qE[cp] = c_s1[0], c_s1[1]
        rx_channel[prn].ss_iP[cp], rx_channel[prn].ss_qP[cp] = c_s1[2], c_s1[3]
        rx_channel[prn].ss_iL[cp], rx_channel[prn].ss_qL[cp] = c_s1[4], c_s1[5]
        cp = rx_channel[prn]._cpcount = rx_channel[prn]._cpcount + 1 #increment the number of code segments counter
        if i == 2:
            rx_channel[prn].ss_iE[cp], rx_channel[prn].ss_qE[cp] = c_s2[0], c_s2[1]
            rx_channel[prn].ss_iP[cp], rx_channel[prn].ss_qP[cp] = c_s2[2], c_s2[3]
            rx_channel[prn].ss_iL[cp], rx_channel[prn].ss_qL[cp] = c_s2[4], c_s2[5]
            cp = rx_channel[prn]._cpcount = rx_channel[prn]._cpcount + 1 #increment the number of code segments counter
        if (i != 0) and (i != 1) and (i != 2):
            print('Warning, the number of correlations i is:'+str(i))
 
    # Increment the internal mscount (millisecond counter)
    rx_channel[prn]._mscount = rx_channel[prn]._mscount + 1







#The following setting dictionary define a set of parameters that is used in the settings.update() function. This function is called inside the __init__() constructor of setttings class..
settings = {'filename':'antenna0_2MHz_40dB.dat', #File name for reading the data
'fs':2.0e6, #Sampling frequency
'fi':0.0,
'ds':1.0,
'datatype':np.dtype([('i',np.short),('q',np.short)]),
'skip_seconds':10.0};

def update(_settings):
	"""
	Adapted from settings.update() & settings.init_secondaries()
	(Re) initialize changes to the default settings
	"""
	global fs, filename, fi, ds, datatype, skip_seconds, skip_samples, channelList, S, fcaid, datacomplex, idc

	fs = _settings['fs']
	filename = _settings['filename']
	fi = _settings['fi']
	ds = _settings['ds']
	datatype = _settings['datatype']	
	skip_seconds = _settings['skip_seconds']
	#---The following section of the code is adapted from init_secondaries
	skip_samples = int(round(skip_seconds * fs))
	channelList = range(1,33) #List of the receiver
	S = int(Tc*fs)  #number of samples per windows
	idc = np.arange(S) #unpermuted sample indices - See init_secondaries in libgnss/settings.py
	fcaid = ds * fc/f
	datacomplex = (datatype.fields.keys()==['i','q'])
 
	
#This function will genearate the C/A codes
def _make_gps_code_chips(prn, dtype = np.int32):
    #This function is from correlator.py
    #Return an array of C/A code chips (over values +1/-1) for a given PRN specified by the prn input
    #Inputs: 
    #prn: specifies which C/A code to be generated
    #dtype = data type of the output code chip. Default: numpy.int32
    #Outputs:
    #An array of code chip for a given PRN
    
    assert 1<= prn <=37, 'Invalid PRN code: ' + str(prn)
    
    #Intialize the shift register tap points
    tap1 = np.array([1,0,0,0,0,0,0,1,0,0])
    tap2 = np.array([1,1,1,0,1,0,0,1,1,0])
    
    #Initialize g1 and g2 shift register
    g1reg, g2reg = np.ones(10), np.ones(10)
    #Generate g1 and g2 outputs by zeropadding
    g1 = np.hstack((g1reg,np.zeros(1013)))
    g2 = np.hstack((g2reg,np.zeros(1013)))
    
    #Generate the g1 and g2 chips based on feedback polynomials
    for i in xrange(10,1023):
        g1[i]=g1reg.dot(tap1) % 2
        g2[i]=g2reg.dot(tap2) % 2
        g1reg = np.append(g1reg[1:10],g1[i])
        g2reg = np.append(g2reg[1:10],g2[i])
    
    #Shift the g2 code to the right by the delay (in chips)
    delays = [5,6,7,8,17,18,139,140,141,251,252,254,255,256,257,258,469,
              470,471,472,473,474,509,512,513,514,515,516,859,860,861,862,
            863,950,947,948,950]
    g2 = np.roll(g2,delays[prn-1])
    
    #Form the chips as G1+G2 modulo 2 taking values of +1 or -1
    return np.where((g1+g2)%2 ==0, -1,1).astype(dtype)



#The following function is adapted from correlator.py
def _trim_mean(arr, percent):
        """
        #Returns the trimmed mean using percentage-based trimming.
        @type arr: numpy.ndarray
        @param arr: One-dimensional array of data.
        @type percent: float
        @param percent: 'middle' percent of data that we want to average.
        @rtype: numpy.ndarray
        @return: trimmed mean using percentage-based trimming
        """
	lower = sps.scoreatpercentile(arr, percent/2)
	upper = sps.scoreatpercentile(arr, 100 - percent/2)
	return sps.tmean(arr, limits=(lower, upper), inclusive=(False, False))

#Implementation of the search_signal function of the correlator class
def search_signal(prn,I=10, dHz = 500, B=24):
#See correlator.py. I added prn as a replacement for self.prn and fc0 as a replacement for sle
#Perform 2D brute force search for the code phase and the Doppler shift
#It save the current location of the file pointer
#reads in I x 1 ms of data, then resets the file pointer to the saved location.
#As such, this method can be called anytime, not only at the begnining of the file

#If coherent integration is used, set I = 10, dHz = 100, B=24*5
#if incoherent integration is used, set I = 10, dHz = 500, B =24
#Inputs:
#   prn: input PRN to generate the C/A code
#   I: an integer specifying the number of integration to perform. Default is 15. Each integration is over S samples (1 millisecond)
#   dHz: a floating point number specifying the size of the Doppler search bin
#   B: an integer specifying the number of coarse acquisition Doppler bins to use. Bins woll be evenly distributed around the intermediate 
#   carrier frequency. If B is even, B will be incremented by 1 to be made odd.

#Outputs:
#   found: a boolean number indicated that a signal was found
#   doppler: a float number corresponding doppler shift in Hz
#   cppr: a float number showing acquisition metric comparing peak height to next highest
#   cppm: a float number showing acquisition metric comparing peak height to data average

    """
    #Added by Tan for debugging
    print "datacomplex: ", datacomplex
    print  " Tc: ", Tc
    print  " ds: ", ds
    print  " fc: ", fc 
    print " f: ", f
    print " Lc: ", Lc
    print  " fs: ", fs
    print  " S", S
    print  " Tc: ", Tc
    print " idc: ", idc
    print  " fi: ", fi
    print  "fcaid: ", fcaid #ds*fc/f
    """	 
    
    global fc; 
    fc = fc0; 
 #   print 'Complex data from file: ', raw_data
 #   print 'Number of complex sample: ', raw_data.shape
 #   print 'iMat.shape: ' , iMat.shape
     
    #Initializing commonly used parameters
    #---[Tan] The following 2-line  initialization was taken from __init__() of correlator.py
    _rc = None #Code phase at the zero sample   
    _ri = None #(itermediate) carrier phase at the zero sample

    #Read in data and revert the file pointer to its original position
    fileposition0 = fileobject.tell()
    raw_data = np.fromfile(fileobject,datatype,I*S); #Read I times of S samples each
    #Convert the data into complex format
    print "raw_data['i'].dtype", raw_data['i'].dtype


    if datacomplex:
        raw_datac = raw_data['i']+1j*raw_data['q']
    #Get back to the original position
    fileobject.seek(fileposition0,0)


    #Get the continuous samping times from 0 to S*I, expressed in seconds
    #continuos time is important for maintaining phase relationship when performing coherent integration
    ts = np.arange(0,S*I)/fs
    
    #Produce the long replica carrier signals (for each Doppler bin)
    B = int(B)+(int(B)+1)%2 #Convert B into an odd number
    bins = np.mat(np.arange((1-B)/2,(B-1)/2+1)).T #Column matrix of doppler bin indices
    
    iMat = np.array(np.exp(-1j*(2*np.pi*(fi + bins*dHz)*ts))) #A 2D array of carriers replicas in time. Each row is for a different Doppler frequency
    
   
    chips = _make_gps_code_chips(prn) #Generate the C/A code given the input PRN
    r = chips.take(np.mod(np.floor(ts*fc),Lc).astype(np.int32))#Returned indexed array elements
    #this section of the code implements the self.chips.take function

    #Set up the acquisition metric (running coherent/noncoherent sums)
    A = np.zeros((np.size(bins),S),dtype = complex)#Each row is a different Doppler
    
    #Perform multiple (I) coherent/noncoherent integration, checking all Doppler bins
    for b in range(np.size(bins)): # Go through all possible Doppler shift
        for k in range(I):
            #take conjugate of fft of the code replica
            rfft = np.conjugate(np.fft.fft(r[(k*S):(k+1)*S]))
            #Mix the raw signal to baseband and take its fft
	    pfft = np.fft.fft(raw_datac[(k*S):((k+1)*S)]*iMat[b,(k*S):((k+1)*S)])
	    #Perform the correlation
	    A[b,:] = A[b,:] + np.abs(np.fft.ifft(pfft*rfft))
	    #A[b,:] = A[b,:] + np.fft.ifft(pfft*rfft)
	    #for coherent integration, we should check the next 10 ms too
    A = np.abs(A)
    
    #For each code phase shift, get the max acquisition value across frequencies
    maxs = np.array([np.max(A[:,s]) for s in idc])
      
    #Now get the max peak values as well as sample and Doppler indices
    sidx = maxs.argmax()  #code phase index
    didx = A[:,sidx].argmax() #Doppler frequency index
    peak = maxs[sidx]
	
    #Produce indices to mask the peak chip for the CPPR and CPPM computation
    m = int(math.ceil(fs/fc))
    maskidx = np.arange(-m,m+1) + sidx

    #Compute the CPPR and CPPM
    maxs[np.where(maskidx> 0, maskidx %s, maskidx)] = 0
    cppr = peak / np.max(maxs)
    cppm = peak / _trim_mean(maxs, 10)
 
   	
    if (cppm >2.0):
	#Update fc using coarse acquisition estimate
	fc = fc + fcaid*(bins[didx,0]*dHz) 
   
	#Create the long replica code (shifted by sidx)
	r = chips.take(np.roll(np.mod(np.floor(ts*fc),Lc).astype(np.int32),sidx))
	
	#Remove the code from raw signal without 0 Hz component
	rawc = (raw_datac-np.mean(raw_datac))*r
	
	if datacomplex:
		#Create signal to mix carrier to baseband
		mix = np.array(np.exp(-1j*(2*pi*fi*ts)))
		#mix raw signal to baseband
		rawc = rawc * mix
	
	#Take the fft of raw carrier and get the unique fft points
	fftpts = 8*(1<<np.size(ts*fc).bit_length())
	rawcfft = np.fft.fft(rawc, fftpts)
        
	#print "rawcfft 1: ", rawcfft
 	
	fftbins = np.fft.fftfreq(n = fftpts, d = 1.0/fs)
    
    	if datacomplex:
		#put a mask so that it doesn't search outside the specified doppler range
		rawcfft[(fftbins<(1.0-B)/2.0*dHz)] = 0.0
		rawcfft[(fftbins>((B-1.0)/2.0 + 1.0)*dHz)]=0.0
		didx = np.abs(rawcfft).argmax()
		#Compute the Doppler shift
		_doppler = fftbins[didx]
	else:
		rawcfft[(fftbins<fi+(1.0-B)/2.0*dHz)]=0.0
		rawcfft[(fftbins>fi+((B-1.0)/2.0+1.0)*dHz)]=0.0
		didx = np.abs(rawcfft).argmax()
		_doppler = fftbins[didx]-fi

	#Save the rest of the acquisition search results
	#code phase (chips) at zero phase sample
	_rc = Lc - sidx *(fc/fs)
	_ri = np.angle(rawcfft[didx])/(2.0*pi)
	"""
	print "raw_data: ", raw_data#matched
	print "fftpts: ", fftpts#matched
	print "B: ", B#matched
	print "r: ", r
	print "dHz: ", dHz#matched
	print "fi: ", fi#matched
 	print "didx: ", didx#matched
    	print "fftbins[didx]: ",fftbins[didx]#matched
	print "I*S", (I*S)#matched
	print "datacomplex: ", datacomplex
	print "Lc: ", Lc #matched
 	print "sidx: ", sidx #matched
	print "fc: ", fc 
 	print "fs: ", fs #matched
 	print "didx: ", didx #matched
	print "rawcfft: ", rawcfft
        print "self.rc: ", _rc
 	print "self.ri: ", _ri
	"""

    else:
	_doppler = bins[didx,0]*dHz

    _found = (cppm>2.0) 
    #Returns:
	#_rc: zero sample code phase
	#_found: boolean whether or not a signal was found
	#_dopler: doppler shift in Hz
	#cppr: acquisition metric comparing peak height to the next heighest
	#cppm: acquisition metric comparing peak height to data average

    #print "fileposition0: ", fileposition0
	
    return _rc, _found, _doppler, cppr, cppm, _ri
   

#This function is adapted from correlator.correlate function().
#[Tan note]: i put 3 new fields l_a, p_a and e_a into the channel() structure
def correlate(prn, fc_in, fi_in):
        """
        Correlate one millisecond of raw data against the replica signal.
        The replica signal is produced based on the input code and carrier frequencies
        (and internally saved code and carrier phase offsets).
        Inputs:
	prn: the C/A code of the kernel of interest
	fi_in: a floating point number corresponds to the current (intermediate) carrier frequency in cycles per second.
        fc_in: a float shows the current code frequency in chips per second.
	rc_in: Code phase chips
	ri_in: intermediate carrier phase
        Outputs
            rc : zero sample code phase,            
            ri : zero sample carrier phase,            
            c_r  : receiver synchronous correlations (from 0:S samples),
            c_s1 : signal synchronous correlations (from prev:xs1 samples),
            c_s2 : signal synchronous correlations (from xs1:xs2 samples).
        """
        
        # note on the variables that are used in this section of the code
        # idc: unpermuted sample index
        # ri : intermediate carrier phase (cycles)
        # Lc : ideal number of code chips per window (1023.0 chips)
        # offset: 0.5 chips, such that the different between E and L is 1 chip
        # chips : 1023 code chips from the function _make_gps_code_chips()
        
        # Read in the next set of samples and convert to baseband.
        # samples are complex signal amplitude values
        
        # initialize commonly used parameters
        rc = rx_channel[channelNr].rc #Code phase (chips) - self.rc in correlator is replaced by the searching reuslts
        ri = rx_channel[channelNr].ri  #Intermediate carrier phase (cycles)
	offset = t_elspacing
	fi = fi_in
	fc = fc_in
	"""
	print "Value of rc, ri before computing the correlation: "
	print "rc: ", rc
	print "ri: ", ri 
	print "fi: ", fi
	print "fc: ", fc
	print "Tc: ", Tc
	"""
	# also can view as the 'remaining code phase' of previous correlation window
        # |bbbb|aaaaaa| |bbbb|rc_aaa| |bbbb|aaaaaa|

        raw = np.fromfile(fileobject, datatype, S)
        if datacomplex:
            raw = raw['i']+1j*raw['q']
        
        trigarg = ((2.0*np.pi*fi)*(idc/fs)) + (2.0*np.pi*ri) 
        baseband = np.exp(-1j*trigarg)*raw

        baseband_f = np.fft.fft(baseband,S) #S is the number of samples per windows
        fftbins    = np.fft.fftfreq(n = S, d = 1.0/fs)
        baseband_f[np.where((fftbins<-fc))] = 0
        baseband_f[np.where((fftbins>fc))]  = 0
        baseband   = np.fft.ifft(baseband_f, S)
        
        # Compute the expected code boundary locations (xs1, xs2) in samples.
        # fc is included in here to address the code doppler
        xs1 = (Lc - rc) * (fs/fc)
        xs2 = xs1 + (Lc * (fs/fc))

        # Produce the code replica indices.
        # fidc is the prompt replica's indices 
        # because it is shifted by xs1 to match the received signal
        fidc = (idc - xs1) * (fc/fs) # float code indices by sample
        #pictorial representation in page 177 of Kaplan and Hegarty
        #data   0 1 2 3 4 5 6 7 fidc
        #early  1 2 3 4 5 6 7 0 fidc + offset
        #prompt 0 1 2 3 4 5 6 7 fidc
        #late   7 0 1 2 3 4 5 6 fidc - offset
        eidc = np.mod(np.floor(fidc + offset),Lc).astype(np.int16) # early indices
        pidc = np.mod(np.floor(fidc         ),Lc).astype(np.int16) # prompt indices
        lidc = np.mod(np.floor(fidc - offset),Lc).astype(np.int16) # late indices
        
        # Produce the shifted replica vectors. 
        # numpy.take returns array values from the given indexes
  	chips = _make_gps_code_chips(prn) #Generate the C/A code given the input PRN
  	early  = chips.take(eidc)
        prompt = chips.take(pidc)
        late   = chips.take(lidc)
        
        # initialize for the next iteration
        rx_channel[channelNr].rc = (rx_channel[channelNr].rc +fc*Tc)% Lc
        rx_channel[channelNr].ri =  rx_channel[channelNr].ri +fi*Tc
        
        # initialize the sample boundaries 
        # (+1 is added to include the sample during array indexing)
        idxs1 = np.mod(np.floor(xs1),S).astype(np.int16) + 1 
       
	"""
	#This section of the code is just for debugging
	print "xs1: ", xs1
	print "xs2: ", xs2
	print "chips: ", chips
	print "early: ", early
	print "prompt: ", prompt
	print "late: ", late
	print "idxs1: ", idxs1
	#print "S", S
	print "baseband: ", baseband
	print "trigarg: ",trigarg
	#print "raw: ",raw
	#print "idc: ", idc
	#print "fs: ", fs
	#print "fi: ", fi
	print "ri: ", ri
	#print "pi: ", pi
	"""

        # The normal case: the first boundary is within this window, and the
        # second boundary is outside of this window.
        if  xs1 <= S < xs2:
            # The flow of this case is as follows:
            # First correlate part B and form signal   synchronous outputs.
            # Next  correlate part A and form receiver synchronous outputs.
            # |bbb|aaaaaa| |bbb|aaaaaa|
        
            # Segment the baseband vector into B and A parts.
            baseband_b = baseband[:idxs1]
            baseband_a = baseband[idxs1:]
            
            # Get the correlations for segments B and A. 
            e_b = np.inner(baseband_b, early[:idxs1])  # early  part B
            p_b = np.inner(baseband_b, prompt[:idxs1]) # prompt part B
            l_b = np.inner(baseband_b, late[:idxs1])   # late   part B
            e_a = np.inner(baseband_a, early[idxs1:])  # early  part A
            p_a = np.inner(baseband_a, prompt[idxs1:]) # prompt part A
            l_a = np.inner(baseband_a, late[idxs1:])   # late   part A
            
            # Prepare the signal synchronous correlation outputs.
            e_s = rx_channel[prn].e_a + e_b # early  signal synchronous
            p_s = rx_channel[prn].p_a + p_b # prompt signal synchronous
            l_s = rx_channel[prn].l_a + l_b # late   signal synchronous
         
            # Record the part A correlations for use in the next update.
            rx_channel[prn].e_a, rx_channel[prn].p_a, rx_channel[prn].l_a = e_a, p_a, l_a 
            
            # c_s1 = i-early, q-early, i-prompt, q-prompt, i-late, q-late.
            c_s1 = e_s.real, e_s.imag, p_s.real, p_s.imag, l_s.real, l_s.imag
            
            # Prepare the receiver synchronous correlation outputs.
            pos = np.abs(e_b + p_b + l_b + e_a + p_a + l_a)
            neg = np.abs(e_b + p_b + l_b - e_a - p_a - l_a)
            
            e_r, p_r, l_r = 0, 0, 0
            
            if pos > neg: # there was no polarity change from B to A
                e_r = e_b + e_a # early  receiver synchronous
                p_r = p_b + p_a # prompt receiver synchronous
                l_r = l_b + l_a # late   receiver synchronous
            else: # there was a polarity change from B to A
                e_r = e_b - e_a # early  receiver synchronous
                p_r = p_b - p_a # prompt receiver synchronous
                l_r = l_b - l_a # late   receiver synchronous
                
            # c_r = i-early, q-early, i-prompt, q-prompt, i-late, q-late.
            c_r = e_r.real, e_r.imag, p_r.real, p_r.imag, l_r.real, l_r.imag
            
            # Returns:
            # rc   : zero sample code phase (for the start of this correlation window),
            # c_r  : receiver synchronous correlations (at S samples),
            # c_s1 : signal synchronous correlations (at xs1 samples),
            # None : second signal synchronous correlations (there were none).
            return rc, ri, c_r, c_s1, None
        
        if  xs1 < xs2 <= S:
            
            # The flow of this case is as follows:
            # First correlate part B and form signal synchronous outputs.
            # Next correlate an A & B pair and form signal synchronous outputs.
            # Finally correlate part A and form receiver synchronous outputs.
            
            # Compute the zero sample index of the second code.
            # (+1 is added to include the sample during array indexing)
            idxs2 = np.mod(np.floor(xs2),S).astype(np.int16)+1
            
            # Segment the baseband vector into B, AB, and A parts.
            baseband_b = baseband[:idxs1]
            baseband_s = baseband[idxs1:idxs2]
            baseband_a = baseband[idxs2:]
            
            # Get the correlations for the first B segment. 
            e_b = np.inner(baseband_b, early[:idxs1])  # early  part B
            p_b = np.inner(baseband_b, prompt[:idxs1]) # prompt part B
            l_b = np.inner(baseband_b, late[:idxs1])   # late   part B
            
            # Prepare the signal synchronous correlation outputs.
            e_s = rx_channel[prn].e_a + e_b
            p_s = rx_channel[prn].p_a + p_b
            l_s = rx_channel[prn].l_a + l_b
            
            # c_s1 = i-early, q-early, i-prompt, q-prompt, i-late, q-late.
            c_s1 = e_s.real, e_s.imag, p_s.real, p_s.imag, l_s.real, l_s.imag
            
            # Get the correlations for the complete A and B segment pair. 
            e_s = np.inner(baseband_s, early[idxs1:idxs2])  # early  part S
            p_s = np.inner(baseband_s, prompt[idxs1:idxs2]) # prompt part S
            l_s = np.inner(baseband_s, late[idxs1:idxs2])   # late   part S
            
            # c_s2 = i-early, q-early, i-prompt, q-prompt, i-late, q-late.
            c_s2 = e_s.real, e_s.imag, p_s.real, p_s.imag, l_s.real, l_s.imag                
            
            # Get the correlations for the final A segment.
            e_a = np.inner(baseband_a, early[idxs2:])  # early  part A
            p_a = np.inner(baseband_a, prompt[idxs2:]) # prompt part A
            l_a = np.inner(baseband_a, late[idxs2:])   # late   part A
            
            # Record the part A correlations for use in the next update.
            rx_channel[prn].e_a, rx_channel[prn].p_a, rx_channel[prn].l_a = e_a, p_a, l_a
            
            # Prepare the receiver synchronous correlation outputs.            
            pos = np.abs(e_b + p_b + l_b + e_s + p_s + l_s)
            neg = np.abs(e_b + p_b + l_b - e_s - p_s - l_s)
            
            e_r, p_r, l_r = 0, 0, 0
            
            if pos > neg: # there was no polarity change from B to AB

                pos = np.abs(e_s + p_s + l_s + e_a + p_a + l_a)
                neg = np.abs(e_s + p_s + l_s - e_a - p_a - l_a)
                
                if pos > neg: # there was no polarity change from AB to A
                    e_r = e_b + e_s + e_a
                    p_r = p_b + p_s + p_a
                    l_r = l_b + l_s + l_a
                else: # there was a polarity change from AB to A
                    e_r = e_b + e_s - e_a
                    p_r = p_b + p_s - p_a
                    l_r = l_b + l_s - l_a
                    
            else: # there was a polarity change from B to AB (thus not AB to A)

                e_r = e_b - e_s - e_a
                p_r = p_b - p_s - p_a
                l_r = l_b - l_s - l_a
                
            # c_r = i-early, q-early, i-prompt, q-prompt, i-late, q-late
            c_r = e_r.real, e_r.imag, p_r.real, p_r.imag, l_r.real, l_r.imag 
            
            # Returns:
            # rc   : zero sample code phase (for the start of this correlation window)
            # c_r  : receiver synchronous correlations (at S samples),
            # c_s1 : signal syncrhonous correlations (at xs1 samples),
            # c_s2 : signal synchronous correlations (at xs2 samples).
            return rc_in, ri, c_r, c_s1, c_s2
         
        if  S < xs1:
            # The flow of this case is as follows:
            # First correlate part B, actually a continuation of A from the
            # last correlation window.  Form the receiver synchronous outputs.
        
            # Get the correlations for the continuous B segment. 
            e_b = np.inner(baseband, early)
            p_b = np.inner(baseband, prompt)
            l_b = np.inner(baseband, late)
            
            # Update the part A correlations for use in the next update.
            rx_channel[prn].e_a = rx_channel[prn].e_a + e_b
            rx_channel[prn].p_a = rx_channel[prn].p_a + p_b
            rx_channel[prn].l_a = rx_channel[prn].l_a + l_b
            
            # Prepare the receiver synchronous correlation outputs.
            # c_r = i-early, q-early, i-prompt, q-prompt, i-late, q-late.
            c_r = e_b.real, e_b.imag, p_b.real, p_b.imag, l_b.real, l_b.imag 
            
            # Returns:
            # rc   : zero sample code phase (for the start of this correlation window)
            # c_r  : receiver synchronous correlations (at S samples),
            # None : first signal synchronous correlations (there were none),
            # None : second signal synchronous correlations (there were none).
            return rc, ri, c_r, None, None

#-----------------Main source code start here-------------------------
update(settings)
print filename
#Open Xillybus device file for data read and write
#ofile = open("/dev/xillybus_write_32","wb")
#ifile = open("/dev/xillybus_read_32","r")

#ofile.write("Demonstration of file writting in python\n")
#ofile.flush()

# Test reading files
#in_data = ifile.read(20) 
#print "Read back string: ", in_data


#--------------------------------------------------------------------

#declare array of Channel objects to store variables
#should replace this to store only signals found when acquisition is tested working
for channelNr in channelList:
    rx_channel[channelNr]=Channel(channelNr)

#Initialize an empty dictionary to store scalar tracking channels
rx_channel = {}

#Open the raw data file - from correlator.open_raw_file
fileobject = open(filename,'rb');
datatype = np.dtype([('i', np.short),('q', np.short)]) #Each field i and q is of type short
datacomplex = datatype.fields.keys()==['i','q'] #Boolean value telling if the data is real or complex see init_secondaries in settings.py
fileobject.seek(skip_samples*datatype.itemsize,0)#Move to a new file position after skipping from the first few samples


print "INITIALIZATION: ACQUISTION"
#---The section of the code below is from cold start....
for channelNr in channelList: #Here PRN = channelNr
#for channelNr in [31]:
    #Open the raw data file - from correlator.open_raw_file
    fileobject = open(filename,'rb');
    datatype = np.dtype([('i', np.short),('q', np.short)]) #Each field i and q is of type short
    datacomplex = datatype.fields.keys()==['i','q'] #Boolean value telling if the data is real or complex see init_secondaries in settings.py
    fileobject.seek(skip_samples*datatype.itemsize,0)#Move to a new file position after skipping from the first few samples

    #print("Working on channel: ", channelNr)
    acq_result = search_signal(channelNr)
    #Display the search_signal results
    print "PRN: ", channelNr, acq_result
    #The following lines are from the rest of cold_start
    fi_bias = acq_result[2]#initialize carrier NCO bias
    _fi = fi0 + fi_bias #Internal intermediate carrier frequency tracker
    _fc = fc0 + fc_bias + fcaid*fi_bias #internal code frequency tracker - [Tan]: Note that i do not understand here why fc in the setting
    
    """
    print "fc: ", fc
    print "fc_bias: ", fc_bias
    print "fcaid: ", fcaid
    print "fi_bias: ", fi_bias
    """

    #print "_fi (iternal intermediate carrier frequency tracker): ", _fi
    #print "_fc (iternal code frequency tracker): ", _fc
   
	
    #Add the search_signal result to a dictionary of rx_channel
    if (acq_result[1]==True):
	rx_channel[channelNr]=Channel(channelNr)
	rx_channel[channelNr].update_found_params({'_fi':_fi,'_fc':_fc,'fi_bias':fi_bias,'fc_bias':fc_bias,'fcaid':fcaid,'rc':acq_result[0],'ri':acq_result[5]})#Update the channel with found results

	#This section of the code implement channel.update() method
	rc, ri, c_r, c_s1, c_s2 = correlate(channelNr,_fc,_fi); #This line is adapted from rc, ri, c_r, c_s1, c_s2 = self.correlator.correlate(self._fi, self._fc) 	
	ms = rx_channel[channelNr]._mscount
        cp = rx_channel[channelNr]._cpcount
	"""
	rx_channel[channelNr].rc[ms] = rc       # records the current zero-sample code phase, at the start of this correlation window
        rx_channel[channelNr].ri[ms] = ri       # records the current zero-sample carrier phase, at the start of this correlation window
        rx_channel[channelNr].fi[ms] = 	rx_channel[channelNr]._fi # records the current intermediate frequency fi
        rx_channel[channelNr].fc[ms] = 	rx_channel[channelNr]._fc # records the current code frequency fc
        rx_channel[channelNr].iE[ms], rx_channel[channelNr].iP[ms], rx_channel[channelNr].iL[ms] = c_r[0], c_r[2], c_r[4] #saved using receiver synchronous correlations
        rx_channel[channelNr].qE[ms], rx_channel[channelNr].qP[ms], rx_channel[channelNr].qL[ms] = c_r[1], c_r[3], c_r[5]
        """
	#Display the correlation results
	print "rc: ", rc
	print "ri: ", ri
	print "c_r: ", c_r
	print "c_s1: ", c_s1
	print "c_s2: ", c_s2
	
    #Close the data file
    #dont close... keep open
    #fileobject.close()

#--Close Xillybus device file for  data read and write
#ifile.close()
#ofile.close()



print ("Done with writing to all channels...");    
