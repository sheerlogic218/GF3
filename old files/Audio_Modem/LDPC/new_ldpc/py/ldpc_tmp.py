import numpy as np
import ctypes as ct

class code:
    def __init__(self, standard = '802.11n', rate = '1/2', z=27, ptype='A'):
        self.standard = standard
        self.rate = rate
        self.z = z
        self.ptype = ptype
        self.proto = self.assign_proto()
        vdeg, cdeg, intrlv = self.prepare_decoder()
        self.vdeg = vdeg
        self.cdeg = cdeg
        self.intrlv = intrlv
        self.Nv = len(vdeg)
        self.Nc = len(cdeg)
        self.Nmsg = len(intrlv)
        self.N = self.Nv
        self.K = self.Nv - self.Nc
        return

    # other functions deleted as they don't interface to ctypes (they are pure python)
    
    def decode(self, ch, dectype='sumprod2', corr_factor=0.7):
        vdeg = self.vdeg
        cdeg = self.cdeg
        intrlv = self.intrlv
        c_ldpc = ct.CDLL('./bin/c_ldpc.so')
        # preliminary consistency checks
        if len(ch) != len(vdeg):
            raise NameError('Channel inputs not consistent with variable degrees')
        # prepare arguments and outputs
        Nv = self.Nv
        Nc = self.Nc
        Nmsg = self.Nmsg
        app = np.zeros(Nv, dtype=np.double)
        app_p = app.ctypes.data_as(ct.POINTER(ct.c_double))
        ch_p = ch.ctypes.data_as(ct.POINTER(ct.c_double))
        vdeg_p = self.vdeg.ctypes.data_as(ct.POINTER(ct.c_long))
        cdeg_p = self.cdeg.ctypes.data_as(ct.POINTER(ct.c_long))
        intrlv_p = self.intrlv.ctypes.data_as(ct.POINTER(ct.c_long))
        # call C function for the sum product algorithm
        if dectype == 'sumprod':
            it = c_ldpc.sumprod(ch_p, vdeg_p, cdeg_p, intrlv_p, Nv, Nc, Nmsg, app_p)
        elif dectype == 'sumprod2':
            it = c_ldpc.sumprod2(ch_p, vdeg_p, cdeg_p, intrlv_p, Nv, Nc, Nmsg, app_p)
        elif dectype == 'minsum':
            it = c_ldpc.minsum(ch_p, vdeg_p, cdeg_p, intrlv_p, Nv, Nc, Nmsg, app_p, ct.c_double(corr_factor))
        else:
            raise NameError('Decoder type unknonwn')
        return app, it

    def Lxor(self, L1, L2, corrflag=1):
        c_ldpc = ct.CDLL('./bin/c_ldpc.so')
        c_ldpc.Lxor.restype = ct.c_double
        return c_ldpc.Lxor(ct.c_double(L1), ct.c_double(L2), corrflag)

    def Lxfb(self, L, corrflag=1):
        c_ldpc = ct.CDLL('./bin/c_ldpc.so')
        dc = len(L)
        L = np.array(L, dtype=float)
        L_p = L.ctypes.data_as(ct.POINTER(ct.c_double))
        c_ldpc.Lxfb.restype = ct.c_double
        return c_ldpc.Lxfb(L_p, dc, corrflag), L
