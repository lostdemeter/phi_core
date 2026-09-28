"""phi_core public surface (frozen set — extend by adding modules)."""
from phi_core.lattice import (
    PHI, LN_PHI, K, BIAS, NLVL, DMAX, FRAC_CAP, FIXED_F, SIG_SPAN,
    L_FRAC, L_COARSE, L_FINE, sig_lut, sigx_lut,
    encode, decode, to_fixed, from_fixed, bit_length_int, tdiv,
)
from phi_core.calibrate import m_of, calibrate_from_hooks
from phi_core import ops, compose
from phi_core.numpy_ops import (
    tmul, prelu_int, phi_conv, deconv_int, warp_fixed, interp_fixed,
    nearest2, sigmoid_int,
)

__all__ = ["PHI", "LN_PHI", "K", "BIAS", "NLVL", "DMAX", "FRAC_CAP",
           "FIXED_F", "SIG_SPAN", "L_FRAC", "L_COARSE", "L_FINE",
           "sig_lut", "sigx_lut", "encode", "decode", "to_fixed",
           "from_fixed", "bit_length_int", "tdiv", "m_of",
           "calibrate_from_hooks", "ops", "compose", "tmul", "prelu_int", "phi_conv",
           "deconv_int", "warp_fixed", "interp_fixed", "nearest2",
           "sigmoid_int"]
