"""Separates spectra into sky and residual components given pre-determined sky and residual covariance matrices

Parameters
----------
spec : :class:Matrix
    DESI spectra in a matrix of size (num_wavelengths, n) where n is an arbitrary number (but usually num_spec or num_good_spec)
Csky : :class:Matrix
    sky covariance matrix of size (num_wavelengths, num_wavelengths)
Cres : :class:Matrix, optional
    residual covariance matrix of size (num_wavelengths, num_wavelengths)

Returns
-------
res : :class:Matrix
    The residual component of spec, size (num_wavelengths, n)
μskyA : :class:Matrix
    The residual component of spec, size (num_wavelengths, n
"""
function apply_Csky_no_split(spec,Csky,Cres)
    Ctotinv = inv(cholesky((Csky+Cres)))

    # infer the sky and residual components
    μsky = (Csky*Ctotinv)*spec
    res = (Cres*Ctotinv)*spec

    return res, μsky
end


