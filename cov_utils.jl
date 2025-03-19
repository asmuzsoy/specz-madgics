"""Separates spectra into sky and residual components given pre-determined sky and residual covariance matrices

Parameters
----------
spec : :class:Matrix
    spectra in a matrix of size (num_wavelengths, n) where n is number of spectra
Csky : :class:Matrix
    sky covariance matrix of size (num_wavelengths, num_wavelengths)
Cres : :class:Matrix, optional
    residual covariance matrix of size (num_wavelengths, num_wavelengths)

Returns
-------
res : :class:Matrix
    The residual component of spec, size (num_wavelengths, n)
μsky : :class:Matrix
    The residual component of spec, size (num_wavelengths, n
"""
function apply_Csky_no_split(spec, Csky, Cres)
    Ctotinv = inv((Csky + Cres))

    # infer the sky and residual components
    μsky = (Csky * Ctotinv) * spec
    res = (Cres * Ctotinv) * spec

    return res, μsky
end


"""Creates mask where 1 if value is not nan, else 0

Parameters
----------
x : :class:Matrix
    Matrix of floats
Returns
-------
Matrix y, of same shape as x, with 0s where x has nans and 1s otherwise
"""
function nan_mask(x)
    # want 1 if value is not nan, else 0
    y = copy(x)
    for i in eachindex(x)
        @inbounds y[i] = ifelse(isnan(x[i]), 0, 1)
    end
    return y
end


"""Makes data-driven wavelength-wavelength covariance matrix from spectra

Parameters
----------
spec : :class:Matrix
    spectra in a matrix of size (num_wavelengths, n) where n is number of spectra
counts_matrix : :class:Bool
    if true, divide by symmetric matrix (num_wavelengths, num_wavelengths) with entries consisting of the number of spectra with non-nan values for that wavelength bin
    if false, will just use the number of spectra for each wavelength

Returns
-------
covariance matrix of size (num_wavelengths, num_wavelengths)
"""
function make_cov(spec; counts_matrix = true)
    spec_to_use = spec
    if sum(isnan.(spec)) > 0 # if there are nans, set them to zero
        spec_to_use = nan_to_zero(copy(spec))
    end
    if counts_matrix
        good_mask = nan_mask(spec)
        return (spec_to_use * spec_to_use') ./ (good_mask * good_mask')
    end
    return (spec * spec') ./ size(spec, 2)
end
