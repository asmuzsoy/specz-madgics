const delLog = 0.5e-4


"""Function that converts linearly spaced spectra to log spaced

Parameters
----------
spec : :class:Matrix
    spectra in linearly spaced wavelength bins
wave : :class:Vector
    corresponding linearly spaced wavelength bins (must be equal sized bins)
real_log_range : :class:Vector
    desired log spaced wavelength bins
kernel_size : :class:Int
    kernel size for Lanczos interpolation
Returns
-------
Log spaced spectrum (flux values)
"""

function linear_to_log(spec, wave; real_log_range = log_wave_range, kernelsize = 4)
    itp = Interpolations.interpolate(spec, Interpolations.Lanczos(kernelsize))
    etp = Interpolations.extrapolate(itp, 0) # fill value is zero

    # these are the wavelength values of log spaced bins
    real_linear_log_range = 10 .^ (real_log_range)

    # Lanczos doesn't want to take in x points, it just does it based on indexing
    # Calculating which indexes these wavelengths would correspond to
    index_range = ((real_linear_log_range .- minimum(wave)) / (wave[2] - wave[1])) .+ 1 # because one indexed

    log_binned_spectrum = etp(index_range)

    return log_binned_spectrum
end

"""Get wavelength of Lyman alpha line in Angstroms

Parameters
----------
z : :class:float
    redshift to get LAE wavelength at
Returns
-------
Wavelength of Lyman alpha emission shifted to given redshift, in Angstroms
"""
function get_wavelength(z)
    return 1215.67 * (1 + z)
end

"""Get log10-wavelength of Lyman alpha line in Angstroms

Parameters
----------
z : :class:float
    redshift to get LAE wavelength at
Returns
-------
Log10-wavelength of Lyman alpha emission shifted to given redshift, in log(Angstroms)
"""
function get_log_wavelength(z)
    return log10.(get_wavelength(z))
end

"""Shift a spectrum to be at a given redshift

Parameters
----------
spectrum : :class:Vector
    log-binned spectrum at redshift z
z : :class:float
    true redshift of spectrum
ref_z : :class:float
    redshift to shift spectrum to
Returns
-------
Spectrum shifted to be at ref_z using the same log-wavelength bins
"""
function adjust_spectrum_log(spectrum, z; ref_z = 2.45)
    adjustment = log10(1 + ref_z) - log10(1 + z)
    pixel_shift = round(Int, adjustment / delLog)
    adjusted_spectrum = ShiftedArrays.circshift(spectrum, pixel_shift)
    if pixel_shift > 0
        adjusted_spectrum[1:pixel_shift] .= NaN
    else
        adjusted_spectrum[(end+pixel_shift):end] .= NaN
    end
    return adjusted_spectrum
end

"""Shift a spectrum to be at a given redshift and also shift
    the ivars in the same way

Parameters
----------
spectrum : :class:Vector
    log-binned spectrum at redshift z
ivar : :class:Vector
    log-binned ivar corresponding to spectrum
z : :class:float
    true redshift of spectrum
ref_z : :class:float
    redshift to shift spectrum to
Returns
-------
Spectrum and ivars shifted to be at ref_z using the same log-wavelength bins
"""
function adjust_spectrum_and_ivar(spectrum, ivar, z; ref_z = 2.45)
    adjustment = log10(1 + ref_z) - log10(1 + z)
    pixel_shift = round(Int, adjustment / delLog)
    adjusted_spectrum = ShiftedArrays.circshift(spectrum, pixel_shift)
    adjusted_ivar = ShiftedArrays.circshift(ivar, pixel_shift)
    if pixel_shift > 0
        adjusted_spectrum[1:pixel_shift] .= NaN
        adjusted_ivar[1:pixel_shift] .= NaN
    else
        adjusted_spectrum[(end+pixel_shift):end] .= NaN
        adjusted_ivar[(end+pixel_shift):end] .= NaN
    end
    return adjusted_spectrum, adjusted_ivar
end

"""In-place replacement of nans with zeros in a matrix

Based on https://stackoverflow.com/questions/53463436/how-to-replace-inf-values-with-nan-in-array-in-julia-1-0

Parameters
----------
x : :class:Matrix
    Matrix of floats
Returns
-------
nothing
"""
function nan_to_zero(x)
    # replaces NaNs with zeros in an array
    # modifies the actual array so be careful!
    for i in eachindex(x)
        @inbounds x[i] = ifelse(isnan(x[i]), 0, x[i])
    end
end

"""Shift a spectrum to be at a given redshift and also shift
    the ivars in the same way

Parameters
----------
spectrum : :class:Vector
    log-binned spectrum at redshift z
ivar : :class:Vector
    log-binned ivar corresponding to spectrum
z : :class:float
    true redshift of spectrum
ref_z : :class:float
    redshift to shift spectrum to
Returns
-------
Spectrum and ivars shifted to be at ref_z using the same log-wavelength bins
"""
function filter_nanvar(spec; nanvar_threshold = 10^-1, zero_out_nans = true)
    nanvars = nanvar(spec, dims = 1)
    nanvar_mask = vec(((nanvars .< nanvar_threshold) .&& (nanvars .> 0)))

    filtered_spec = copy(spec[:, nanvar_mask])
    if zero_out_nans
        nan_to_zero(filtered_spec)
    end
    return filtered_spec
end

"""Returns the index of the array closest to the given value

Parameters
----------
array : :class:Vector
    array to get closest index of
value : :class:float
    value that we want to find the closest index of array to
Returns
-------
index of closest value in array (int)
"""
function get_closest_index(value, array)
    return partialsortperm(abs.(array .- value), 1)
end