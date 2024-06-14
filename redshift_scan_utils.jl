"""Shift a spectrum template which is currently at default_z to an arbitrary z, including sub-pixel shifts

Parameters
----------
V : :class:Matrix
    spectrum template in a matrix of size (10, num_wavelength_bins + padding * 2, num_eigen vectors)
z : :class:float
    redshift to shift to
default_z : :class:float, optional
    redshift that the template is currently at, defaults to 2.45
padding : :class:int, optional
    amount of zero-padding on either end of the template, defaults to 1000 wavelength bins

Returns
-------
adjusted_V : :class:Matrix
    Adjusted spectrum template, size (num_wavelength_bins, num_eigenvectors)
pixel_shift : :class:int
    The number of pixels to shift required to shift to the given redshift
"""
function adjust_V(V, z; default_z = 2.45, padding=1000)
    adjustment = log10(1 + z) - log10(1 + default_z)
    
    pixel_shift_float, decimal_part = fldmod(adjustment / delLog, 1)
    pixel_shift = round(Int, pixel_shift_float)
    subpixel_shift = round(Int, decimal_part*10)
    if subpixel_shift == 10
        subpixel_shift = 0
        pixel_shift += 1
    end
    
    adjusted_V = V[subpixel_shift + 1, :, :]

    adjusted_V = circshift(adjusted_V, pixel_shift)
    return adjusted_V[padding+1:end - padding,:], pixel_shift
end

"""Determines optimal spectroscopic redshift for any number of spectra given template

Parameters
----------
spec : :class:Matrix
    Matrix of spectram size (num_wavelength_bins, num_spectra)
padded_Vmat : :class:Matrix
    spectrum template in a matrix of size (10, num_wavelength_bins + padding * 2, num_eigen vectors)
wave_range : :class:vector
    range of indexes of the total wavelength bins that the template is, i.e. if they are the same it will be 1:num_wavelength_bins
z_to_test : :class:vector
    array of redshifts to try

Returns
-------
chisq : :class:Matrix
    Matrix of chisq surfaces, of size (num_spectra, length(z_to_test))
min_chisqs : :class:vector
    Vector of best delta-chisq values for each spectrum, size (num_spectra,)
new_zs : :class:vector
    Vector of best-fit redshifts for each spectrum, size (num_spectra,)
"""
function scan(spec, padded_Vmat, wave_range, z_to_test)
    small_Cinv = Cinv[wave_range, wave_range]

    numspec = size(spec)[2]

    chisq = zeros(length(z_to_test), numspec)

    log_spec[isnan.(log_spec)] .= 0
    
    # just zero out the data and Vmat
    @time begin
    @showprogress for (i, z) in enumerate(z_to_test)
        adjustedV, pixel_shift = adjust_V(padded_Vmat, z)
        newV = adjustedV
        CinvV = small_Cinv * newV
        M = (I + (newV' * CinvV))
        VtCinvXd = CinvV'*spec[wave_range,:]
        ΔTS = -sum(VtCinvXd.*(M\VtCinvXd),dims=1) 
        chisq[i,:] = ΔTS
    end
    end
    min_chisqs, min_chisq_indexes = findmin(chisq, dims=1)
    min_chisqs = reshape(collect(min_chisqs), numspec);
    
    new_zs = zeros(numspec)

    for (i, index_pair) in enumerate(min_chisq_indexes)
        z_index = index_pair[1]
        new_zs[i] = z_to_test[z_index]
    end
    
    return chisq, min_chisqs, new_zs
end