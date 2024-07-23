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
function scan(spec, padded_Vmat, wave_range, Cinv, z_to_test)
    small_Cinv = Cinv[wave_range, wave_range]

    numspec = size(spec)[2]

    chisq = zeros(length(z_to_test), numspec)

    spec[isnan.(spec)] .= 0
    
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

"""Determines optimal spectroscopic redshift for any number of spectra given template,
including a different residual covariance for every target

Parameters
----------
spec : :class:Matrix
    Matrix of spectra, size (num_wavelength_bins, num_spectra)
ivar : :class:Matrix
    Matrix of inverse variances, size (num_wavelength_bins, num_spectra)
padded_Vmat : :class:Matrix
    spectrum template in a matrix of size (10, num_wavelength_bins + padding * 2, num_eigen vectors)
wave_range : :class:vector
    range of indexes of the total wavelength bins that the template is, i.e. if they are the same it will be 1:num_wavelength_bins
Vsky : :class:Matrix
    low-rank approximation of sky covariance, size (num_wavelength_bins, num_eigenvectors) 
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
function scan_target_Cres(spec, ivar, padded_Vmat, wave_range, Vsky, z_to_test)
    numspec = size(spec)[2]
    num_wave_bins = length(wave_range)
    chisq = zeros(length(z_to_test), numspec)
    spec[isnan.(spec)] .= 0
    M_length = num_sky_eigs + 1
    target_variances = 1 ./ ivar
    
    W = zeros(num_wave_bins, M_length)
    W[:, 1:num_sky_eigs] = Vsky
    AinvW = zeros(num_wave_bins, M_length)

    @showprogress for j in 1:numspec
        Cresinv = inv(Diagonal(target_variances[wave_range,j]))
        d = spec[wave_range, j]
        AinvV = Cresinv * Vsky
        AinvW[:,1:num_sky_eigs] = AinvV
        M = zeros(M_length, M_length)
        M[1:num_sky_eigs, 1:num_sky_eigs] = (I + (Vsky' * AinvV))
        # M = (I + (Vsky' * AinvV))
        VtAinvD = AinvV' * d # V' Ainv' D
        chisq1 = VtAinvD' * (M[1:num_sky_eigs, 1:num_sky_eigs] \ VtAinvD)
        for (i, z) in enumerate(z_to_test)
            # Vlae, _ = adjust_V(padded_Vmat, z)
            Vlae = shifted_lae_templates[:,i]
            # W = hcat(Vsky, Vlae) # preallocate this
            W[:,end] = Vlae
            # AinvW = hcat(AinvV, Cresinv * Vlae)
            AinvW[:,end] = Cresinv * Vlae
            # M2 = @time I + (W' * AinvW)
            cross_mat = (Vsky' * Cresinv * Vlae)
            M[end,1:num_sky_eigs] = cross_mat
            M[1:num_sky_eigs, end] = cross_mat'
            M[end, end] = (Vlae' * Cresinv * Vlae)[1,1]
            WtAinvD = AinvW' * d # V' Ainv' D
            chisq2 = WtAinvD' * (M \ WtAinvD)
            chisq[i,j] = chisq1 - chisq2
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

"""Turns an array of pixel offsets into an array of redshifts given a pixel spacing, assuming 0 pixel offset is z=2.45. This method is modified from apMADGICS.jl.

Parameters
----------
pix : :class:vector
    array of pixel offsets
delLog : :class:float
    pixel spacing between log wavelength bins

Returns
-------
z : :class:vector
    Vector of redshifts
"""
function get_z_from_pixels(pix; delLog = 0.5e-4)
    pix_offset = 10756.381901465484 # assuming we're shifting everything to z = 2.45
    z = 10^((pix_offset + pix)*delLog)-1 
    return z
end

"""Turns an array of pixel offsets into an array of redshifts given a pixel spacing, assuming 0 pixel offset is z=2.45. This method is modified from apMADGICS.jl.

Parameters
----------
pix : :class:vector
    array of pixel offsets
delLog : :class:float
    pixel spacing between log wavelength bins

Returns
-------
z : :class:vector
    Vector of redshifts
"""
function plot_chisq_vs_correctness(true_zs, new_zs, min_chisqs)
    p = plot((new_zs .- true_zs), 
    abs.(min_chisqs), 
    seriestype=:scatter, yscale=:log10, alpha=0.1, 
    ylabel=L"|\Delta \chi^2|",
    xlabel="New z - VI z")
    return p
end

function calculate_accuracy(true_zs, new_zs; threshold=0.01)
    return sum(abs.(new_zs .- true_zs) .< threshold) / length(true_zs)
end