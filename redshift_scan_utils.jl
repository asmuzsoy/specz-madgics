const pix_offset = 10756.381901465484 # assuming we're shifting everything to z = 2.45


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
    adjustment = log10(1 + z) - log10(1 + default_z) # units of log wavelength
    
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
function scan_target_Cres(spec, ivar, padded_Vmat, Vsky, z_to_test; 
        wave_range = 1:length(log_wave_range), fine_scan = true)
    numspec = size(spec)[2]
    num_wave_bins = length(wave_range)
    chisq = zeros(length(z_to_test), numspec)
    spec[isnan.(spec)] .= 0
    M_length = num_sky_eigs + num_lae_eigs
    target_variances = 1 ./ ivar
    num_z = length(z_to_test)
    new_zs = zeros(numspec)
    min_chisqs = zeros(numspec)
    z_uncertainties = zeros(numspec)
    # W = zeros(num_wave_bins, M_length)
    # W[:, 1:num_sky_eigs] = Vsky
    # W = hcat(Vsky, Vlae) # preallocate this
    # AinvW = hcat(AinvV, Cresinv * Vlae)
    AinvW = zeros(num_wave_bins, M_length)
    # wave_range = 1:length(log_wave_range)
    WtAinvD = zeros(M_length)
    sky_length = M_length - num_lae_eigs + 1
    Vsky = Vsky[wave_range, :]
    num_fine_in_pixel = 10
    p = plot()
    @showprogress for j in 1:numspec
    # @showprogress for j in 1:1
        Cresinv = inv(Diagonal(target_variances[wave_range,j]))
        d = spec[wave_range, j]
        AinvV = Cresinv * Vsky[wave_range,:]
        AinvW[:,1:num_sky_eigs] = AinvV
        M = zeros(M_length, M_length)
        M[1:num_sky_eigs, 1:num_sky_eigs] = (I + (Vsky' * AinvV)) # M = I + (W' * AinvW)
        # M = (I + (Vsky' * AinvV))
        # VtAinvD = AinvV' * d # V' Ainv' D
        WtAinvD[1:num_sky_eigs] = AinvV' * d 
        # chisq1 = VtAinvD' * (M[1:num_sky_eigs, 1:num_sky_eigs] \ VtAinvD)
        chisq1 = WtAinvD[1:num_sky_eigs]' * (M[1:num_sky_eigs, 1:num_sky_eigs] \ WtAinvD[1:num_sky_eigs])
        # for (i, z) in enumerate(z_to_test) # course
        for i in 1:num_z
        # for i in 1:1
            Vlae = shifted_lae_templates[wave_range,:,i] # the rest of this template is zero
                       
            AinvVlae = Cresinv * Vlae # 8 μs
            AinvW[:,sky_length:M_length] = AinvVlae # 9 μs
            cross_mat = (Vsky' * AinvVlae) # (Vsky' * Cresinv * Vlae) # 145 μs
            M[sky_length:M_length,1:num_sky_eigs] = cross_mat' # 7 μs
            M[1:num_sky_eigs, sky_length:M_length] = cross_mat # 3 μs
            M[sky_length:M_length, sky_length:M_length] = (I + (Vlae' * AinvVlae)) # (I + (Vlae' * Cresinv * Vlae)) # 23 μs
            
            # @time WtAinvD = AinvW' * d # V' Ainv' D
            WtAinvD[sky_length:M_length] = AinvVlae' * d # 5 μs 
            chisq2 = WtAinvD' * (M \ WtAinvD) # A \ B = inv(A) * B # 29 μs
            chisq[i,j] = chisq1 - chisq2 # 6 μs
        end
        if fine_scan
            min_chisq, best_index = findmin(chisq[:,j])
            # println(best_index, " ", z_to_test[best_index])
            fine_index = (best_index - 1) * num_fine_in_pixel + 1
            new_chisqs = zeros(10*num_fine_in_pixel + 1)
            min_index = max(fine_index - 5*num_fine_in_pixel, 1)
            max_index = min(fine_index + 5*num_fine_in_pixel, 45001)
            # fine scan
            for (k, z) in enumerate(fine_z_to_test[min_index:max_index])
                Vlae = fine_shifted_lae_templates[wave_range,:,(min_index + k - 1)] # the rest of this template is zero

                AinvVlae = Cresinv * Vlae # 8 μs
                AinvW[:,sky_length:M_length] = AinvVlae # 9 μs
                cross_mat = (Vsky' * AinvVlae) # (Vsky' * Cresinv * Vlae) # 145 μs
                M[sky_length:M_length,1:num_sky_eigs] = cross_mat' # 7 μs
                M[1:num_sky_eigs, sky_length:M_length] = cross_mat # 3 μs
                M[sky_length:M_length, sky_length:M_length] = (I + (Vlae' * AinvVlae)) # (I + (Vlae' * Cresinv * Vlae)) # 23 μs

                # @time WtAinvD = AinvW' * d # V' Ainv' D
                WtAinvD[sky_length:M_length] = AinvVlae' * d # 5 μs 
                chisq2 = WtAinvD' * (M \ WtAinvD) # A \ B = inv(A) * B # 29 μs
                new_chisqs[k] = chisq1 - chisq2
            end
            # println(new_chisqs)
            plot!(p, new_chisqs)
            min_chisqs[j], best_index = findmin(new_chisqs)
            z_uncertainties[j] = get_z_unc(new_chisqs, best_index, pixel_scan_range=fine_pixel_scan_range[min_index:max_index])
            new_zs[j] = (fine_z_to_test[min_index:max_index])[best_index]
            
        else
            min_chisqs, min_chisq_indexes = findmin(chisq, dims=1)
            min_chisqs = reshape(collect(min_chisqs), numspec);

            new_zs = zeros(numspec)

            for (i, index_pair) in enumerate(min_chisq_indexes)
                z_index = index_pair[1]
                new_zs[i] = z_to_test[z_index]
            end
    
        end
    end

    return chisq, min_chisqs, new_zs, z_uncertainties, p
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

function err1d(vrng,chi2,minind;stepx=3)
    lval = length(vrng)
    # subgrid = minind .+ (-stepx:stepx)
    subgrid = minind .+ [-stepx,0,stepx]

    # println(length(subgrid))
    if (subgrid[1] < 1) | (lval < subgrid[end])
        return NaN
    else
        c1, c2, c3 = chi2[subgrid]
        # println(c1, c2, c3)
        x1, x2, x3 = vrng[subgrid]
        dx = (x2-x1)
        # print(dx, " ", (x3-x2))
        if !((x3-x2) ≈ dx)
            @warn "Non uniform grid error estimates not implemented"
        end
        # println(dx, (c1-2*c2+c3))
        return (dx^2)/(c1-2*c2+c3)
    end
end

function prop_p2z(p; delLog=delLog)
    return delLog*log(10)*(10^(pix_offset*delLog))*(10^(p*delLog))
end

function get_z_unc(chisq_surface, min_index; pixel_scan_range = pixel_scan_range)
    pixel_var = err1d(pixel_scan_range, chisq_surface, min_index, stepx=5)
    # println(pixel_var)
    if pixel_var > 0
        zerr = prop_p2z(pixel_scan_range[min_index], delLog=delLog).*sqrt(pixel_var)
        return zerr
    end
    return NaN
end

function scan_target_Cres_redo(spec, ivar, padded_Vmat, Vsky, z_to_test; 
        wave_range = 1:length(log_wave_range), fine_scan = true)
    numspec = size(spec)[2]
    num_wave_bins = length(wave_range)
    chisq = zeros(length(z_to_test), numspec)
    spec[isnan.(spec)] .= 0
    M_length = num_sky_eigs + num_lae_eigs
    target_variances = 1 ./ ivar
    num_z = length(z_to_test)
    new_zs = zeros(numspec)
    min_chisqs = zeros(numspec)
    z_uncertainties = zeros(numspec)
    # W = zeros(num_wave_bins, M_length)
    # W[:, 1:num_sky_eigs] = Vsky
    # W = hcat(Vsky, Vlae) # preallocate this
    # AinvW = hcat(AinvV, Cresinv * Vlae)
    AinvW = zeros(num_wave_bins, M_length)
    # wave_range = 1:length(log_wave_range)
    WtAinvD = zeros(M_length)
    sky_length = M_length - num_lae_eigs + 1
    Vsky = Vsky[wave_range, :]
    num_fine_in_pixel = 20
    p = plot()
    # @showprogress for j in 1:numspec
    @showprogress for j in 1:1
        Cresinv = inv(Diagonal(target_variances[wave_range,j]))
        d = spec[wave_range, j]

        CresinvV = Cresinv * Vsky
        
        middle_term = (I + (Vsky' * CresinvV))
        @time Ainv = (Cresinv - (CresinvV * (middle_term \ CresinvV')))
        # for i in 1:num_z
        for i in 1:1
            @time Vlae = shifted_lae_templates[wave_range,:,i] # the rest of this template is zero
                       
            @time AinvVlae = Ainv * Vlae 
            @time this_middle_term = (I + (Vlae' * AinvVlae))
            @time VtAinvD = AinvVlae' * d 
            
            @time chisq[i,j] = VtAinvD' * (this_middle_term \ VtAinvD) # A \ B = inv(A) * B 
        end
        if fine_scan
            min_chisq, best_index = findmin(chisq[:,j])
            # println(best_index, " ", z_to_test[best_index])
            fine_index = (best_index - 1) * num_fine_in_pixel + 1
            new_chisqs = zeros(4*num_fine_in_pixel + 1)
            min_index = max(fine_index - 2*num_fine_in_pixel, 1)
            max_index = min(fine_index + 2*num_fine_in_pixel, 45001)
            # fine scan
            for (k, z) in enumerate(fine_z_to_test[min_index:max_index])
                Vlae = fine_shifted_lae_templates[wave_range,:,(min_index + k - 1)] # the rest of this template is zero

                AinvVlae = Cresinv * Vlae # 8 μs
                AinvW[:,sky_length:M_length] = AinvVlae # 9 μs
                cross_mat = (Vsky' * AinvVlae) # (Vsky' * Cresinv * Vlae) # 145 μs
                M[sky_length:M_length,1:num_sky_eigs] = cross_mat' # 7 μs
                M[1:num_sky_eigs, sky_length:M_length] = cross_mat # 3 μs
                M[sky_length:M_length, sky_length:M_length] = (I + (Vlae' * AinvVlae)) # (I + (Vlae' * Cresinv * Vlae)) # 23 μs

                # @time WtAinvD = AinvW' * d # V' Ainv' D
                WtAinvD[sky_length:M_length] = AinvVlae' * d # 5 μs 
                chisq2 = WtAinvD' * (M \ WtAinvD) # A \ B = inv(A) * B # 29 μs
                new_chisqs[k] = chisq1 - chisq2
            end
            # println(new_chisqs)
            if j==2
                plot!(p, new_chisqs)
            end
            min_chisqs[j], best_index = findmin(new_chisqs)
            z_uncertainties[j] = get_z_unc(new_chisqs, best_index, pixel_scan_range=fine_pixel_scan_range[min_index:max_index] ./ 10)
            new_zs[j] = (fine_z_to_test[min_index:max_index])[best_index]
            
        else
            min_chisqs, min_chisq_indexes = findmin(chisq, dims=1)
            min_chisqs = reshape(collect(min_chisqs), numspec);

            new_zs = zeros(numspec)

            for (i, index_pair) in enumerate(min_chisq_indexes)
                z_index = index_pair[1]
                new_zs[i] = z_to_test[z_index]
            end
    
        end
    end

    return chisq, min_chisqs, new_zs, z_uncertainties, p
end


function multiply_three_things(A, B, C)
    a, b = size(A)
    _, c = size(C)
    D = zeros(a, c)
    @tturbo for i in 1:a, j in 1:c, k in 1:b
            D[i,j] += A[i,k] * B[k] * C[k,j]
            # D[i,j] += C[k,i] * B[k] * C[k,j]
    end
    return D
end

function scan_target_Cres_turbo(spec, ivar, shifted_lae_templates, fine_shifted_lae_templates, Vsky, z_to_test; 
        wave_range = 1:length(log_wave_range), fine_scan = true)
    numspec = size(spec)[2]
    num_wave_bins = length(wave_range)
    chisq = zeros(length(z_to_test), numspec)
    spec[isnan.(spec)] .= 0
    M_length = num_sky_eigs + num_lae_eigs
    # target_variances = 1 ./ ivar # update this
    num_z = length(z_to_test)
    new_zs = zeros(numspec)
    min_chisqs = zeros(numspec)
    z_uncertainties = zeros(numspec)
    AinvW = zeros(num_wave_bins, M_length)
    # wave_range = 1:length(log_wave_range)
    WtAinvD = zeros(M_length)
    sky_length = M_length - num_lae_eigs + 1
    Vsky = Vsky[wave_range, :]
    num_fine_in_pixel = 10
    @showprogress for j in 1:numspec
    # @showprogress for j in 1:1
        Cresinv = Diagonal(ivar[wave_range,j])

        # Cresinv = inv(Cres)
        # d = reshape(spec[wave_range, j], 5001,1)
        d = spec[wave_range, j]
        # CresinvV = Cresinv * Vsky
        
        # middle_term = (I + (Vsky' * CresinvV))
        # Msinv = inv(cholesky(Symmetric(I + Vsky'*(Cres\Vsky))))
        Msinv = inv(cholesky(Symmetric(I + multiply_three_things(Vsky', diag(Cresinv), Vsky))))
        
        StCinvX = Vsky'*(Cresinv*d)
        # for (i, z) in enumerate(z_to_test) # course
        for i in 1:num_z
        # for i in 1:1
            Vlae = shifted_lae_templates[wave_range,:,i] # the rest of this template is zero
                       
            StCinvV = multiply_three_things(Vsky', diag(Cresinv), Vlae)
            VtAinvV = multiply_three_things(Vlae', diag(Cresinv), Vlae) - StCinvV'*Msinv*StCinvV
            XtAinvV = d'*(Cresinv*Vlae) - StCinvX'*Msinv*StCinvV

            # XtAinvV = multiply_three_things(d', diag(Cresinv), Vlae) - StCinvX'*Msinv*StCinvV
            chisq[i,j] = -(XtAinvV * inv(I + VtAinvV) * XtAinvV')
        end
        if fine_scan
            min_chisq, best_index = findmin(chisq[:,j])
            # println(best_index, " ", z_to_test[best_index])
            fine_index = (best_index - 1) * num_fine_in_pixel + 1
            new_chisqs = zeros(10*num_fine_in_pixel + 1)
            min_index = max(fine_index - 5*num_fine_in_pixel, 1)
            max_index = min(fine_index + 5*num_fine_in_pixel, 45001)
            # fine scan
            for (k, z) in enumerate(fine_z_to_test[min_index:max_index])
                Vlae = fine_shifted_lae_templates[wave_range,:,(min_index + k - 1)] # the rest of this template is zero
                
                StCinvV = multiply_three_things(Vsky', diag(Cresinv), Vlae)

                VtAinvV = multiply_three_things(Vlae', diag(Cresinv), Vlae) - StCinvV'*Msinv*StCinvV

                XtAinvV = d'*(Cresinv*Vlae) - StCinvX'*Msinv*StCinvV

                # XtAinvV = multiply_three_things(d', diag(Cresinv), Vlae) - StCinvX'*Msinv*StCinvV
                new_chisqs[k] = -(XtAinvV * inv(I + VtAinvV) * XtAinvV')
            end
            # println(new_chisqs)
            min_chisqs[j], best_index = findmin(new_chisqs)
            z_uncertainties[j] = get_z_unc(new_chisqs ./ 2, best_index, pixel_scan_range=fine_pixel_scan_range[min_index:max_index])
            try
                new_zs[j] = (fine_z_to_test[min_index:max_index])[best_index]
            catch
                println(min_index, max_index, best_index)
            end
            
        else
            min_chisqs, min_chisq_indexes = findmin(chisq, dims=1)
            min_chisqs = reshape(collect(min_chisqs), numspec);

            new_zs = zeros(numspec)

            for (i, index_pair) in enumerate(min_chisq_indexes)
                z_index = index_pair[1]
                new_zs[i] = z_to_test[z_index]
            end
    
        end
    end

    return chisq, min_chisqs, new_zs, z_uncertainties
end