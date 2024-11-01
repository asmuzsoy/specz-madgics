const delLog = 0.5e-4

using LinearAlgebra, Interpolations, Random, LsqFit, SparseArrays

# Author: Andrew Saydjari
function find_yinx(x::AbstractVector,y::AbstractVector)
    lx = length(x)
    ly = length(y)
    out = zeros(Int,ly)
    @inbounds for i=1:length(y)
        ind = searchsortedfirst(x,y[i])
        if ind == 1
            out[i] = ind
        elseif ind > lx
            out[i] = ind-1
        else
            if abs(x[ind]-y[i]) < abs(x[ind-1]-y[i])
                out[i] = ind
            else
                out[i] = ind-1
            end
        end
    end
    return out
end

# Author: Andrew Saydjari
function returnWeights_inv(obsCoordall::AbstractVector,obsBitMsk::Vector{Int},pixindx::AbstractVector,targVal::Float64,cindx::Int;
        kernsize::Int=4,linFallBack::Bool=true)
    obslen = length(obsCoordall)
    if obslen == 0 #when is this happening and why?
        return zeros(Int,2*kernsize), NaN*ones(2*kernsize)
    end
    diffwav = diff(obsCoordall[maximum([1,(cindx-1)]):minimum([(cindx+1),obslen])])
    diffpixind = diff(pixindx[maximum([1,(cindx-1)]):minimum([(cindx+1),obslen])])
    pscale = minimum(abs.(diffwav./diffpixind))
    offset = (obsCoordall[cindx].-targVal)/pscale
    cbit = obsBitMsk[cindx]
    if (cbit .& 2^1)!=0
        cchip = 1
    elseif (cbit .& 2^2)!=0
        cchip = 2
    elseif (cbit .& 2^3)!=0
        cchip = 3
    else
        cchip = 0
        # println("NO CHIP?")
    end

    indvec = (-kernsize:kernsize) .+ cindx
    offvec = (-kernsize:kernsize) .+ offset
    msk = (1 .<= indvec .<= obslen) # within bounds range
    msk .&= (-kernsize .<= offvec .<= kernsize) # within kernel bounds
    indvecr = indvec[msk]

    mskb = ((obsBitMsk[indvecr] .& 2^cchip).!=0) # same chip mask
    mskb .&= ((obsBitMsk[indvecr] .& 2^4).==0) #bad pix mask
    mskb .&= (-kernsize .<= (pixindx[indvecr].- pixindx[cindx]) .<= kernsize)
    indvecrr = indvecr[mskb]
    
    if offset == 0
        return cindx, 1.0
    end
    if (count(mskb) >= 2*kernsize)
        if (maximum(diff((pixindx[indvecr].- pixindx[cindx])[mskb]))==1)
            wvec = Interpolations.lanczos.(offvec[msk][mskb],kernsize)
            return indvecrr, wvec
        end
    end
    if linFallBack & (count(mskb) > 1)
        pixOffset = pixindx[indvecrr].- pixindx[cindx]
        obsOffset = offvec[msk][mskb]
        lindx = findlast(obsOffset.<=0)
        rindx = findfirst(obsOffset.>=0)
        if !isnothing(lindx) & !isnothing(rindx)
            if (pixOffset[lindx]>=-1) & (pixOffset[rindx]<=1)
                offlst = offvec[msk][mskb]
                totoff = offlst[rindx].-offlst[lindx]
                return [indvecrr[lindx],indvecrr[rindx]],[1-abs(offlst[lindx])/totoff, 1-abs(offlst[rindx])/totoff]
            end
        end
    end
    return zeros(Int,2*kernsize), NaN*ones(2*kernsize)
end

# Author: Andrew Saydjari
function generateInterpMatrix_sparse_inv(waveobs::AbstractVector,obsBitMsk::Vector{Int},wavemod::AbstractVector,pixindx::AbstractVector;kernsize::Int=4,linFallBack::Bool=true)
    obslen = length(waveobs)
    modlen = length(wavemod)
    cindx = find_yinx(waveobs,wavemod)
    row, col, val = Int[], Int[], Float64[]
    for (modind, modval) in enumerate(wavemod)
        indxvec, wvec = returnWeights_inv(waveobs,obsBitMsk,pixindx,modval,cindx[modind],kernsize=kernsize,linFallBack=linFallBack)
        if !isnan(wvec[1]) .& (wvec[1].!=1.0)
            wvec ./= sum(wvec)
            nz = (wvec.!=0)
            push!(row, (modind.*ones(Int,length(indxvec[nz])))...)
            push!(col, indxvec[nz]...)
            push!(val, wvec[nz]...)
        end
    end
    return sparse(row,col,val,modlen,obslen)
end

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
#     itp = Interpolations.interpolate(spec, Interpolations.Lanczos(kernelsize))
#     etp = Interpolations.extrapolate(itp, 0) # fill value is zero

#     # these are the wavelength values of log spaced bins
#     real_linear_log_range = 10 .^ (real_log_range)

#     # Lanczos doesn't want to take in x points, it just does it based on indexing
#     # Calculating which indexes these wavelengths would correspond to
#     index_range = ((real_linear_log_range .- minimum(wave)) / (wave[2] - wave[1])) .+ 1 # because one indexed

#     log_binned_spectrum = etp(index_range)
    # return log_binned_spectrum
    xobs = wave
    xtarg = 10 .^ log_wave_range
    y = spec
    obsBit = ones(Int,length(xobs))*2^1;
    Rinv = generateInterpMatrix_sparse_inv(xobs,obsBit,xtarg,1:length(xobs))
    y_log = Rinv*y;
    return y_log
end

"""Function that converts linearly spaced spectra to log spaced with cubic spline interpolation

Parameters
----------
spec : :class:Matrix
    spectra in linearly spaced wavelength bins
wave : :class:Vector
    corresponding linearly spaced wavelength bins (must be equal sized bins)
real_log_range : :class:Vector
    desired log spaced wavelength bins
Returns
-------
Log spaced spectrum (flux values)
"""
function linear_to_log_cubic(spec, wave, log_wave_range)
    interp_cubic = Interpolations.cubic_spline_interpolation(wave, spec, extrapolation_bc = Throw())
    return interp_cubic(10 .^ log_wave_range)
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

function separate_components(spectrum, redshift, Csky, Cres, Vlae; wave_range = 1:length(spectrum), linear = false)
    adjV, _  = adjust_V(padded_Vmat, redshift)[1];
    temp_Clae = adjV * adjV';
    temp_Csky = Csky[wave_range, wave_range]
    temp_Cres = Cres[wave_range, wave_range];
    temp_Ctotinv = inv(temp_Clae + temp_Csky + temp_Cres);

    sky = temp_Csky * temp_Ctotinv * log_spec[wave_range, index]
    res = temp_Cres * temp_Ctotinv * log_spec[wave_range, index]
    lae = temp_Clae * temp_Ctotinv * log_spec[wave_range, index];
    
    if !linear
       p = plot(log_wave_range[wave_range],  [i for i in [sky, lae, res, log_spec[wave_range, index]]], layout = (4, 1),  title=["sky" "LAE" "residual" "total"], 
    legend=:none, xlabel = ["" "" "" "log(wavelength (Å))"], 
    ylabel="Flux",size=(600,500)) 
        return p
    end
    linear_range = 3800:800:8000
    wavelength_ticks = log10.(linear_range)
    tick_labels = string.(linear_range)
    p = plot(log_wave_range[wave_range],  [i .* sqrt_sky_poly[wave_range] for i in [sky, lae, res, log_spec[wave_range, index]]], layout = (4, 1),  title=["sky residual" "LAE" "residual" "total"], 
        legend=:none, xlabel = ["" "" "" "Wavelength (Å)"], 
        ylabel="Flux",size=(600,500), xticks=(wavelength_ticks, tick_labels), xtickfontsize=12, labelfontsize=16)
    return p
end

function get_wave_index(cutoff)
   return partialsortperm(abs.(log_wave_range .- cutoff), 1) 
end


function get_snr(noisy_spectrum, z)
    lya_wave = log10(get_wavelength(z)) 
    cutoff1 = lya_wave - 0.02
    cutoff2 = lya_wave + 0.03
    index1 = get_wave_index(cutoff1) # take noise of spectrum far from lya line
    index2 = get_wave_index(cutoff2)
    spectrum_except_lya = vcat(noisy_spectrum[1:index1], noisy_spectrum[index2:end])
    sigma = std(spectrum_except_lya)
    lya_start = get_wave_index(lya_wave - 0.0015)
    lya_end = get_wave_index(lya_wave + 0.0015)

    signal = sum(noisy_spectrum[lya_start:lya_end])
    noise = sqrt(lya_end - lya_start) .* sigma

    return signal / noise
end