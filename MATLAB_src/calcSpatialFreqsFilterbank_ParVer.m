function [w_phi, theta_or, phi_x, phi_y, M_proc] = calcSpatialFreqsFilterbank_ParVer(g, M, wTh, wmin, wmax, Dw, freqUnits, calcMethod, filterFlag)
%CALCSPATIALFREQSFILTERBANK Local spatial frequency estimation for fringe patterns.
%   [w_phi, theta_or, phi_x, phi_y, M_proc] = CALCSPATIALFREQSFILTERBANK_PARVER(g)
%   This is the paralelized version of CALCSPATIALFREQSFILTERBANK(g)
%   estimates the local spatial frequency magnitude w_phi, the fringe
%   orientation theta_or, and the phase-gradient components phi_x and phi_y
%   of the input fringe pattern
%
%       g = a + b * cos(phi),
%
%   using a bank of Gaussian / Gabor-like filters in the spatial-frequency
%   domain. The method is based on the adaptive filtering approach described in:
%
%     J. Vargas, J. A. Quiroga, and T. Belenguer,
%     "Local frequency determination by adaptive filtering,"
%     Opt. Lett. 36, 70–72 (2011).
%
%   This version uses an orthogonal 1D Gabor filter bank along u and v
%   (horizontal and vertical frequency axes), so it works naturally in
%   non-square images and allows an easy conversion to rad/px units.
%
%   The implementation parallelizes the filter bank in two levels:
%     1) All filters are applied in parallel using a 3-D stack of
%        frequency-domain filters (page-wise 2-D IFFTs).
%     2) The local frequency estimation by parabolic interpolation around
%        the maximum response is fully vectorized (no pixel loops).
%
%   Input arguments
%   ---------------
%   g          : 2-D numeric array
%                Input fringe pattern (igram), modeled as g = a + b*cos(phi).
%
%   M          : 2-D numeric or logical array, same size as g
%                ROI mask. Non-zero or true values indicate valid pixels.
%                Default: ones(size(g))  (full image is valid).
%
%   wTh        : scalar numeric
%                Threshold for the spatial-frequency magnitude (in fft
%                frequency units, "ff"). Spatial frequencies below wTh are
%                filtered out. Default: 5.
%
%   wmin       : scalar numeric
%                Minimum spatial frequency (in "ff" units) for the radial
%                filter bank scan. Default: wTh.
%
%   wmax       : scalar numeric
%                Maximum spatial frequency (in "ff" units) for the radial
%                filter bank scan. Default: 0.25*mean(size(g)).
%
%   Dw         : scalar numeric
%                Frequency step (in "ff" units) between successive radial
%                filters in the bank. Default: 1.
%
%   freqUnits  : string scalar
%                Units for the output spatial frequencies. Allowed values:
%                  "ff"      - normalized fft-frequency units
%                  "rad/px"  - radians per pixel (default)
%                Default: "rad/px"
%
%   calcMethod : string scalar
%                Method for estimating the local frequency from the filter
%                responses. Allowed values:
%                  "interpFreq" - parabolic interpolation around the
%                                  maximum filter response (default)
%                  "maxFreq"    - use the discrete frequency of the
%                                  maximum response.
%
%   filterFlag : logical scalar or numeric convertible to logical
%                If true, applies additional filtering/smoothing to the
%                estimated gradient components (phi_x, phi_y) inside the
%                ROI. If false, returns the raw estimates.
%                Default: true.
%
%   Output arguments
%   ----------------
%   w_phi      : 2-D numeric array, same size as g
%                Local spatial frequency magnitude at each pixel, expressed
%                in "ff" units or "rad/px" depending on freqUnits.
%
%   theta_or   : 2-D numeric array, same size as g
%                Local fringe orientation in radians, in [0, pi].
%
%   phi_x      : 2-D numeric array, same size as g
%                Local phase gradient component in x.
%
%   phi_y      : 2-D numeric array, same size as g
%                Local phase gradient component in y.
%
%   M_proc     : 2-D logical array, same size as g
%                Processed mask after optional filtering of the ROI borders.
%
%   ---------------------------------------------------------------------

arguments
    g (:,:) {mustBeNumeric}; % input igram
    M (:,:) {mustBeNumericOrLogical}=ones(size(g)); % input ROI
    wTh (1,1) {mustBeNumeric}=5; % spatial freq modulus threshold in ff
    wmin (1,1) {mustBeNumeric}=wTh; % min spatial freq in ff for scanning
    wmax (1,1) {mustBeNumeric}=0.25*mean(size(g)); % max spatial freq in ff
    Dw (1,1) {mustBeNumeric}=1;  % spatial freq interval for scanning in ff
    freqUnits string {mustBeMember(freqUnits,["ff", "rad/px"])}="rad/px";  % output spatial freqs units
    calcMethod string {mustBeMember(calcMethod,["interpFreq", "maxFreq"])}="interpFreq";  % estimation method
    filterFlag (1,1) {mustBeNumericOrLogical}=true;  % wx and wy filter flag
end

% get igram size
[NR, NC] = size(g);

%% (optional) filter mask borders (currently disabled)
% T=round(mean(NR, NC)/wTh);
% M=conv2(M, ones(T)/T^2, 'same');
% M=2*(M-0.5); M(M<0)=0;
% g=g.*M;

%% spatial freqs: cartesian freq units in "ff"
[u, v] = meshgrid(1:NC, 1:NR);        % cartesian freq space in ff
u0 = floor(NC/2) + 1;  u = u - u0;
v0 = floor(NR/2) + 1;  v = v - v0;
q  = abs(u + 1i*v);                   % radial distance freq space in ff

%% g's FT and HighPass filter
G  = fft2(g);
H1 = 1 - exp(-0.5*(q/wTh).^2);
G  = G .* ifftshift(H1);

%% filter bank (PARALLEL 3-D IMPLEMENTATION)

sw = 10*Dw; % gabor filter width (sigma) (franjas/cpo)
nFilters = round((wmax - wmin)/Dw);

% spatial freqs fringes/field for the gabor filter bank
qList = linspace(wmin, wmax, nFilters);

% --- Parallel 3-D implementation of the Gabor filter bank -----------------
% We build two 3-D stacks of filters in the frequency domain:
%   Hx(:,:,k) peaks at qList(k) along the u-axis (horizontal),
%   Hy(:,:,k) peaks at qList(k) along the v-axis (vertical).
% Then we apply all filters in parallel using page-wise 2-D IFFTs.

qList3 = reshape(qList, 1, 1, []);             % 1 x 1 x nFilters
u3     = repmat(u, 1, 1, nFilters);            % NR x NC x nFilters
v3     = repmat(v, 1, 1, nFilters);            % NR x NC x nFilters

% 3-D Gabor filters in the (u,v) frequency plane
Hx = exp(-0.5 * ((u3 - qList3)/sw).^2);
Hy = exp(-0.5 * ((v3 - qList3)/sw).^2);

% Match fft2 convention: move DC component to (1,1) for each slice
Hx = ifftshift(ifftshift(Hx,1),2);
Hy = ifftshift(ifftshift(Hy,1),2);

% Apply filter bank in parallel (page-wise 2-D IFFT)
G3 = repmat(G, 1, 1, nFilters);                 % NR x NC x nFilters

gHx = ifft2(G3 .* Hx);                          % NR x NC x nFilters
gHy = ifft2(G3 .* Hy);                          % NR x NC x nFilters

% Magnitude of responses; each row of vmgH* corresponds to one filter
mgHx = abs(gHx);
mgHy = abs(gHy);

vmgHx = reshape(permute(mgHx, [3 1 2]), nFilters, NR*NC); % [nFilters, NR*NC]
vmgHy = reshape(permute(mgHy, [3 1 2]), nFilters, NR*NC); % [nFilters, NR*NC]

%AQDEBUG to check for the correlation for a given pixel (r,c)
AQDEBUG = false;
if AQDEBUG
    c    = round(0.5*NC + 10);
    r    = round(0.5*NR);
    indx = sub2ind([NR NC], r, c);
    figure;
    plot(qList, vmgHx(: ,indx), '.-');
    title(['Correlation X for row ' num2str(r) ' col ' num2str(c)]);
end

%% spatial freq estimation

if calcMethod=="maxFreq" % fast but lower precision depending on Dw
    [~, pos] = max(vmgHx, [], 1);
    phi_x    = reshape(qList(pos), NR, NC); % measured spatial x freq
else % "interpFreq" - parabolic interpolation around max (vectorized)
    phi_x = calcFreqFromFilterRespose(M, NR, NC, vmgHx, nFilters, qList);
end

% filter impulsive noise
phi_x = medfilt2(phi_x, [5 5]);

if filterFlag
    % weighted filter
    phi_x(isnan(phi_x)) = 0;
    phaseFactor = 0.01;
    wxQuality   = reshape(std(vmgHx).^2, NR, NC); % the bigger the better
    zx          = wxQuality .* exp(1i*phaseFactor*phi_x); % phasor
    zxf         = conv2(zx, ones(10,10)/100, 'same');     % phasor filtering
    phi_x       = angle(zxf)/phaseFactor;                 % border-resistant
    % filter Mask
    M_proc  = conv2(M, ones(10,10)/100, 'same');
    M_proc  = M_proc>0.999;
else
    M_proc  = logical(M);
end

% check for calcMethod for y
if calcMethod=="maxFreq" % fast but low precision depending on Dw
    [~, pos] = max(vmgHy, [], 1);
    phi_y    = reshape(qList(pos).^2, NR, NC); % measured spatial y freq (as in previous version)
else % "interpFreq" - parabolic interpolation around max (vectorized)
    phi_y = calcFreqFromFilterRespose(M, NR, NC, vmgHy, nFilters, qList);
end

% filter impulsive noise
phi_y = medfilt2(phi_y, [5 5]);

if filterFlag
    % weighted filter
    phi_y(isnan(phi_y)) = 0;
    phaseFactor = 0.01;
    wyQuality   = reshape(std(vmgHy).^2, NR, NC); % the bigger the better
    zy          = wyQuality .* exp(1i*phaseFactor*phi_y); % phasor
    zyf         = conv2(zy, ones(10,10)/100, 'same');     % phasor filtering
    phi_y       = angle(zyf)/phaseFactor;                 % filtered freq
end

% units conversion factor
if freqUnits=="rad/px"
    Cx = 2*pi/NC;  % 1 field NC in px, 1 fringe = 2*pi rad
    Cy = 2*pi/NR;  % 1 field NR in px, 1 fringe = 2*pi rad
    phi_x = phi_x*Cx; % rad/px
    phi_y = phi_y*Cy; % rad/px

    w_phi = abs(phi_x + 1i*phi_y); % rad/px
else
    w_phi = abs(phi_x + 1i*phi_y); % ff
end

% fringe orientation [0, pi]
theta_or = atan2(-phi_y, phi_x); % fringe orientation


% ========================================================================
% Local helper (vectorized): frequency from filter response using
% parabolic interpolation (3-point) around the maximum.
%
% Inputs:
%   M      : NR x NC mask
%   NR,NC  : image size
%   vmgH   : [nFilters x (NR*NC)] magnitudes for the filter bank
%   nFilters: number of filters
%   qList  : 1 x nFilters list of spatial freqs
%
% Output:
%   w      : NR x NC map of interpolated frequency (same units as qList)
% ========================================================================
function w = calcFreqFromFilterRespose(M, NR, NC, vmgH, nFilters, qList)

% Ensure row vector
qList = qList(:).';          % 1 x nFilters
dq    = qList(2) - qList(1); % assumed constant step
q0    = qList(1);

Npix = NR * NC;

% 1) Discrete maximum per pixel (across filters)
[~, k] = max(vmgH, [], 1);        % 1 x Npix, indices in [1 .. nFilters]
k = double(k);

% 2) Neighbour indices (clamped at the borders)
km = max(k-1, 1);
kp = min(k+1, nFilters);

idx0 = sub2ind([nFilters, Npix], k,  1:Npix);
idxm = sub2ind([nFilters, Npix], km, 1:Npix);
idxp = sub2ind([nFilters, Npix], kp, 1:Npix);

y0 = vmgH(idx0);
ym = vmgH(idxm);
yp = vmgH(idxp);

% 3) Parabolic interpolation in index units.
%    Standard formula (assuming unit spacing in index):
%       delta = 0.5 * (ym - yp) / (ym - 2*y0 + yp)
%    where delta is the sub-sample offset from the max index k.
den = (ym - 2*y0 + yp);

% Avoid division by zero: where den ~ 0, set delta = 0 (no interpolation)
tol = 1e-12;
denAbsSmall = abs(den) < tol;
den(denAbsSmall) = 1;   % temporary to avoid NaNs in division
delta = 0.5 * (ym - yp) ./ den;
delta(denAbsSmall) = 0;

% 4) Convert from index to frequency using the linear mapping of qList
kEff      = k + delta;                  % effective (sub-sample) index
qMax_vec  = q0 + (kEff - 1) * dq;       % 1 x Npix, in same units as qList

% 5) Reshape and apply mask
w = reshape(qMax_vec, NR, NC);
w(~M) = NaN;

end % calcFreqFromFilterRespose

end %calcSpatialFreqsFilterbank
