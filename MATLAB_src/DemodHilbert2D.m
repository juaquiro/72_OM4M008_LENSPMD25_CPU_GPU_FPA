function z = DemodHilbert2D(g, M, opts)
%DEMODHILBERT2D Demodulate 2D interferogram using a steerable Hilbert filter
%
%   z = DEMODHILBERT2D(g)
%   z = DEMODHILBERT2D(g, M)
%   z = DEMODHILBERT2D(g, M, wTh=3, filter_orientation=0)
%
%   Performs 2D Hilbert demodulation of an interferogram using a
%   steerable half-plane Hilbert filter in the Fourier domain.
%
%   INPUTS (positional)
%     g   - Input interferogram (2D numeric array).
%     M   - (Optional) ROI mask, same size as g.
%           Default: ones(size(g)).
%
%   KEYWORD ARGUMENTS (Name-Value pairs)
%     wTh               - Spatial frequency threshold (Gaussian HPF).
%                         Removes frequencies below wTh in the Fourier domain.
%                         Default: 3.
%
%     filter_orientation - Orientation of the 2D steerable Hilbert filter [rad].
%                         0     → horizontal-sensitive
%                         pi/2  → vertical-sensitive
%                         Default: 0.
%
%   OUTPUT
%     z   - Complex analytic signal:   z ≈ m * exp(1i*phi)
%
%   NOTES
%     Hilbert filter:   H = (u*cos(θ) + v*sin(θ)) > 0
%     High-pass filter: H1 = 1 − exp(−½*(|q|/wTh)²)
%
%   EXAMPLE
%     z = DemodHilbert2D(g, [], wTh=4, filter_orientation=pi/2);
%
%   AQ 11-APR-2024
% ----------------------------------------------------------------------

arguments
    g (:,:) {mustBeNumeric}
    M (:,:) {mustBeNumericOrLogical} = ones(size(g))
    opts.wTh (1,1) {mustBeNumeric} = 3
    opts.filter_orientation (1,1) {mustBeNumeric} = 0
end

% Extract keyword arguments:
wTh               = opts.wTh;
filter_orientation = opts.filter_orientation;

% Ensure mask exists
if isempty(M)
    M = ones(size(g));
end

% Apply ROI mask when needed
if ~all(M(:) == 1)
    g = g .* M;
end

%% Spatial frequency grid (centered FFT coordinates)
[NR, NC] = size(g);

[u, v] = meshgrid(1:NC, 1:NR);
u = u - (floor(NC/2) + 1);
v = v - (floor(NR/2) + 1);

q = abs(u + 1i*v);   % frequency magnitude

%% Gaussian high-pass filter
H1 = 1 - exp(-0.5 * (q / wTh).^2);

%% Steerable Hilbert filter (half-plane)
H = (u * cos(filter_orientation) + v * sin(filter_orientation)) > 0;

%% Demodulation
G = fft2(g);
G = G .* ifftshift(H .* H1);
z = 2 * ifft2(G);     % analytic signal m·exp(j·phi)
