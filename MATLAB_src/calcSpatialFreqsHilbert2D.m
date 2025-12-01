function [w_phi, theta_or, phi_x, phi_y, M_proc] = calcSpatialFreqsHilbert2D(g, M, opts)
%CALCSPATIALFREQSHILBERT2D Local spatial frequency and fringe orientation (Hilbert-based)
%
%   [w_phi, theta_or, phi_x, phi_y, M_proc] = CALCSPATIALFREQSHILBERT2D(g)
%   [w_phi, theta_or, phi_x, phi_y, M_proc] = CALCSPATIALFREQSHILBERT2D(g, M)
%   [w_phi, theta_or, phi_x, phi_y, M_proc] = CALCSPATIALFREQSHILBERT2D(g, M, ...
%       wTh=5, filter_orientation=0, phasor_filter_size=5)
%
%   Computes the local spatial frequency components (phi_x, phi_y),
%   the spatial frequency magnitude w_phi, and the local fringe
%   orientation theta_or of an interferogram using 2D Hilbert demodulation
%   followed by a phasor-based phase-gradient estimation.
%
%   INPUTS (positional)
%     g   - Input interferogram, 2D numeric array.
%     M   - (Optional) Region of interest mask. 
%           Default: ones(size(g)).
%
%   KEYWORD ARGUMENTS (Name-Value)
%     wTh               - Spatial frequency cutoff for Hilbert demodulation.
%                         Frequencies below wTh (in ff units) are attenuated.
%                         Default: 5.
%
%     filter_orientation - Orientation of the steerable 2D Hilbert filter [rad].
%                         0   → Hilbert filter sensitive to horizontal fringes
%                         pi/2 → Sensitive to vertical fringes
%                         Default: 0.
%
%     phasor_filter_size - Half-window size used in phaseGradient().
%                          Neighborhood = (2*size + 1).
%                          Default: 5.
%
%   OUTPUTS
%     w_phi   - Spatial frequency magnitude |∇φ| in rad/px.
%     theta_or- Fringe orientation angle = atan2(-phi_y, phi_x).
%     phi_x   - Phase gradient component ∂φ/∂x in rad/px.
%     phi_y   - Phase gradient component ∂φ/∂y in rad/px.
%     M_proc  - Processed mask from phaseGradient (validity map).
%
%   EXAMPLE
%     [w_phi, theta_or, phi_x, phi_y, M_proc] = calcSpatialFreqsHilbert2D(g, [], ...
%         wTh=4, filter_orientation=pi/2, phasor_filter_size=7);
%
%   AQ 2024
% ---------------------------------------------------------------------

arguments
    g (:,:)  {mustBeNumeric}
    M (:,:)  {mustBeNumericOrLogical} = ones(size(g))
    opts.wTh (1,1) {mustBeNumeric} = 5
    opts.filter_orientation (1,1) {mustBeNumeric} = 0
    opts.phasor_filter_size (1,1) {mustBeNumeric} = 5
end

% Extract keyword arguments
wTh               = opts.wTh;
filter_orientation = opts.filter_orientation;
phasor_filter_size = opts.phasor_filter_size;

% ---------------------------------------------------------------------
% 1) Hilbert Demodulation: z ≈ m * exp(1i φ)
% ---------------------------------------------------------------------
z = DemodHilbert2D(g, M, wTh=wTh, filter_orientation=filter_orientation);

% ---------------------------------------------------------------------
% 2) Phase-gradient estimation (via phasor filtering)
% ---------------------------------------------------------------------
Nmed     = 2;   % median-filter size (fixed)
LPCycles = 2;   % low-pass cycles (fixed)

[phi_x, phi_y, M_proc] = phaseGradient( ...
    z, M, phasor_filter_size, Nmed, LPCycles);

% ---------------------------------------------------------------------
% 3) Spatial frequency magnitude and fringe orientation
% ---------------------------------------------------------------------
w_phi    = abs(phi_x + 1i * phi_y);        % |∇φ|
theta_or = atan2(-phi_y, phi_x);           % local fringe angle [rad]
