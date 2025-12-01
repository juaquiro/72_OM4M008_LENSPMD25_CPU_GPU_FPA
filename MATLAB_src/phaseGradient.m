function [phi_x, phi_y, M_proc] = phaseGradient(z, M, NS, Nmed, LPCycles)
%PHASEGRADIENT Phase-gradient estimation from complex phasor data.
%
%   [phi_x, phi_y, M_proc] = PHASEGRADIENT(z, M) computes the phase
%   gradients (phi_x, phi_y) of a complex phasor field
%       z = b .* exp(1i * phi)
%   using centered finite differences and optional spatial filtering.
%   The sign convention is the same as MATLAB's GRADIENT function.
%
%   [phi_x, phi_y, M_proc] = PHASEGRADIENT(z, M, NS, Nmed, LPCycles)
%   allows control over the filtering parameters:
%
%     z        - complex phasor array b .* exp(1i * phi)
%     M        - ROI mask with valid points (logical or numeric)
%     NS       - half-size of the low-pass box filter window
%                (filter size is (2*NS+1) x (2*NS+1), default: 5)
%     Nmed     - median filter window size for phi_x and phi_y
%                (square window [Nmed Nmed], default: 2; 0 disables)
%     LPCycles - number of times the low-pass filter is applied
%                to the derivatives and mask (default: 2)
%
%   Outputs:
%     phi_x  - phase gradient along x in px^-1
%     phi_y  - phase gradient along y in px^-1
%     M_proc - ROI mask with valid centered differences
%
%   Note:
%     Centered differences are used, so the practical limit for the
%     phase variation is pi/2 rad/px instead of pi rad/px.
%
%   Example:
%     [phi_x, phi_y, Mproc] = phaseGradient(z, roiMask, 5, 2, 2);
%
%   See also ANGLE, CONV2, MEDFILT2, GRADIENT.

    % ------------------------
    % Input parsing & defaults
    % ------------------------
    arguments
    z (:,:) {mustBeNumeric} % phasor z=b*exp(1i*phi)
    M (:,:) {mustBeNumericOrLogical} %M ROI with valid points
    NS (1,1) {mustBeNumeric} = 5 % 2*NS+1 is the neigbouhoord size for phasor filtering
    Nmed (1,1) {mustBeNumeric} = 2 % median filter size for phase only cosine-sine filtering
    LPCycles (1,1) {mustBeNumeric} = 2 % number of low pass cycles that we apply
	end

    % force mask as logical
    if ~islogical(M)
        M = (M ~= 0);
    end

    % ------------------------
    % Basic size and indexing
    % ------------------------
    [NR, NC] = size(z);

    % x-indices for centered first difference
    A = [2:NC, NC];
    B = [1, 1:NC-1];

    % y-indices for centered first difference
    C = [2:NR, NR];
    D = [1, 1:NR-1];

    % Set borders to zero in the ROI mask (no differences on image border)
    M(:, 1)   = false;
    M(1, :)   = false;
    M(NR, :)  = false;
    M(:, NC)  = false;

    % ------------------------
    % Centered phase differences
    % ------------------------

    % dx: compute centered phase difference along x
    zd = z(:, A) ./ z(:, B);
    zd(isnan(zd)) = 0;           % avoid NaNs from zero division
    phi_x = 0.5 .* angle(zd);    % centered difference -> 0.5*angle

    % dy: compute centered phase difference along y
    zd = z(C, :) ./ z(D, :);
    zd(isnan(zd)) = 0;
    phi_y = 0.5 .* angle(zd);

    % ------------------------
    % Optional median filtering
    % ------------------------
    % If the phasor is well sampled, derivatives should be continuous;
    % median filtering removes isolated outliers without blurring too much.
    if Nmed > 0
        phi_x = medfilt2(phi_x, [Nmed, Nmed]);
        phi_y = medfilt2(phi_y, [Nmed, Nmed]);
    end

    % ------------------------
    % Valid-difference mask
    % ------------------------
    % A pixel has a valid centered difference only if it and all its
    % neighbors involved in the finite difference are valid in M.
    M_proc = M(:, A) & M(:, B) & M & M(C, :) & M(D, :);

    % ------------------------
    % Low-pass smoothing
    % ------------------------
    % Smooth derivatives and mask: derivatives should be relatively smooth
    % if the phasor is well sampled. The box filter size is (2*NS+1)^2.
    hLP = ones(2*NS + 1) / (2*NS + 1)^2;

    for n = 1:LPCycles
        phi_x  = conv2(phi_x,  hLP, 'same');
        phi_y  = conv2(phi_y,  hLP, 'same');
        % Smooth the mask as well, then threshold later
        M_proc = conv2(double(M_proc), hLP, 'same');
    end

    % Threshold the smoothed mask back to logical
    M_proc = (M_proc > 0.999);

    % ------------------------
    % Apply ROI to gradients
    % ------------------------
    phi_x = M_proc .* phi_x;
    phi_y = M_proc .* phi_y;

end
