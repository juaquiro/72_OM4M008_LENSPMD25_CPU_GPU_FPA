function [w_phi, theta, phi_x, phi_y, MH]=calcSpatialFreqsHilbert2D(g, M, wTh, hilbertFilterDirection, filterFreqsFlag, av_filter_size)
% calcSpatialFreqsHilbert2D(g, M, wTh, hilbertFilterDirection) calculates
% the spatial freq module w_phi, fringe orientation theta and its components phi_x and phi_y all in rad/px for
% the input igram g with ROI M, wTH is the threshold to filter out low
% freqs and hilbertFilterDirection is the hilbert filter direction

arguments
    g (:,:) {mustBeNumeric}; % input igram
    M (:,:) {mustBeNumericOrLogical}=ones(size(g)); % input ROI
    wTh (1,1) {mustBeNumeric}=5; % statial freq module threshold in ff, filter spatial freqs bellow wTH
    hilbertFilterDirection string {mustBeMember(hilbertFilterDirection,["X", "Y"])}="X";  % output spatial freqs units
    filterFreqsFlag (1,1) {mustBeNumericOrLogical}=true;  % output spatial freqs filter flag 
    av_filter_size (2,1) {mustBeNumeric}=[15 15]; %averaging filter size for spatial freqs
end

z=DemodHiltert2D(g, M, wTh, hilbertFilterDirection);

NS=round(mean(av_filter_size)); % 2*NS+1 is the neigbouhoord size for phasor filtering
Nmed=3; % median filter size for phase only cosine-sine filtering
LPCycles=3; %low pass cycles
[phi_x, phi_y, MH]=phaseGradient(z, M, NS, Nmed, LPCycles); %rad/px
w_phi=abs(phi_x+1i*phi_y);
theta=atan2(-phi_y, phi_x); % fringe orientation

