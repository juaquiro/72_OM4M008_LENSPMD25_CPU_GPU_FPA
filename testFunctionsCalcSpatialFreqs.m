%% Test script for spatial-frequency estimation functions
% This script generates a synthetic interferogram (fringe pattern) with a
% known XY carrier and computes the corresponding ground-truth spatial
% frequency maps:
%   - w_phi  : local angular spatial frequency (rad/px)
%   - phi_x  : phase gradient along x (rad/px)
%   - phi_y  : phase gradient along y (rad/px)
%   - theta_or : fringe orientation (rad)
%
% The script then applies a set of spatial-frequency estimation methods
% (e.g., calcSpatialFreqsFilterbank, calcSpatialFreqsHilbert2D) defined in
% the methods_SF array, compares their estimates with the ground truth, and
% reports the recovery error statistics (mean and standard deviation) as
% well as a robust estimate of the execution time for each method
% (mean and standard deviation over multiple runs).
%
% It also illustrates how to:
%   - work with an ROI mask M for valid pixels
%   - compute error histograms for the estimated quantities
%   - organize the numerical results in MATLAB tables for neat display.
%
%% Clear workspace and close figures
close all
clearvars

%% igram

Caso=1;
switch Caso
    case 1 %HIFREQ
        NR=558; NC=553;
        [x, y]=meshgrid(1:NC, 1:NR); x=x-0.5*NC; y=y-0.5*NR;

        %weights
        M=abs(x+1i*y)<sqrt(NR*NC)/3;
        %M=ones(NR, NC);

        %spatial carrier
        w0=[pi/4, pi/4]; %rad/px
        fringe_carrier_or=atan(w0(2)/w0(1));

        %peaks
        phi_0=3*peaks(max(NR, NC));
        phi=imresize(phi_0, [NR, NC]) + w0(1)*x+w0(2)*y;

        %ground truth igram spatial freqs
        [phi_x, phi_y]=gradient(phi); %rad/px
        w_phi=abs(phi_x+1i*phi_y).*M; %rad/px
        theta_or=atan2(-phi_y, phi_x); % fringe orientation [0,pi]

        % DC and modulation in GV
        b=100; m=64;
        % noise level in GV
        nL=10;
        % igram in GV [0 255]
        g=round(b+m*cos(phi))+nL*randn(size(x));
        g=g.*M;
    otherwise
        error('bad selection')
end


%% calculate spatial freqs

% methods list, all have at least g, and M params
methods_SF = [
    % wTh default: 5
    % wmin default: wTh
    % wmax default: 0.25 * mean(size(g))
    % Dw default: 1
    % freqUnits default: "interpFreq"
    % calcMethod default: "interpFreq"
    % filterFlag default: true
    struct("fun_name","filterbank", ...
    "fun", @(g,M) calcSpatialFreqsFilterbank(g,M))

    % wTh default 3     
    struct("fun_name","hilbert2D", ...
    "fun", @(g,M) calcSpatialFreqsHilbert2D(g,M, "filter_orientation", fringe_carrier_or))
    ];

results_SF = struct([]);
nTimingRuns = 10;  % number of repetitions per method for timing

for k = 1:numel(methods_SF)
    fprintf("Running: %s\n", methods_SF(k).fun_name);

    elapsedRuns = zeros(nTimingRuns, 1);

    % Repeat the call several times to obtain a robust timing estimate
    for iRun = 1:nTimingRuns
        tStart = tic;
        [pred_w_phi, pred_theta_or, pred_phi_x, pred_phi_y, M_proc] = methods_SF(k).fun(g, M);
        elapsedRuns(iRun) = toc(tStart);
    end

    % Store results from the last run
    results_SF(k).pred_w_phi   = pred_w_phi;
    results_SF(k).pred_theta_or = pred_theta_or;
    results_SF(k).pred_phi_x   = pred_phi_x;
    results_SF(k).pred_phi_y   = pred_phi_y;
    results_SF(k).M_proc       = M_proc;

    % Store timing statistics
    results_SF(k).elapsedTimeMean_s = mean(elapsedRuns);
    results_SF(k).elapsedTimeStd_s  = std(elapsedRuns);

    % Store method name for reference
    results_SF(k).fun_name = methods_SF(k).fun_name;
end

%% Display results

%ground truth
figure; imagesc(g); colorbar; title(['IGRAM'], 'Interpreter', 'none'); figure(gcf)
figure; imagesc(w_phi); colorbar; title(['w_\phi GROUND TRUTH rad/px ']); figure(gcf)
figure; imagesc(phi_x); colorbar; title(['\phi_x GROUND TRUTH rad/px ']); figure(gcf)
figure; imagesc(phi_y); colorbar; title(['\phi_y GROUND TRUTH rad/px ']); figure(gcf)

% Histograms on valid pixels
histEdge1=linspace(-pi, pi, 100);
figure; histogram(w_phi(M), histEdge1); title('hist(w_\phi) GROUND TRUTH rad/px'); xlabel('w_\phi rad/px')
figure; histogram(theta_or(M), histEdge1); title('hist(\theta) GROUND TRUTH rad/px'); xlabel('\theta rad/px')
figure; histogram(phi_x(M), histEdge1); title('hist(\phi_x) GROUND TRUTH rad/px'); xlabel('\phi_x rad/px')
figure; histogram(phi_y(M), histEdge1); title('hist(\phi_y) GROUND TRUTH rad/px'); xlabel('\phi_y rad/px')

 
% predicted values
for k = 1:numel(methods_SF)

    fprintf("Plotting error results for: %s\n", methods_SF(k).fun_name);

    pred_w_phi=results_SF(k).pred_w_phi;
    pred_phi_x=results_SF(k).pred_phi_x;
    pred_phi_y=results_SF(k).pred_phi_y;
    pred_theta_or=results_SF(k).pred_theta_or;
    M_proc=results_SF(k).M_proc;
    fig_info=char(sprintf("%s, NOISE nL=%d", results_SF(k).fun_name, nL) );
    r=250; c=250;

    % w_phi
    figure; imagesc(pred_w_phi); colorbar; title(['w_\phi rad/px' fig_info]); figure(gcf)
    figure; plot(1:NC, pred_w_phi(r, :), 1:NC, w_phi(r, :)); legend({fig_info, 'GT'}); title(['w_\phi ' fig_info ' row ' num2str(r)]);
    figure; plot(1:NR, pred_w_phi(:, c), 1:NR, w_phi(:, c)); legend({fig_info, 'GT'}); title(['w_\phi ' fig_info ' col ' num2str(c)]);

    %phi_x
    figure; imagesc(pred_phi_x); colorbar; title(['\phi_x rad/px' fig_info]); figure(gcf)
    figure; plot(1:NC, pred_phi_x(r, :), 1:NC, phi_x(r, :)); legend({fig_info, 'GT'}); title(['\phi_x ' fig_info ' row ' num2str(r)]);
    figure; plot(1:NR, pred_phi_x(:, c), 1:NR, phi_x(:, c)); legend({fig_info, 'GT'}); title(['\phi_x ' fig_info ' col ' num2str(r)]);

    %phi_y
    figure; imagesc(pred_phi_y); colorbar; title(['\phi_y rad/px' fig_info]); figure(gcf)
    figure; plot(1:NC, pred_phi_y(r, :), 1:NC, phi_y(r, :)); legend({fig_info, 'GT'}); title(['\phi_x ' fig_info ' row ' num2str(r)]);
    figure; plot(1:NR, pred_phi_y(:, c), 1:NR, phi_y(:, c)); legend({fig_info, 'GT'}); title(['\phi_x ' fig_info ' col ' num2str(r)]);


    % error histograms
    histEdge2 = linspace(-pi/10, pi/10, 100);

    error_map_phi_x     = phi_x(M_proc)     - pred_phi_x(M_proc);
    error_map_phi_y     = phi_y(M_proc)     - pred_phi_y(M_proc);
    error_map_w_phi     = w_phi(M_proc)     - pred_w_phi(M_proc);
    error_map_theta_or  = theta_or(M_proc)  - pred_theta_or(M_proc);

    % mean and std of errors
    mean_phi_x     = mean(error_map_phi_x, 'omitnan');
    std_phi_x      = std(error_map_phi_x,  'omitnan');

    mean_phi_y     = mean(error_map_phi_y, 'omitnan');
    std_phi_y      = std(error_map_phi_y,  'omitnan');

    mean_w_phi     = mean(error_map_w_phi, 'omitnan');
    std_w_phi      = std(error_map_w_phi,  'omitnan');

    mean_theta_or  = mean(error_map_theta_or, 'omitnan');
    std_theta_or   = std(error_map_theta_or,  'omitnan');


    figure; histogram(error_map_phi_x, histEdge2); title(['hist(error \phi_x) ' fig_info]); xlabel('\phi_x rad/px')
    figure; histogram(error_map_phi_y, histEdge2); title(['hist(error \phi_y) ' fig_info]); xlabel('\phi_y rad/px')
    figure; histogram(error_map_w_phi, histEdge2); title(['hist(error w_\phi) ' fig_info]); xlabel('w_\phi rad/px')
    figure; histogram(error_map_theta_or, histEdge2); title(['hist(error \theta) ' fig_info]); xlabel('\theta rad/px')

    % Create a table for neat display, including timing information
    errorStats = table( ...
        [mean_phi_x;     mean_phi_y;     mean_w_phi;     mean_theta_or], ...
        [std_phi_x;      std_phi_y;      std_w_phi;      std_theta_or], ...
        repmat(results_SF(k).elapsedTimeMean_s, 4, 1), ...
        repmat(results_SF(k).elapsedTimeStd_s, 4, 1), ...
        'VariableNames', {'Mean', 'Std', 'ElapsedTimeMean_s', 'ElapsedTimeStd_s'}, ...
        'RowNames', {'phi_x', 'phi_y', 'w_phi', 'theta_or'} ...
        );

    disp('Error statistics and timing for method:');
    disp(results_SF(k).fun_name);
    disp(errorStats)
end

