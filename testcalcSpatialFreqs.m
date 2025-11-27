%% test 1 for the calcSpatialFreqs function
% Here we set up a igram with XY carrier and compare with the ground truth values
% the estimated w, wx and wy using calcSpatialFreqs
% also we show how to make change units between 'rad/px' and 'ff'

%% clear all close all
close all
clear all

%% igram

Caso=1;
switch Caso
    case 1 %HIFREQ
        NR=558; NC=553;
        [x, y]=meshgrid(1:NC, 1:NR); x=x-0.5*NC; y=y-0.5*NR;

        %weights
        M=abs(x+1i*y)<sqrt(NR*NC)/3;

        %spatial carrier
        w0=[pi/4, pi/5]; %rad/px

        %peaks
        phi=2*peaks(max(NR, NC));
        phi=imresize(3*phi, [NR, NC]) + w0(1)*x+w0(2)*y;

        %ground truth igram spatial freqs
        [phi_x, phi_y]=gradient(phi);
        phi_theta=atan2(-phi_y, phi_x); % fringe orientation
        
        b=100; m=64; nL=50;
        g=round(b+m*cos(phi))+nL*randn(size(x));
        g=g.*M;

        %calcSpatialFreqs freq params
        wTh=10; %HIGH FREQ
        wmin=30; %ff
        wmax=180; %ff
        Dw=1; %ff      

    otherwise
        error('bad selection')
end

%calculate spatial freqs
sfreqUnits="rad/px";
calcMethod="interpFreq";
%[pred_w_phi, pred_phi_x, pred_phi_y, M]=calcSpatialFreqsFilterbank(g, M, wTh, wmin, wmax, Dw, sfreqUnits, calcMethod);
[pred_w_phi, pred_phi_x, pred_phi_y, M_proc]=calcSpatialFreqsFilterbank(g, M);

pred_w_phi=pred_w_phi.*M;    pred_w_phi=medfilt2(pred_w_phi, [5,5]);
pred_phi_x=pred_phi_x.*M;  pred_phi_x=medfilt2(pred_phi_x, [5,5]);
pred_phi_y=pred_phi_y.*M;  pred_phi_y=medfilt2(pred_phi_y, [5,5]);

%% display
%units conversion factor
C=1;
switch sfreqUnits
    case "rad/px"
        Cx=1;
        Cy=1;
    case "ff"
        Cy=NR/(2*pi);
        Cx=NC/(2*pi);
    otherwise
        error(['OM4M:' funcName ':invalid funit'], funits);
end

% ground truth values for comparison
wxt=Cx*phi_x;
wyt=Cy*phi_y;
w_phi=abs(wxt+1i*wyt).*M;


srcTitle=['spatialFreq in ' char(sfreqUnits) ' '];

figure; imagesc(g); colorbar; title(['IGRAM'], 'Interpreter', 'none'); figure(gcf)

figure; imagesc(pred_w_phi); colorbar; title(['w Radial filters ESTIMATION ' srcTitle], 'Interpreter', 'none'); figure(gcf)
figure; imagesc(w_phi); colorbar; title(['w GROUND TRUTH ' srcTitle], 'Interpreter', 'none'); figure(gcf)
r=250;
figure; plot(1:NC, pred_w_phi(r, :), 1:NC, w_phi(r, :)); legend({'RF', 'GT'}); title(['w ' srcTitle 'row ' num2str(r)], 'Interpreter', 'none');
c=250;
figure; plot(1:NR, pred_w_phi(:, c), 1:NR, w_phi(:, c)); legend({'RF', 'GT'}); title(['w ' srcTitle 'col ' num2str(c)], 'Interpreter', 'none');

figure; imagesc(pred_phi_x); colorbar; title(['wx Gausian filters ESTIMATION ' srcTitle], 'Interpreter', 'none'); figure(gcf)
figure; imagesc(wxt); colorbar; title(['wx GROUND TRUTH ' srcTitle], 'Interpreter', 'none'); figure(gcf)
r=250;
figure; plot(1:NC, pred_phi_x(r, :), 1:NC, wxt(r, :)); legend({'RF', 'GT'}); title(['wx ' srcTitle 'row ' num2str(r)], 'Interpreter', 'none');
c=250;
figure; plot(1:NR, pred_phi_x(:, c), 1:NR, wxt(:, c)); legend({'RF', 'GT'}); title(['wx ' srcTitle 'col ' num2str(r)], 'Interpreter', 'none');


figure; imagesc(pred_phi_y); colorbar; title(['wy Gausian filters ESTIMATION ' srcTitle], 'Interpreter', 'none'); figure(gcf)
figure; imagesc(wyt); colorbar; title(['wy GROUND TRUTH ' srcTitle], 'Interpreter', 'none'); figure(gcf)
r=250;
figure; plot(1:NC, pred_phi_y(r, :), 1:NC, wyt(r, :)); legend({'RF', 'GT'}); title(['wy ' srcTitle 'row ' num2str(r)], 'Interpreter', 'none');
c=250;
figure; plot(1:NR, pred_phi_y(:, c), 1:NR, wyt(:, c)); legend({'RF', 'GT'}); title(['wy ' srcTitle 'col ' num2str(r)], 'Interpreter', 'none');


% Histograms on valid pixels
histEdge1=linspace(-pi, pi, 100);
figure; histogram(w_phi(M_proc), histEdge1); title('hist(w_\phi)'); xlabel('w_\phi rad/px')
figure; histogram(phi_theta(M_proc), histEdge1); title('hist(\theta)'); xlabel('\theta rad/px')
figure; histogram(phi_x(M_proc), histEdge1); title('hist(\phi_x)'); xlabel('\phi_x rad/px')
figure; histogram(phi_y(M_proc), histEdge1); title('hist(\phi_y)'); xlabel('\phi_y rad/px')

histEdge2=linspace(-pi/10, pi/10, 100);
figure; histogram(phi_x(M_proc)-pred_phi_x(M_proc), histEdge2); title('hist(error \phi_x)'); xlabel('\phi_x rad/px')
figure; histogram(phi_y(M_proc)-pred_phi_y(M_proc), histEdge2); title('hist(error \phi_y)'); xlabel('\phi_y rad/px')
figure; histogram(w_phi(M_proc)-pred_w_phi(M_proc), histEdge2); title('hist(error w_\phi)'); xlabel('w_\phi rad/px')
%figure; histogram(phi_theta(M_proc)-pred_phi_theta(M_proc), histEdge2); title('hist(error \theta)'); xlabel('\theta rad/px')
