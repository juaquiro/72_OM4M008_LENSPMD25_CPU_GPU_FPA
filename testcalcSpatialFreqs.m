%% test 1 for the calcSpatialFreqs function
% Here we set up a igram with XY carrier and compare with the ground truth values
% the estimated w, phi_x and phi_y using calcSpatialFreqs
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
        %M=ones(NR, NC);

        %spatial carrier
        w0=[pi/4, 0*pi/4]; %rad/px

        %peaks
        phi=2*peaks(max(NR, NC));
        phi=imresize(3*phi, [NR, NC]) + w0(1)*x+w0(2)*y;

        %ground truth igram spatial freqs
        [phi_x, phi_y]=gradient(phi); %rad/px
        w_phi=abs(phi_x+1i*phi_y).*M; %rad/px
        theta_or=atan2(-phi_y, phi_x); % fringe orientation [0,pi]
        
        b=100; m=64; nL=20;
        g=round(b+m*cos(phi))+nL*randn(size(x));
        g=g.*M;

        %calcSpatialFreqs freq params
        % wTh=10; %HIGH FREQ
        % wmin=30; %ff
        % wmax=180; %ff
        % Dw=1; %ff      

    otherwise
        error('bad selection')
end

%calculate spatial freqs
sfreqUnits="rad/px";
% calcMethod="interpFreq";
% %[pred_w_phi, pred_phi_x, pred_phi_y, M]=calcSpatialFreqsFilterbank(g, M, wTh, wmin, wmax, Dw, sfreqUnits, calcMethod);
%[pred_w_phi, pred_theta_or, pred_phi_x, pred_phi_y, M_proc]=calcSpatialFreqsFilterbank(g, M);
[pred_w_phi, pred_theta_or, pred_phi_x, pred_phi_y, M_proc]=calcSpatialFreqsHilbert2D(g, M);


%% display
%units conversion factor
switch sfreqUnits
    case "rad/px"
    case "ff"
        Cy=NR/(2*pi);
        Cx=NC/(2*pi);

        phi_x=Cx*phi_x; %fringes/field
        phi_y=Cy*phi_y; %fringes/field
        w_phi=abs(phi_x+1i*phi_y).*M; %fringes/field
    otherwise
        error(['OM4M:' funcName ':invalid funit'], funits);
end

srcTitle=['spatialFreqs in ' char(sfreqUnits) ' '];

figure; imagesc(g); colorbar; title(['IGRAM'], 'Interpreter', 'none'); figure(gcf)

figure; imagesc(pred_w_phi); colorbar; title(['w_\phi Radial filters ESTIMATION ' srcTitle], 'Interpreter', 'none'); figure(gcf)
figure; imagesc(w_phi); colorbar; title(['w_\phi GROUND TRUTH ' srcTitle], 'Interpreter', 'none'); figure(gcf)
r=250;
figure; plot(1:NC, pred_w_phi(r, :), 1:NC, w_phi(r, :)); legend({'RF', 'GT'}); title(['w ' srcTitle 'row ' num2str(r)], 'Interpreter', 'none');
c=250;
figure; plot(1:NR, pred_w_phi(:, c), 1:NR, w_phi(:, c)); legend({'RF', 'GT'}); title(['w ' srcTitle 'col ' num2str(c)], 'Interpreter', 'none');

figure; imagesc(pred_phi_x); colorbar; title(['\phi_x Gausian filters ESTIMATION ' srcTitle], 'Interpreter', 'none'); figure(gcf)
figure; imagesc(phi_x); colorbar; title(['\phi_x GROUND TRUTH ' srcTitle], 'Interpreter', 'none'); figure(gcf)
r=250;
figure; plot(1:NC, pred_phi_x(r, :), 1:NC, phi_x(r, :)); legend({'RF', 'GT'}); title(['\phi_x ' srcTitle 'row ' num2str(r)], 'Interpreter', 'none');
c=250;
figure; plot(1:NR, pred_phi_x(:, c), 1:NR, phi_x(:, c)); legend({'RF', 'GT'}); title(['\phi_x ' srcTitle 'col ' num2str(r)], 'Interpreter', 'none');


figure; imagesc(pred_phi_y); colorbar; title(['\phi_y Gausian filters ESTIMATION ' srcTitle], 'Interpreter', 'none'); figure(gcf)
figure; imagesc(phi_y); colorbar; title(['\phi_y GROUND TRUTH ' srcTitle], 'Interpreter', 'none'); figure(gcf)
r=250;
figure; plot(1:NC, pred_phi_y(r, :), 1:NC, phi_y(r, :)); legend({'RF', 'GT'}); title(['\phi_y ' srcTitle 'row ' num2str(r)], 'Interpreter', 'none');
c=250;
figure; plot(1:NR, pred_phi_y(:, c), 1:NR, phi_y(:, c)); legend({'RF', 'GT'}); title(['\phi_y ' srcTitle 'col ' num2str(r)], 'Interpreter', 'none');


% Histograms on valid pixels
histEdge1=linspace(-pi, pi, 100);
figure; histogram(w_phi(M_proc), histEdge1); title('hist(w_\phi)'); xlabel('w_\phi rad/px')
figure; histogram(theta_or(M_proc), histEdge1); title('hist(\theta)'); xlabel('\theta rad/px')
figure; histogram(phi_x(M_proc), histEdge1); title('hist(\phi_x)'); xlabel('\phi_x rad/px')
figure; histogram(phi_y(M_proc), histEdge1); title('hist(\phi_y)'); xlabel('\phi_y rad/px')

histEdge2=linspace(-pi/10, pi/10, 100);
figure; histogram(phi_x(M_proc)-pred_phi_x(M_proc), histEdge2); title('hist(error \phi_x)'); xlabel('\phi_x rad/px')
figure; histogram(phi_y(M_proc)-pred_phi_y(M_proc), histEdge2); title('hist(error \phi_y)'); xlabel('\phi_y rad/px')
figure; histogram(w_phi(M_proc)-pred_w_phi(M_proc), histEdge2); title('hist(error w_\phi)'); xlabel('w_\phi rad/px')
figure; histogram(theta_or(M_proc)-pred_theta_or(M_proc), histEdge2); title('hist(error \theta)'); xlabel('\theta rad/px')
