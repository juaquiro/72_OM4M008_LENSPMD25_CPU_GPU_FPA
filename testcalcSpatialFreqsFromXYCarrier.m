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

        %carrier
        w0=[pi/4, pi/5]; %rad/px

        %peaks
        p=2*peaks(max(NR, NC));
        p=imresize(3*p, [NR, NC]) + w0(1)*x+w0(2)*y;

        b=100; m=64; nL=50;
        g=round(b+m*cos(p))+nL*randn(size(x));
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
[w, wx, wy, M]=calcSpatialFreqsFilterbank(g, M, wTh, wmin, wmax, Dw, sfreqUnits, calcMethod);

w=w.*M;    w=medfilt2(w, [5,5]);
wx=wx.*M;  wx=medfilt2(wx, [5,5]);
wy=wy.*M;  wy=medfilt2(wy, [5,5]);

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
[px, py]=gradient(p);
wxt=Cx*px;
wyt=Cy*py;
wt=abs(wxt+1i*wyt).*M;


srcTitle=['spatialFreq in ' char(sfreqUnits) ' '];

figure; imagesc(g); colorbar; title(['IGRAM'], 'Interpreter', 'none'); figure(gcf)

figure; imagesc(w); colorbar; title(['w Radial filters ESTIMATION ' srcTitle], 'Interpreter', 'none'); figure(gcf)
figure; imagesc(wt); colorbar; title(['w GROUND TRUTH ' srcTitle], 'Interpreter', 'none'); figure(gcf)
r=250;
figure; plot(1:NC, w(r, :), 1:NC, wt(r, :)); legend({'RF', 'GT'}); title(['w ' srcTitle 'row ' num2str(r)], 'Interpreter', 'none');
c=250;
figure; plot(1:NR, w(:, c), 1:NR, wt(:, c)); legend({'RF', 'GT'}); title(['w ' srcTitle 'col ' num2str(c)], 'Interpreter', 'none');

figure; imagesc(wx); colorbar; title(['wx Gausian filters ESTIMATION ' srcTitle], 'Interpreter', 'none'); figure(gcf)
figure; imagesc(wxt); colorbar; title(['wx GROUND TRUTH ' srcTitle], 'Interpreter', 'none'); figure(gcf)
r=250;
figure; plot(1:NC, wx(r, :), 1:NC, wxt(r, :)); legend({'RF', 'GT'}); title(['wx ' srcTitle 'row ' num2str(r)], 'Interpreter', 'none');
c=250;
figure; plot(1:NR, wx(:, c), 1:NR, wxt(:, c)); legend({'RF', 'GT'}); title(['wx ' srcTitle 'col ' num2str(r)], 'Interpreter', 'none');


figure; imagesc(wy); colorbar; title(['wy Gausian filters ESTIMATION ' srcTitle], 'Interpreter', 'none'); figure(gcf)
figure; imagesc(wyt); colorbar; title(['wy GROUND TRUTH ' srcTitle], 'Interpreter', 'none'); figure(gcf)
r=250;
figure; plot(1:NC, wy(r, :), 1:NC, wyt(r, :)); legend({'RF', 'GT'}); title(['wy ' srcTitle 'row ' num2str(r)], 'Interpreter', 'none');
c=250;
figure; plot(1:NR, wy(:, c), 1:NR, wyt(:, c)); legend({'RF', 'GT'}); title(['wy ' srcTitle 'col ' num2str(r)], 'Interpreter', 'none');
