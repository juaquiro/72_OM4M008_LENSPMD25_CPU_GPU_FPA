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
        M=true(NR, NC);

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
        % noise level modulation in GV
        nL_mod=0; 
        nL=100*nL_mod/m; %noise level in % of modulation
        % igram in GV [0 255]
        g=uint8(b+m*cos(phi)+nL_mod*randn(size(x)));
        g=double(g).*M;

        %ground truth phasor
        z=b.*exp(1i*phi);
    otherwise
        error('bad selection')
end

%% demod Hilbert 2D
pred_z=DemodHilbert2D(g, M);
error_map_phi=angle(pred_z./z);
rmse_phi = sqrt(mean(error_map_phi(M).^2));
assert(rmse_phi<0.02)
