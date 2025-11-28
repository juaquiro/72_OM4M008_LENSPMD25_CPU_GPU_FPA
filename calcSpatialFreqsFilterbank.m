function [w_phi, theta_or, phi_x, phi_y, M_proc]=calcSpatialFreqsFilterbank(g, M, wTh, wmin, wmax, Dw, freqUnits, calcMethod, filterFlag)
% calcSpatialFreqs spatial frequencies estimation for igram g
% [w_phi, theta_or, phi_x, phi_y, M_proc]=calcSpatialFreqsFilterbank(g, M, wTh, wmin, wmax, Dw, freqUnits, calcMethod) computes the module w_phi of the local spatial frequencies vector [phi_x, phi_y]=grad(phi) of the
% input igram g=a+b*cos(phi), in sfreqUnits={"rad_px"(default), "ff" }, filters out the low spatial freqs up to wTh (default 10 ff) and
% also returns the fringe orientation angle [0 pi] theta_or.
% uses nFilters=(wmax-wmin)/Dw radial gausian filters scanning from wmin to
% wmax every Dw all in ff

% Ref: J. Vargas, J. Antonio Quiroga, and T. Belenguer, "Local fringe density determination by adaptive filtering," Opt. Lett. 36, 70-72 (2011)

%   AQ 11APR24
%   Copyright 2009 OM4M

arguments
    g (:,:) {mustBeNumeric}; % input igram
    M (:,:) {mustBeNumericOrLogical}=ones(size(g)); % input ROI
    wTh (1,1) {mustBeNumeric}=5; % statial freq module threshold in ff, filter spatial freqs bellow wTH
    wmin (1,1) {mustBeNumeric}=wTh; % min spatial freq in ff for scanning with the radial filters
    wmax (1,1) {mustBeNumeric}=0.25*mean(size(g)); % max spatial freq in ff for scanning with the radial filters
    Dw (1,1) {mustBeNumeric}=1;  % spatial freq interval for scanning in ff
    freqUnits string {mustBeMember(freqUnits,["ff", "rad/px"])}="rad/px"  % output spatial freqs units
    calcMethod string {mustBeMember(calcMethod,["interpFreq", "maxFreq"])}="interpFreq"  % output spatial freqs units
    filterFlag (1,1) {mustBeNumericOrLogical}=true;  % wx and wy filter flag
end

%get igram size and init wmin and wmax in terms of wTh and size(g)
[NR, NC]=size(g);

% if wmin<0
%     wmin=wTh;
% end
% 
% if wmax<0
%     wmax=0.5*min([NR, NC]);
% end

%% filter mask borders
% T=round(mean(NR, NC)/wTh);
% M=conv2(M, ones(T)/T^2, 'same');
% M=2*(M-0.5); M(M<0)=0;
% g=g.*M;

%% spatial freqs
%cartesian freq units
[u, v]=meshgrid(1:NC, 1:NR); %cartesian freq space in ff
u0=floor(NC/2)+1; u=u-u0;
v0=floor(NR/2)+1;  v=v-v0;
q=abs(u+1i*v); %radial distance freq space in ff

%% g's FT and HighPass filter
G=fft2(g);
H1=1-exp(-0.5*(q/wTh).^2);
G=G.*ifftshift(H1);

%% filter bank

%spatial freq min y max in ff
sw=10*Dw; %gabor filter width (sigma) (franjas/cpo)
nFilters=round((wmax-wmin)/Dw);

%spatial freqs fringes/field for the gabor filter bank
qList = linspace(wmin,wmax,nFilters);

vmgHx=zeros(nFilters, NR*NC);
vmgHy=vmgHx;

for n=1:nFilters
    % this is the Vargas radial implementation but it makes not possible to 
    %change units to rad/px in case of non-square images, so we will
    %implemeneten the method as two orthogonal gabor filter banks
    % H = exp(-0.5*( ((q-qList(n))/sw).^2));
    % gH = ifft2(G.*ifftshift(H));  
    % gHm=abs(gH);
    % mgH(n, :)=gHm(:); [NFilters, NR*NC] each row is a vectorized module abs(q)-filter response to qList(n) freq

    % x-gabor filter bank
    Hx = exp(-0.5*( ((u-qList(n))/sw).^2));
    gHx = ifft2(G.*ifftshift(Hx));

    mgHx=abs(gHx); % module x-filter response
    vmgHx(n, :)=mgHx(:); %[NFilters, NR*NC] each row is a vectorized module x-filter response to qList(n) freq

    % y-gabor filter bank
    Hy = exp(-0.5*( ((v-qList(n))/sw).^2));
    gHy = ifft2(G.*ifftshift(Hy));

    mgHy=abs(gHy); % module y-filter response
    vmgHy(n, :)=mgHy(:); %[NFilters, NR*NC] each row is a vectorized module y-filter response to qList(n) freq
   
end



%AQDEBUG to check for the correlation for a given pixel (r,c)
c=round(0.5*NC+10); r=round(0.5*NR); indx=sub2ind([NR NC], r, c)
plot(qList, vmgHx(: ,indx), '.-')

%% spatial freq estimation

% for the radial implementation we have to look at mgH
% [~, pos] = max(mgH);
% w = reshape(qList(pos),NR,NC);% measured spatial freq in fringes/field

if calcMethod=="maxFreq" %fast but low precission depening on Dw
    [~, pos] = max(vmgHx);
    phi_x = reshape(qList(pos),NR,NC);% measured spatial x freq in fringes/field
else %slow and accurate very independent of Dw
    %calculate wx interpolating maxima for each filter respose
    phi_x = calcFreqFromFilterRespose(M, NR, NC, vmgHx, nFilters, qList);
end

%filter impulsive noise
phi_x=medfilt2(phi_x, [5 5]); 

if filterFlag
    % weigthed filter
    phi_x(isnan(phi_x))=0;
    phaseFactor=0.01;
    wxQuality=reshape(std(vmgHx).^2, NR,NC); %the bigger the better
    zx=wxQuality.*exp(1i*phaseFactor*phi_x); %phasor
    zxf=conv2(zx, ones(10,10)/100, 'same'); %phasor filtering
    phi_x=angle(zxf)/phaseFactor; %filtered freq border resistant

    %filter Mask
    M_proc=conv2(M, ones(10,10)/100, 'same');
    M_proc=M_proc>0.999;
end

% check for calcMethod 
if calcMethod=="maxFreq" %fast but low precission depening on Dw
    [~, pos] = max(vmgHy);
    phi_y = reshape(qList(pos).^2,NR,NC);% measured spatial x freq in fringes/field
else %slow and accurate very independent of Dw
    %calculate wy interpolating maxima for each filter respose
    phi_y = calcFreqFromFilterRespose(M, NR, NC, vmgHy, nFilters, qList);
end

%filter impulsive noise
phi_y=medfilt2(phi_y, [5 5]); 

if filterFlag
    % weigthed filter
    phi_y(isnan(phi_y))=0;
    phaseFactor=0.01;
    wyQuality=reshape(std(vmgHy).^2, NR,NC); %the bigger the better
    zy=wyQuality.*exp(1i*phaseFactor*phi_y); %phasor
    zyf=conv2(zy, ones(10,10)/100,'same'); %phasor filtering
    phi_y=angle(zyf)/phaseFactor; %filtered freq border resistant
end

%units conversion factor
if freqUnits=="rad/px"
    Cx=2*pi/NC;  %1 field NC in px  , 1 fringe =2*pi rad
    Cy=2*pi/NR;  %1 field NR in px  , 1 fringe =2*pi rad
    phi_x=phi_x*Cx; % rad/px
    phi_y=phi_y*Cy; % rad/px

    w_phi=abs(phi_x+1i*phi_y); %rad/px
else
    w_phi=abs(phi_x+1i*phi_y); %ff
end

% fringe orientation [0, pi]
theta_or=atan2(-phi_y, phi_x); % fringe orientation



function w = calcFreqFromFilterRespose(M, NR, NC, vmgH, nFilters, qList)
w=zeros(NR, NC);
for c=1:NC
    for r=1:NR
        if M(r,c)
            n=sub2ind([NR NC], r, c); %pixel index in vmgHx for pixel [r, c]
            if n==283024
                vv=1;
            end

            [~, pos] = max(vmgH(:, n)); %maximum response for pixel n
            indx=pos-3:pos+3; % use 7 points
            validIndx=(indx>0)&(indx<=nFilters); %check for boundaries 1<=index<=NFilters
            indx=indx(validIndx);

            x=qList(indx); %spatial freqs
            y=vmgH(indx, n); %filters response for pixel n arround maximum response

            p=polyfit(x,y, 2);
            a=p(1); b=p(2);
            if a<=0
                qMax=-b/(2*a);
            else
                qMax=nan;
            end

            w(r,c)=qMax;
        else
            w(r,c)=nan;
        end
    end
end
