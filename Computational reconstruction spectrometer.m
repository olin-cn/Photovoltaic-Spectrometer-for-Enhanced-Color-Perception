% --------------------- 
% @Description: Spectrum Reconstruction for vdWH
% @Version: 2.0
% @Author: Xiangfu Lei\ Hanxiao Cui
% @Date: 2026-01-29 
% @Autor: Hanxiao Cui
% --------------------- 
clearvars;clc

% load Model for Testing

M=readmatrix('RMatrix_1.xlsx');w0=M(2:end,1);R0=M(2:end,2:end);R0=R0/max(R0(:));
t0=(1:numel(w0))';t=linspace(1,numel(w0),601)';wl=interp1(t0,w0,t,'pchip');R=interp1(t0,R0,t,'pchip');

% load test data for spectrum 
load('narrow360-1000.mat')
lam=700;c=45;r=narrow(:,c);r=r/max(r);
x0=exp(-4*log(2)*((wl-lam)/19.2).^2);x0=x0/max(x0);

%spectrum reconstruction
mu=linspace(0,1,1000);s=0.03/(2*sqrt(2*log(2)));B=exp(-0.5*((wl-wl(1))/(wl(end)-wl(1))-mu).^2/s^2);
A=zeros(size(R,2),size(B,2));for i=1:size(R,2),A(i,:)=trapz(wl,R(:,i).*B);end
D=diff(eye(size(B,2)),1);H=A'*A+1e-6*(D'*D);f=-A'*r;
y=quadprog(H,f,[],[],[],[],zeros(size(B,2),1),[],[],optimoptions('quadprog','Display','off'));
x=B*y;x=max(x,0);x=x/max(x);

% Plots
figure(1);clf
subplot(1,2,1)
pcolor(wl,1:size(R,2),R');shading flat;colormap jet;colorbar
xlabel('Wavelength (nm)');ylabel('No.');title('Response Matrix');pbaspect([1 1 1])

subplot(1,2,2)
plot(wl,x0,'k--','LineWidth',1.2);hold on
plot(wl,x,'r','LineWidth',1.2);grid on;box on
xlabel('Wavelength (nm)');ylabel('Intensity');title('Spectrum Reconstruction')
legend('True','Reconstructed');axis([wl(1) wl(end) -0.05 1.05]);pbaspect([1 1 1])
