function p=normprb(d,m,s)
p=1/(s*sqrt(2*pi))*exp(-((d-m).^2)/(2*s^2));
% p(d/m>1)=1/(s*sqrt(2*pi));
end