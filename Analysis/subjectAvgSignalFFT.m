% giraycoskun@sabanciuniv.edu
%{
 AIM:
 Take Average from block FFT signals of one channel and one target 
%}

function [Y] = subjectAvgSignalFFT(channel, target, isPlot)

%defined for Benchmark Dataset
subjectCount = 35;
subjectList = (1:subjectCount);
subjectList = subjectList(subjectList ~= 5);

subjectFFT = [];
for s=subjectList
    subjectY = blockAvgSignalFFT(channel, target, s, 0);
    subjectFFT = [subjectFFT; {subjectY}];
    
end
subjectFFT = cell2mat(subjectFFT);
Fs = 250;  L = 1500;
avgFFT = mean(subjectFFT, 1);
Y = avgFFT;
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
if(isPlot == 1)
plot(f,P1); 
title('Single-Sided Amplitude Spectrum of X(t)');
xlabel('f (Hz)');
ylabel('|P1(f)|');
end

%end

