% giraycoskun@sabanciuniv.edu
%{
 AIM:
 Take FFT from Average Signals of one subject, one channel, one target 
%}

function [Y] = blockAvgSignalFFT(channel, target, subject, isPlot)
% defined for Benchmark-Dataset
Fs = 250;                    % Sampling frequency
%T = 1/Fs;                     % Sampling period
L = 1500;                     % Length of signal
%t = (0:L-1)*T;                % Time vector
%targetCount = 40;
%channelCount = 64;
blockCount = 6;

dataFileName = ['Data/Benchmark-DataSet/s',num2str(subject),'.mat'];
data=load(dataFileName); % Loading the subject data
data = data.data;


targetBlockSignal = data(channel,:,target, 1:blockCount);
avgTargetSignal = squeeze(mean(targetBlockSignal, 4));


Y = fft(avgTargetSignal);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

if(isPlot == 1)
    subplot(2,1,1); plot(avgTargetSignal);
    subplot(2,1,2); plot(f,P1); 
    title('Single-Sided Amplitude Spectrum of X(t)');
    xlabel('f (Hz)');
    ylabel('|P1(f)|');
end
end