% giraycoskun@sabanciuniv.edu
%{
 AIM:
 Plot ACG FFT for each target and channel 
%}

channel = 56;
frequencyList = [15, 15.2, 15.4, 15.6, 15.8];
targetList = [8,16,24,32,40];
for index = 1:length(frequencyList)
    subplot(5,1,index);
    subjectAvgSignalFFT(channel, targetList(index), 1);
    title(['Single-Sided Amplitude Spectrum of X(t), Character Index = ',num2str(targetList(index)), ', Flickering Frequency (Hz) = ', num2str(frequencyList(index))]);
end
