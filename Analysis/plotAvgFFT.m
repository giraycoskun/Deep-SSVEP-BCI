% giraycoskun@sabanciuniv.edu
%{
 AIM:
 Plot ACG FFT for each target and channel 
%}

channel = 56;
for target=1:5
    subplot(5,1,target);
    subjectAvgSignalFFT(channel, target, 1);
    title('Single-Sided Amplitude Spectrum of X(t)');
end
