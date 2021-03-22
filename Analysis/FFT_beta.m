%% BETA
%display(D.data)
 %          EEG: [64×750×4×40 double] : [channels x signal x blocks x characters]
 %          suppl_info: [1×1 struct]

fs =250;
channel = 57; % CHANNEL SELECTION

Fs = 250;  L = 1250;
for char = 36:40
    fft_ave = zeros(1,1250);
    fft_s = [];
    for subject=1:70
        nameofdata=['S',num2str(subject),'.mat'];
        D=load(nameofdata); % Loading the subject data
        D=D.data;
        supp_freqs = D.suppl_info.freqs;
        D= D.EEG;	

        ave_signal=zeros(1,L); % Initialization
        for i= 1:(D(channel, : 1, char)) % 
            ave_signal(i) = (( D(channel, i, 1, char)+ D(channel, i, 2, char) + D(channel, i, 3, char)+ D(channel, i, 4, char) ) /4) ;
        end

        Y = fft(ave_signal);
        fft_s = [fft_s; {Y}];
    end

    fft_s = cell2mat(fft_s);

    avgFFT = mean(fft_s, 1);
    Y = avgFFT;
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
    
    char_freq = supp_freqs(char);
    subplot(5,1,(char-35))
    plot(f,P1)
    title('Single-Sided Amplitude Spectrum of X(t), Character Index = '+string(char)+ ", Flickering Frequency (Hz) = "+ string(char_freq) )
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
end

%% deneme

Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);

X = S + 2*randn(size(t));

plot(1000*t(1:50),X(1:50))
title('Signal Corrupted with Zero-Mean Random Noise')
xlabel('t (milliseconds)')
ylabel('X(t)')

Y = fft(X);

P2 = abs(Y/L);
plot(P2)
disp(size(Y));
%% 
P1 = P2(1:L/2+1);
disp(size(P1));
P1(2:end-1) = 2*P1(2:end-1);
disp(size(P1));
f = Fs*(0:(L/2))/L;
disp(size(f));

title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')