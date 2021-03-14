%% Benchmark

%{
Data epochs were extracted from continuous EEG record- ings according to stimulus onsets from the event channel. 
For each trial, six seconds (including 0.5 s before stimulus onset, 5 s for stimulation, and 0.5 s after stimulus offset) of data were extracted. 
Based on our previous finding that the upper- bound frequency of the SSVEP harmonics in this paradigm is around 90 Hz [19], 
all epochs were simply down-sampled to 250 Hz to reduce storage and computation costs. No digital filters were applied in data preprocessing.
%}

% Number of Subjects: 35
% Data: 64x1500x40x6
% Number of Channels: 64
% Number of Targets: 40
% Number of Blocks: 6

dataset = "Bench";
total_subject=35;
total_block=6;
total_character=40;
total_channel=64;
sampling_rate=250;
	
visual_cue=0.5;
subban_no = 3;
signal_length = 5;
sample_length=sampling_rate*signal_length; 

visual_latency=0.14; %visual_latency=0.13;
%max_epochs=800;
%dropout_second_stage=0.7;

char_freqs = zeros(1, total_character);
idx = 8;
for i = 1:8:40
    char_freqs(i:i+7) = idx:idx+7;
    idx = idx + 0.2;
end


%% Preprocessing 

total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
sample_interval = (delay_sample_point+1):delay_sample_point+sample_length; % Extract desired signal
channels=[48 54 55 56 57 58 61 62 63];% Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
% To use all the channels set channels to 1:total_ch=64;

[AllData,y_AllData]=Preprocess2(channels,sample_length,sample_interval,subban_no,total_subject,total_block,total_character,sampling_rate,dataset);

%{
AllData: Preprocessing the data with bandpass filter/s,
					Dimension of AllData:
					(# of channels, # sample length, #subbands,
					 # of characters, # of blocks, # of subjects)
y_AllData: Labels of characters in AllData	

Benchmark-Dataset 
AllData: 9x1250x3x40x6x35
y_AllData: 1x40x4x35
%}

%% CCA
%accuracy plot
samp_pts = (50:50:1250);
%accs = [];
ave_accs = [];
for idx = 1:length(samp_pts)
 
    T = samp_pts(idx);
    
    S = sampling_rate; %sampling rate = 250
    t = linspace(1/S, T/S, T); %t = 1/S, 2/S, ... , T/S

    total_acc = 0;
    for subj = 1:total_subject
        character_accuracy = 0;
            for char_chosen = 1:total_character
                true = 0;
                for block_chosen = 1:total_block
                    %X -> channels, datapoints in time T, bandpass = 1, target, block, subject                
                    X = AllData(:, (1:T), 1,char_chosen , block_chosen, subj); %[8 x 750] -Block, subband ve subject secimi?
                    %create harmonics given the frequency of the characters
                    r_list = [];
                    for char = 1:total_character
                        f = char_freqs(char); %character frequency
                        Y = [sin(2*pi*f*t);
                            cos(2*pi*f*t); 
                            sin(4*pi*f*t);
                            cos(4*pi*f*t);
                            sin(6*pi*f*t);
                            cos(6*pi*f*t);
                            sin(8*pi*f*t);
                            cos(8*pi*f*t);
                            sin(10*pi*f*t);
                            cos(10*pi*f*t)];  %y(t) 6 x 750 

                       % Y= transpose(Y); disp(size(Y));
                       [A,B,r,U,V,stats] = canoncorr(X',Y');
                       r_list(end+1) = r(1);
                    end
                    max_cor = max(r_list);
                    found_char = find(r_list==max_cor);
                    if (found_char == char_chosen)
                        true = true + 1;
                    end
                end
                accuracy = true / total_block;
                character_accuracy = character_accuracy + accuracy;
            end
        accuracy = character_accuracy / total_character;
        total_acc = total_acc + accuracy;
    end
    ave_acc = total_acc / total_subject;
    ave_accs(end+1) = ave_acc;
end
disp(ave_accs);
x = samp_pts./250;
ln = plot(x, ave_accs);
ln.Marker = 'o';
ln.LineWidth = 2;
title("CCA")
xlabel('Time (s)')
%xticks(samp_pts./250);
ylabel('Accuracy (%)')