%% CCA Statistics on Benchmark


%% Benchmark Dataset

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

visual_latency=0.136; %visual_latency=0.13;
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

load("/Data/Bench/S1.mat")

%[AllData,y_AllData]=Preprocess2(channels,sample_length,sample_interval,subban_no,total_subject,total_block,total_character,sampling_rate,dataset);


