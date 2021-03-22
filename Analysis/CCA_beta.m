%% BETA
%display(D.data)
 %          EEG: [64×750×4×40 double] : [channels x signal x blocks x characters]
 %          suppl_info: [1×1 struct]

	

% Preprocessing 

%BETA PARAMETERS
dataset = "BETA";
supp_freqs = [8.60000000000000,8.80000000000000,9,9.20000000000000,9.40000000000000,9.60000000000000,9.80000000000000,10,10.2000000000000,10.4000000000000,10.6000000000000,10.8000000000000,11,11.2000000000000,11.4000000000000,11.6000000000000,11.8000000000000,12,12.2000000000000,12.4000000000000,12.6000000000000,12.8000000000000,13,13.2000000000000,13.4000000000000,13.6000000000000,13.8000000000000,14,14.2000000000000,14.4000000000000,14.6000000000000,14.8000000000000,15,15.2000000000000,15.4000000000000,15.6000000000000,15.8000000000000,8,8.20000000000000,8.40000000000000];
subban_no = 3;
signal_length = 1;
totalsubject=70;
totalblock=4;
totalcharacter=40;
sampling_rate=250;
visual_latency=0.13;
visual_cue=0.5;
%sample_length=sampling_rate*signal_length; 
total_ch=64;
max_epochs=800;
dropout_second_stage=0.7;

%CHANNEL PARAMETERS

total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
channels=[48 54 55 56 57 58 61 62 63];% Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
% To use all the channels set channels to 1:total_ch=64;

%[AllData,y_AllData]=PreProcess(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset);

%% -AllData: Preprocessing the data with bandpass filter/s,
%					Dimension of AllData:
%					(# of channels, # sample length, #subbands,
%					 # of characters, # of blocks, # of subjects)
%         -y_AllData: Labels of characters in AllData					

%% CCA
samp_pts = (50:25:500);
ave_accs = [];

signal_lengths = samp_pts./250;
for idx = 1:length(samp_pts)
    
    sample_length = signal_lengths(idx)*sampling_rate;
    sample_interval = (delay_sample_point+1):delay_sample_point+sample_length;
    [AllData,y_AllData]=PreProcess(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset);

    T = samp_pts(idx);
    S = sampling_rate; %sampling rate = 246
    t = linspace(1/S, T/S, T); %t = 1/S, 2/S, ... , T/S

    total_acc = 0;
    for block_idx = 1:4
        
        for subj = 1:70
            true = 0;
            for char_chosen = 1:40
                X = AllData(:, (1:T), 1,char_chosen , block_idx, subj); %[8 x 750] -Block, subband ve subject secimi?
                %create harmonics given the frequency of the characters
                r_list = [];
                for char = 1:40
                    f = supp_freqs(char); %character frequency
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
            accuracy = true / 40;
            total_acc = total_acc + accuracy;
        end
    end
        ave_acc = total_acc / (70 * 4);
        ave_accs(end+1) = ave_acc;
       
end
%% Plot
disp(ave_accs);
bar( (signal_lengths),ave_accs);
%title("CCA")
xlabel('Time (s)')
ylabel('Accuracy (%)')
grid on

%% Statistical Tests


