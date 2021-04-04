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

[AllData,y_AllData]=Preprocess2(channels,sample_length,sample_interval,subban_no,total_subject,total_block,total_character,sampling_rate,dataset);

%% CCA Stats
%{
subject = 1;
block = 1;
target = 1;
nameofdata=strcat('Data/', dataset ,'/s',num2str(subject),'.mat');
data=load(nameofdata);
data = data.data;

%}

rank = 0;

means_r = [];
means_a =[];
means_b = [];
vars_r = [];
cov_a = [];
cov_b = [];
traces_a = [];
traces_b = [];
N = 40*6*35; %8400

samp_pts = (50:50:1250);

signal_lengths = samp_pts./250;
for idx = 1:length(samp_pts)
    sample_length = signal_lengths(idx)*sampling_rate;
    sample_interval = (delay_sample_point+1):delay_sample_point+sample_length;
    [AllData,y_AllData]=Preprocess2(channels,sample_length,sample_interval,subban_no,total_subject,total_block,total_character,sampling_rate,dataset);
    
    
    T = samp_pts(idx);
    
    S = sampling_rate; %sampling rate = 250
    t = linspace(1/S, T/S, T); %t = 1/S, 2/S, ... , T/S

    max_as = [];
    max_bs =[];
    max_rs = [];
    for subj = 1:total_subject
        for char_chosen = 1:total_character
            for block_chosen = 1:total_block
                %X -> channels, datapoints in time T, bandpass = 1, target, block, subject                
                    X = AllData(:, (1:T), 1,char_chosen , block_chosen, subj); %[8 x 750] -Block, subband ve subject secimi?
                    %create harmonics given the frequency of the characters
                    r_list = [];
                    A_list = [];
                    B_list = [];
                    
                    if rank == 0
                        f = char_freqs(char_chosen); %character frequency
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


                            [A,B,r,U,V,stats] = canoncorr(X',Y');
                            r_list(end+1) = r(1);
                            A_list = [A_list A(:,1)];
                            B_list = [B_list B(:,1)];
                        
                    else  
                        for target = 1:total_character
                            %create harmonics given the frequency of the characters
                            f = char_freqs(target); %character frequency
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


                            [A,B,r,U,V,stats] = canoncorr(X',Y');
                            r_list(end+1) = r(1);
                            A_list = [A_list A(:,1)];
                            B_list = [B_list B(:,1)];
                        end
                    end
                    %TODO with rank
                    max_cor = max(r_list);
                    found_char = find(r_list==max_cor); %return the index with maximum corr coeffient (r)
                    max_a = A_list(:,found_char);
                    max_b = B_list(:,found_char);

                    max_as = [max_as max_a];
                    max_bs = [max_bs max_b];
                    max_rs(end+1) = max(r_list);
            end
        end
    end
    
    means_r(end+1) = mean(max_rs);
    vars_r(end+1) = var(max_rs);

    %center matrices
    t = size(max_as);
    rows = t(1);
    for i = 1:rows
        cent_as(i,:) = max_as(i,:) - mean(max_as(i,:));
        cent_bs(i,:) = max_bs(i,:) - mean(max_bs(i,:));
    end

    cov_as = cent_as * cent_as' / N;
    cov_bs = cent_bs * cent_bs' / N;


    trace_as = trace(cov_as);
    trace_bs = trace(cov_bs);
    traces_a = [traces_a trace_as];
    traces_b = [traces_b trace_bs];

    means_a = [means_a mean(mean(max_as))];
    means_b = [means_b mean(mean(max_bs))];
end

%%
set(gcf,'color','w');
set(gca, 'FontName', 'Arial','FontSize', 12);
plot( samp_pts./250,vars_r,'-o','LineWidth',2);

xlabel('Time (s)')
ylabel('Variance of R')

%%
set(gcf,'color','w');
set(gca, 'FontName', 'Arial','FontSize', 12);
plot( samp_pts./250,means_r,'-o','LineWidth',2);

xlabel('Time (s)')
ylabel('Mean of R')

%%
set(gcf,'color','w');
set(gca, 'FontName', 'Arial','FontSize', 12);
plot( samp_pts./250,traces_a,'-o','LineWidth',2);

xlabel('Time (s)')
ylabel('Trace of A (W1)')

%%
set(gcf,'color','w');
set(gca, 'FontName', 'Arial','FontSize', 12);
plot( samp_pts./250,traces_b,'-o','LineWidth',2);

xlabel('Time (s)')
ylabel('Trace of B (W2)')

%% 
set(gcf,'color','w');
set(gca, 'FontName', 'Arial','FontSize', 12);
plot( samp_pts./250,means_b,'-o','LineWidth',2);

xlabel('Time (s)')
ylabel('Mean of B (W2)')

%%
set(gcf,'color','w');
set(gca, 'FontName', 'Arial','FontSize', 12);
plot( samp_pts./250,means_a,'-o','LineWidth',2);

xlabel('Time (s)')
ylabel('Mean of A (W1)')

%% Means and Traces of A-B
set(gcf,'color','w');
set(gca, 'FontName', 'Arial','FontSize', 12);
A = plot( samp_pts./250,means_a,'-o','LineWidth',2);
hold on
B = plot( samp_pts./250,means_b,'-+','LineWidth',2);
C = plot( samp_pts./250,traces_a,'-x','LineWidth',2);
D = plot( samp_pts./250,traces_b,'-^','LineWidth',2);
hold off
legend('Mean of A (W1)','Mean of B (W2)', 'Trace of A (W1)','Trace of B (W2)')
legend boxoff;

%% Traces of A-B
set(gcf,'color','w');
set(gca, 'FontName', 'Arial','FontSize', 12);
C = plot( samp_pts./250,traces_a,'-o','LineWidth',2);
hold on
D= plot( samp_pts./250,traces_b,'-^','LineWidth',2);
hold off
legend( [C, D],'Trace of A (W1)','Trace of B (W2)')
legend boxoff;


%% 

C = plot(traces_a, means_a);
hold on
D= plot( traces_b,means_b);
E= plot( traces_a,means_r);
F= plot( traces_b,means_r);
hold off
xlabel("Trace");
legend( [C, D,E, F],'Mean of A (W1)','Mean of B (W2)', "Mean of r (wrt Trace of A)", "Mean of r (wrt Trace of B)")

