%% Statistical Tests

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
samp_pts = (50:25:500);  %(50:25:500);
ave_accs = [];

signal_lengths = samp_pts./250;
means_r = [];
means_a =[];
means_b = [];
vars_r = [];
traces_a = [];
traces_b = [];
rank = 2;

N = 1120; %40*4*70


for idx = 1:length(samp_pts)
    sample_length = signal_lengths(idx)*sampling_rate;
    sample_interval = (delay_sample_point+1):delay_sample_point+sample_length;
    [AllData,y_AllData]=PreProcess(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset);

    T = samp_pts(idx);
    S = sampling_rate; %sampling rate = 246
    t = linspace(1/S, T/S, T); %t = 1/S, 2/S, ... , T/S

    total_acc = 0;
    
    max_as = [];
    max_bs =[];
    max_rs = [];
    for char_chosen = 1:40
        for block_idx = 1:4
            for subj = 1:70
                true = 0;

                %for char_chosen = 1:40
                    X = AllData(:, (1:T), 1,char_chosen , block_idx, subj); %[9 x 750] -Block, subband ve subject secimi?
                    %create harmonics given the frequency of the characters
                    r_list = [];
                    a_list = [];
                    b_list = [];
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
                            cos(10*pi*f*t)];  %y(t) 10 x 750 

                       % Y= transpose(Y); disp(size(Y));
                       [A,B,r,U,V,stats] = canoncorr(X',Y');
                       r_list(end+1) = r(rank); %get the rank'th best one 
                       a_list = [a_list A(:,rank)];
                       b_list = [b_list B(:,rank)];


                    end
                    max_cor = max(r_list);
                    found_char = find(r_list==max_cor); %return the index with maximum corr coeffient (r)
                    max_a = a_list(:,found_char);
                    max_b = b_list(:,found_char);

                    max_as = [max_as max_a];
                    max_bs = [max_bs max_b];
                    max_rs(end+1) = max(r_list);
            end
        end
    end
        means_r(end+1) = mean(max_rs);
        vars_r(end+1) = var(max_rs);
        
        %center matrices
        for i = 1:n
        cent_as(:,i) = max_as(:,i) - mean(max_as(:,i));
        cent_bs(:,i) = max_bs(:,i) - mean(max_bs(:,i));
        end
        
        cov_as = cent_as * cent_as' / N;
        cov_bs = cent_bs * cent_bs' / N;
       
        
        trace_as = trace(cov_as);
        trace_bs = trace(cov_bs);
        traces_a = [traces_a trace_as];
        traces_b = [traces_b trace_bs];
        
        means_a = [means_a mean(mean(max_as))]; %mean(mean()) ??
        means_b = [means_b mean(mean(max_bs))];
 
end
%%
set(gcf,'color','w');
plot( samp_pts./250,vars_r);

xlabel('Time (s)')
ylabel('Variance of R')

%%
plot( samp_pts./250,means_r);

xlabel('Time (s)')
ylabel('Mean of R')

%%
plot( samp_pts./250,traces_a);

xlabel('Time (s)')
ylabel('Trace of A (W1)')

%%
plot( samp_pts./250,traces_b);

xlabel('Time (s)')
ylabel('Trace of B (W2)')

%% 
plot( samp_pts./250,means_b);

xlabel('Time (s)')
ylabel('Mean of B (W2)')

%%
plot( samp_pts./250,means_a);

xlabel('Time (s)')
ylabel('Mean of A (W1)')

%% Means and Traces of A-B

A =plot( samp_pts./250,means_a);
B =plot( samp_pts./250,means_b);
C = plot( samp_pts./250,traces_a);
D= plot( samp_pts./250,traces_b);
legend([A, B, C, D],{ 'Mean of A (W1)','Mean of B (W2)', 'Trace of A (W1)','Trace of B (W2)'})

%% Traces of A-B

C = plot( samp_pts./250,traces_a);
hold on
D= plot( samp_pts./250,traces_b);
hold off
legend( [C, D],'Trace of A (W1)','Trace of B (W2)')

%% 

C = plot(traces_a, means_a);
hold on
D= plot( traces_b,means_b);
E= plot( traces_a,means_r);
F= plot( traces_b,means_r);
hold off
xlabel("Trace");
legend( [C, D,E, F],'Mean of A (W1)','Mean of B (W2)', "Mean of r (wrt Trace of A)", "Mean of r (wrt Trace of B)")




