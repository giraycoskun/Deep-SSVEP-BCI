%% Replicate Results of Network with Benchmark Dataset

%{
whos -file ssvep-global-models-02/main_net_0.2_1.mat
%}


%% Benchmark Dataset

% Number of Subjects: 35
% Data: 64x1500x40x6
% Number of Channels: 64
% Number of Targets: 40
% Number of Blocks: 6

dataset = "Bench";
totalsubject=35;
totalblock=6;
totalcharacter=40;
total_channel=9; % CHANNEL COUNT 
sampling_rate=250;
	
visual_cue=0.5;
subban_no = 3;
signal_length = 0.2; %%SIGNAL LENGTH
sample_length=sampling_rate*signal_length; 

visual_latency=0.13;

total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
sample_interval = (delay_sample_point+1):delay_sample_point+sample_length; % Extract desired signal
channels=[48 54 55 56 57 58 61 62 63]; % Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
%channels = (1:64);

%% Preprocess

dirname = "preprocessedSignals/";
filename = dirname + "signal_length_" + num2str(signal_length) + "_9channels.mat";


[AllData,y_AllData]=Preprocess2(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset); save(filename,'AllData','y_AllData');

% load(filename);

%% Predict Results

sizes = [total_channel, sample_length ,subban_no];

acc_matrix=zeros(totalsubject,1); % Initialization of accuracy matrix
all_conf_matrix=zeros(40,40); % Initialization of confusion matrix 

for i = 1:totalsubject

    % Load Data -> main_net variable
    target_subject = i;
    dirname = "ssvep-global-models-02/";
    filename = dirname + "main_net_0.2_" + num2str(target_subject) + ".mat";
    load(filename);
    
    % Predict
    block_acc = zeros(6, 1);
    for testblock=1:totalblock
        testdata=AllData(:,:,:,1:40,testblock,target_subject);
        testdata=reshape(testdata,[sizes(1),sizes(2),sizes(3),totalcharacter]);

        test_y=y_AllData(:,1:40,testblock,target_subject);
        test_y=reshape(test_y,[1,totalcharacter*1]);
        test_y=categorical(test_y);

        [YPred,~] = classify(main_net,testdata);
        %disp(YPred);
        acc=mean(YPred==test_y'); disp(acc);
        block_acc(testblock) = acc;
        all_conf_matrix=all_conf_matrix+confusionmat(test_y,YPred);

    end    
    
    acc_matrix(target_subject)=mean(block_acc);
    

end

%% Save Results

dirname = "ssvep-global-models-02/";
sv_name = dirname + 'confusion_matrix.mat';
save(sv_name,'all_conf_matrix');    

sv_name= dirname + 'acc_matrix.mat';
save(sv_name,'acc_matrix'); 

