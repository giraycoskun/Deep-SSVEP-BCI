%% Unsupervised Method



%{ 
    NOTES:
    The cross-entropy for each pair of output-target elements is calculated as: ce = -t .* log(y).
    The binary cross-entropy expression is: ce = -t .* log(y) - (1-t) .* log(1-y) .

%}

dirname = "ssvep-global-models-02/";
model_name = "main_net_0.2_";  %% SET SIGNAL LENGTH BASED ON MODEL

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

visual_latency=0.14; %Beta -> .13

total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
sample_interval = (delay_sample_point+1):delay_sample_point+sample_length; % Extract desired signal
channels=[48 54 55 56 57 58 61 62 63]; % Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
%channels = (1:64);

%% Preprocess

filename = "preprocessedSignals/" + "signal_length_" + num2str(signal_length) + "_9channels.mat";


%[AllData,y_AllData]=Preprocess2(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset); save(filename,'AllData','y_AllData');

load(filename);

%% Load Global Model

target_subject = 3;
   
filename = dirname + model_name + num2str(target_subject) + ".mat";
load(filename);

net = main_net;

%% Get Softmax Output for first features

softmax_0 = zeros(totalblock*totalcharacter, totalcharacter);
feature_size = 25 * 120;
feature_0 = zeros(totalblock*totalcharacter, feature_size);
sizes = [total_channel, sample_length ,subban_no];

for testblock=1:totalblock
    
    for char=1:totalcharacter

    

        testdata=AllData(:,:,:,char,testblock,target_subject);
        %testdata=reshape(testdata,[sizes(1),sizes(2),sizes(3),totalcharacter]);

        index = (testblock-1)*40 + char;
        X = testdata;
        layer = 11; % 'softmax'
        act = activations(net,X,layer);
        softmax_0(index, :) = act(1,1,:);
        
        layer = 8; % 'conv_4'
        act = activations(net,X,layer);
        feature = reshape(act, [feature_size, 1]);
        feature_0(index, :) = feature;
    end
end

%LAST dimension holds prediction probabilities

%% Calcuate Target Probabilities
% https://www.mathworks.com/help/deeplearning/ref/dlarray.crossentropy.html#mw_e33463eb-140f-4b8d-955c-5130c7d11c83

target_probs = zeros(6, 40, 40);
target_vector = zeros(40, 40);

for char=1:totalcharacter
   target = zeros(1, 40); target(char) = 1;
   target_vector(char, :) = target;
end
for block=1:totalblock
   target_probs(block,:,:) = target_vector;
end

%% Calculate Nearest Neighbors

k = 11;

X = feature_0;

Y = feature_0(41, :);

Idx = knnsearch(X ,Y, 'K', k, 'Distance', 'euclidean');

%% Calculate Cross-Entropy

lambda = 0.5;

%THERE IS ALFA


total_loss = 0;
for block=1:totalblock
    for char=1:totalcharacter
        index = (testblock-1)*40 + char;
        loss = crossentropy(softmax_0(index, :), softmax_0(index, :));
        total_loss = total_loss + loss;
        disp(loss);
    end
end




