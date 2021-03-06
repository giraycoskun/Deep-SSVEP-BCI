%% A network(1) for SSVEP based EEG Signal Character Recognition
%{
https://www.mathworks.com/help/deeplearning/ug/list-of-deep-learning-layers.html
%}

%% Benchmark Dataset

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
signal_length = 0.2;
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

total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
sample_interval = (delay_sample_point+1):delay_sample_point+sample_length; % Extract desired signal
%channels=[48 54 55 56 57 58 61 62 63];% Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
channels = (1:64);
%% Beta Dataset


%% Preprocess
totalsubject = 5;
totalblock = 2;
totalcharacter = 40;
%[AllData,y_AllData]=Preprocess2(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset);
%% NN Architecture

%[AllData,y_AllData]=Preprocess2(channels,sample_length,sample_interval,subban_no,total_subject,total_block,total_character,sampling_rate,dataset);

T = sample_length;
alpha = 10;
subNetworkSize = 4;
channelSize = 64;
d = channelSize;
subbanNum = 3;

sizes = [T, d, subbanNum];
%sizes=size(AllData);

lgraph = layerGraph;

input_layer = imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none','Name','input_layer'); %targetNum, sampleNum, subbanNum
lgraph = addLayers(lgraph, input_layer);

for c = 1:40
    
    
    subbanComb_layer = convolution2dLayer([1,1],1,'WeightsInitializer','ones','Name',['subbanComb_sublayer_',num2str(c)]);
    %lgraph = addLayers(lgraph, input_layer);
    lgraph = addLayers(lgraph, subbanComb_layer);
    
    lgraph = connectLayers(lgraph, 'input_layer', ['subbanComb_sublayer_',num2str(c)]);
    %lgraph = connectLayers(lgraph, ['input_sublayer_',num2str(c)],['subbanComb_sublayer_',num2str(c)]);
    %plot(lgraph);
    
    for divTime=1:subNetworkSize
         
         
        layers = [
    convolution2dLayer([1, d],d/2,'WeightsInitializer','ones', 'Name', ['channelComb', num2str(divTime) ,'_sublayer_',num2str(c)])
    resize3dLayer('OutputSize',[T,d/2,1],'Name',['resize1_', num2str(divTime) ,'_sublayer_',num2str(c)])
    convolution2dLayer([T, 1],alpha*d/2, 'WeightsInitializer','narrow-normal','Name',['inputComb', num2str(divTime) ,'_sublayer_',num2str(c)])
    resize3dLayer('OutputSize',[alpha*d/2, d/2, 1], 'Name',['resize2_', num2str(divTime) ,'_sublayer_',num2str(c)])
    leakyReluLayer('Name',['activationLR', num2str(divTime) ,'_sublayer_',num2str(c)])
            ];
        
        if (divTime == 1)
            T = alpha*d;
        else
            T = T/2;
        end
        
        d = d/2;
        
        lgraph = addLayers(lgraph, layers);
        
    end
    d = channelSize;
    lgraph = connectLayers(lgraph,['subbanComb_sublayer_',num2str(c)] ,['channelComb', num2str(1) ,'_sublayer_',num2str(c)] );
    
    for divTime=1:subNetworkSize-1
        
        lgraph = connectLayers(lgraph, ['activationLR', num2str(divTime) ,'_sublayer_',num2str(c)],['channelComb', num2str(divTime+1) ,'_sublayer_',num2str(c)]);
    end
    
    fcLayer = fullyConnectedLayer(1, 'Name',['fc_layer_sublayer_', num2str(c)]);
    lgraph = addLayers(lgraph,fcLayer);
    lgraph = connectLayers(lgraph,['activationLR', num2str(subNetworkSize) ,'_sublayer_',num2str(c)] ,['fc_layer_sublayer_', num2str(c)]);
    
    
end



concatLayer = concatenationLayer(3,40,'Name','concat_layer');
lgraph = addLayers(lgraph,concatLayer);



%plot(lgraph);
for c = 1:40
 %disp(c)
lgraph = connectLayers(lgraph, ['fc_layer_sublayer_', num2str(c)] , ['concat_layer/in',num2str(c)]);
end

outLayers = [
    softmaxLayer('Name','softMax_layer')
    classificationLayer('Name','classification_layer')
    ];
lgraph = addLayers(lgraph, outLayers);
lgraph = connectLayers(lgraph,'concat_layer', 'softMax_layer');

plot(lgraph);
analyzeNetwork(lgraph);
return

%% Training
max_epochs=500;
acc_matrix=zeros(totalsubject,totalblock); % Initialization of accuracy matrix

allblock=1:totalblock;
%allblock(block)=[]; Exclude the block used for testing     

%layers(2, 1).BiasLearnRateFactor=0; % At first layer, sub-bands are combined with 1 cnn layer, 
% bias term basically adds DC to signal, hence there is no need to use 
% bias term at first layer. Note: Bias terms are initialized with zeros by default.  
train=AllData(:,:,:,:,allblock,:); %Getting training data
train=reshape(train,[sizes(1),sizes(2),sizes(3),totalcharacter*length(allblock)*totalsubject*1]);

train_y=y_AllData(:,:,allblock,:);
train_y=reshape(train_y,[1,totalcharacter*length(allblock)*totalsubject*1]);    
train_y=categorical(train_y);

options = trainingOptions('adam',... % Specify training options for first-stage training
    'InitialLearnRate',0.0001,...
    'MaxEpochs',max_epochs,...
    'MiniBatchSize',100, ...
    'Shuffle','every-epoch',...
    'L2Regularization',0.001,...
    'ExecutionEnvironment','cpu',...
    'Plots','training-progress');    
main_net = trainNetwork(train,train_y,lgraph,options);    
sv_name=['main_net_',int2str(block),'.mat']; 
save(sv_name,'main_net'); % Save the trained model
all_conf_matrix=zeros(40,40); % Initialization of confusion matrix 

%% Notes



