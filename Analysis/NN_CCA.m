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

addpath('/cta/users/caksoy/Bench/');
total_subject=35;
total_block=6;
total_character=40;
total_channel=9;
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
%channels = (1:64);
channels=[48 54 55 56 57 58 61 62 63]; %first 8 channels of 9 previously selected channels
%% Beta Dataset


%% Preprocess
totalsubject = 35;
totalblock = 6;
totalcharacter = 40;
[AllData,y_AllData]=Preprocess2(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset);
%% CCA - Take the average of A's and B's over subjects and blocks
%CCA_As Dimensions:
%                   ( # sample length,# of subjects, #subbands,
%					  # of blocks,  # of characters,  9, 9)

%CCA_Bs Dimensions:
%                   ( # sample length,# of subjects, #subbands,
%					  # of blocks,  # of characters, 10, 9)

allblock=1:5; 
sampleLength = sampling_rate * signal_length / 50; %"/50" since the A's and B's are recorded in samp_pts = (50:50:1250)

%average over training blocks and all subjects
As_all =load("CCA_As").As;
As = As_all(:,:,:,allblock,:,:,:);%As with training blocks only
As_ave = As(sampleLength,:,:,:,:,:,:); %Take the corresponding sample_length
As_ave = squeeze(mean(As_ave, [2,4])); %dimension reduced to: subbands x characters x 9 x 9 

%Prepare A's as weight initializer: ch x d x 1 x d x b = 40 x 9 x 1 x 9 x 3
A_weights = zeros(40, 9, 1 , 9 , 3);
%weight dimension -> FilterSize(1)-by-FilterSize(2)-by-NumChannelsPerGroup-by-NumFiltersPerGroup-by-NumGroups
%                 -> 1 x d x 1 x d x b
A = As_ave(:,:,1,:); % 3 x 40 x 1 x 9
A = permute(A, [2, 3, 4, 1]); %40 x 1 x 9 x 3
for ch=1:40
    for d=1:9
    A_weights(ch, d , : , : , :) = A(ch, :,:,:);
    end
end


Bs_all =load("CCA_Bs").Bs;
Bs = Bs_all(:,:,:,allblock,:,:,:);%Bs with training blocks only
Bs_ave = Bs(sampleLength,:,:,:,:,:,:);
Bs_ave = squeeze(mean(Bs_ave, [2,4])); %dimension reduced to: subbands x characters x 10 x 9 



%% NN Architecture

T = sample_length;
alpha = 5;
subNetworkSize = 2;
networkSize = 40;
channelSize = 8;
k_d=total_channel;

d = total_channel-1;
%d = channelSize;
subbanNum = 3;

subbanNum = 3;
sizes = [T, k_d, subbanNum];
%sizes=size(AllData);

lgraph = layerGraph;

input_layer = imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none','Name','input_layer'); %targetNum, sampleNum, subbanNum
lgraph = addLayers(lgraph, input_layer);


for c = 1:networkSize
    k_d=total_channel;
    d = total_channel-1;
    cca_init_layer = groupedConvolution2dLayer([1,total_channel],total_channel,subbanNum,'Weights',A_weights(c,:,:,:,:),'Name',['gconv',num2str(c)] );
    lgraph = addLayers(lgraph, cca_init_layer);
    lgraph = connectLayers(lgraph, 'input_layer', ['gconv',num2str(c)]);
    
    reshape_layer = resize3dLayer('OutputSize',[sample_length,total_channel,subbanNum],'Name',['depthSpace', num2str(c)]);
    lgraph = addLayers(lgraph, reshape_layer);
    lgraph = connectLayers(lgraph, ['gconv',num2str(c)], ['depthSpace', num2str(c)] );
    subbanComb_layer = convolution2dLayer([1,1],1,'WeightsInitializer','ones','Name',['subbanComb_sublayer_',num2str(c)]);
    %lgraph = addLayers(lgraph, input_layer);
    lgraph = addLayers(lgraph, subbanComb_layer);
    
    
    lgraph = connectLayers(lgraph, ['depthSpace',num2str(c)], ['subbanComb_sublayer_',num2str(c)]);
    %lgraph = connectLayers(lgraph, ['input_sublayer_',num2str(c)],['subbanComb_sublayer_',num2str(c)]);
    %plot(lgraph);
    
    T = sizes(1);
    %d = sizes(2);
    
   for divTime=1:subNetworkSize
         
         
        layers = [
   convolution2dLayer([1, k_d],d*2,'WeightsInitializer','he', 'Name', ['channelComb', num2str(divTime) ,'_sublayer_',num2str(c)])
    
    %convolution2dLayer([1, d],d/2,'WeightsInitializer','ones', 'Name', ['channelComb', num2str(divTime) ,'sublayer',num2str(c)])
    
    %resize3dLayer('OutputSize',[T,d/2,1],'Name',['resize1_', num2str(divTime) ,'sublayer',num2str(c)])
    depthToSpace2dLayer_our(1, 2*d,['depthSpace', num2str(divTime) ,'sublayer',num2str(c)])
     
   convolution2dLayer([T, 1],alpha*d, 'WeightsInitializer','narrow-normal','Name',['inputComb', num2str(divTime) ,'sublayer',num2str(c)])
   
    %convolution2dLayer([T, 1],alpha*d/2, 'WeightsInitializer','narrow-normal','Name',['inputComb', num2str(divTime) ,'sublayer',num2str(c)])
    dropoutLayer(0.5,'Name',['drop1', num2str(divTime) ,'_sublayer_',num2str(c)])
    %resize3dLayer('OutputSize',[alpha*d/2, d/2, 1], 'Name',['resize2_', num2str(divTime) ,'sublayer',num2str(c)])
    %depthToSpace2dLayer_our(alpha*d/2, 1,['depthSpace2', num2str(divTime) ,'sublayer',num2str(c)])
    depthToSpace2dLayer_our(alpha*d, 1,['depthSpace2', num2str(divTime) ,'_sublayer_',num2str(c)])
    leakyReluLayer(1, 'Name',['activationLR', num2str(divTime) ,'_sublayer_',num2str(c)])
    %dropoutLayer(0.5,'Name',['drop', num2str(divTime) ,'_sublayer_',num2str(c)])
            ];
        
        if (divTime == 1)
            T = alpha*d/2;
        else
            T = T/2;
        end
        k_d=2*d;
        d = d/2;
        
        lgraph = addLayers(lgraph, layers);
        
    end
    d = channelSize;
    lgraph = connectLayers(lgraph,['subbanComb_sublayer_',num2str(c)] ,['channelComb', num2str(1) ,'_sublayer_',num2str(c)] );
    
    for divTime=1:subNetworkSize-1
        lgraph = connectLayers(lgraph, ['activationLR', num2str(divTime) ,'_sublayer_',num2str(c)], ['channelComb', num2str(divTime+1) ,'_sublayer_',num2str(c)]);
        %lgraph = connectLayers(lgraph, ['depthSpace2', num2str(divTime) ,'_sublayer_',num2str(c)],['channelComb', num2str(divTime+1) ,'_sublayer_',num2str(c)]);
        %lgraph = connectLayers(lgraph, ['drop', num2str(divTime) ,'_sublayer_',num2str(c)],['channelComb', num2str(divTime+1) ,'_sublayer_',num2str(c)]);
    end
    
    fcLayer = fullyConnectedLayer(1, 'Name',['fc_layer_sublayer_', num2str(c)]);
    lgraph = addLayers(lgraph,fcLayer);
    tanhlayer = tanhLayer('Name',['tanh1', num2str(c)]);
    lgraph = addLayers(lgraph,tanhlayer);
    %lgraph = connectLayers(lgraph,['drop', num2str(subNetworkSize) ,'_sublayer_',num2str(c)] ,['fc_layer_sublayer_', num2str(c)]); 
    %lgraph = connectLayers(lgraph,['depthSpace2', num2str(subNetworkSize) ,'_sublayer_',num2str(c)] ,['fc_layer_sublayer_', num2str(c)]); 
    lgraph = connectLayers(lgraph,['activationLR', num2str(subNetworkSize) ,'_sublayer_',num2str(c)] ,['fc_layer_sublayer_', num2str(c)]); 
    lgraph = connectLayers(lgraph ,['fc_layer_sublayer_', num2str(c)],['tanh1', num2str(c)]);
    
end



concatLayer = concatenationLayer(3,networkSize,'Name','concat_layer');
lgraph = addLayers(lgraph,concatLayer);



%plot(lgraph);
for c = 1:networkSize
 %disp(c)
lgraph = connectLayers(lgraph, ['tanh1', num2str(c)] , ['concat_layer/in',num2str(c)]);
end

outLayers = [
    softmaxLayer('Name','softMax_layer')
    classificationLayer('Name','classification_layer')
    ];
lgraph = addLayers(lgraph, outLayers);
lgraph = connectLayers(lgraph,'concat_layer', 'softMax_layer');

%plot(lgraph);
%analyzeNetwork(lgraph);




%% Training
max_epochs=400;
%acc_matrix=zeros(totalsubject,totalblock); % Initialization of accuracy matrix

allblock=1:5;
%allblock(block)=[]; Exclude the block used for testing     

%layers(2, 1).BiasLearnRateFactor=0; % At first layer, sub-bands are combined with 1 cnn layer, 
% bias term basically adds DC to signal, hence there is no need to use 
% bias term at first layer. Note: Bias terms are initialized with zeros by default. 



train=AllData(:,:,:,:,allblock,:); %Getting training data

train=reshape(train,[sizes(1),sizes(2),sizes(3),totalcharacter*length(allblock)*totalsubject*1]);

train_y=y_AllData(:,:,allblock,:);
train_y=reshape(train_y,[1,totalcharacter*length(allblock)*totalsubject*1]);    
train_y=categorical(train_y);

testblock = 6;
testdata=AllData(:,:,:,:,testblock,:);
testdata=reshape(testdata,[sizes(1),sizes(2),sizes(3),totalcharacter* totalsubject]);

test_y=y_AllData(:,:,testblock,:);
test_y=reshape(test_y,[1,totalcharacter*totalsubject]);
test_y=categorical(test_y);


options = trainingOptions('adam',... % Specify training options for first-stage training
    'InitialLearnRate',0.0001,...
    'MaxEpochs',max_epochs,...
    'ValidationData',{testdata,test_y},...
    'MiniBatchSize',300, ...
    'Shuffle','every-epoch',...
    'L2Regularization',0.001,...
    'ExecutionEnvironment','cpu');    
main_net = trainNetwork(train,train_y,lgraph,options);    
sv_name=['main_net_',int2str(2),'.mat']; 
save(sv_name,'main_net'); % Save the trained model

%% Testing
all_conf_matrix=zeros(40,40); % Initialization of confusion matrix 
acc_matrix=zeros(totalsubject,1); % Initialization of accuracy matrix

for s=1:totalsubject
testdata=AllData(:,:,:,:,testblock,s);
testdata=reshape(testdata,[sizes(1),sizes(2),sizes(3),totalcharacter]);

test_y=y_AllData(:,:,testblock,s);
test_y=reshape(test_y,[1,totalcharacter*1]);
test_y=categorical(test_y);

[YPred,~] = classify(main_net,testdata);
acc=mean(YPred==test_y');
acc_matrix(s,testblock)=acc;
disp(acc);

all_conf_matrix=all_conf_matrix+confusionmat(test_y,YPred);

end    

sv_name=['confusion_mat_',int2str(testblock),'.mat'];
save(sv_name,'all_conf_matrix');    

sv_name=['acc_matrix','.mat'];
save(sv_name,'acc_matrix');  
disp(mean(mean(acc_matrix(:,6))));

%% Notes



