%% k-Means Model

%{ 
    NOTES:
    
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

sizes = [total_channel, sample_length ,subban_no];

% for target_subject=1:totalsubject

target_subject = 1;
   
filename = dirname + model_name + num2str(target_subject) + ".mat";
load(filename);


net = main_net;
feature_vector_size = 120*25;
feature_vectors = zeros(feature_vector_size, 240);
feature_labels = zeros(240, 1);

for testblock=1:totalblock

    testdata=AllData(:,:,:,1:40,testblock,target_subject);
    testdata=reshape(testdata,[sizes(1),sizes(2),sizes(3),totalcharacter]);

    test_y=y_AllData(:,1:40,testblock,target_subject);
    test_y=reshape(test_y,[1,totalcharacter*1]);
    test_y=categorical(test_y);

    X = testdata;
    layer = 8; % 'conv_4'
    act = activations(net,X,layer);
    feature = reshape(act, [feature_vector_size, totalcharacter]);
    start_index = (40 * (testblock -1)) + 1;
    end_index = testblock * 40;

    feature_vectors(:, start_index:end_index) = feature;
    feature_labels(start_index:end_index,:) = test_y;

end


%% K-Means Model
cluster_num = 40;
X = feature_vectors';
idx = kmeans(X, cluster_num);

cluster_labels = zeros(40, 6);

for class=1:40
    
    for block=1:6
        index = (block-1)*40 + class;
        cluster_labels(class, block)= idx(index);
    end
end

[M ,F] = mode(cluster_labels, 2);
    


%% PCA

X = feature_vectors;
algorithm = 'svd'; % 'eig', 'als', 'svd'
num_components = 10;
[coeff,score,latent] = pca(X,'Algorithm',algorithm, 'NumComponents', num_components);

%% Fit GMM
k = cluster_num;
X = coeff;
GMModel = fitgmdist(X,k, 'SharedCovariance', true, 'CovarianceType', 'diagonal');

%% Cluster
gmm_cluster_labels = cluster(GMModel, X);

[M ,F] = mode(gmm_cluster_labels, 2);
