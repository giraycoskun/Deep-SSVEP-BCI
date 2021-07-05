%% Principal Component Analysis of Feature Vectors From Global Model Output

%{ 
    NOTES:
    pca -> Returns the principal component coefficients, also 
    known as loadings, for the n-by-p data matrix X. 
    Rows of X correspond to observations and columns correspond to variables. 
    The coefficient matrix is p-by-p. Each column of coeff contains coefficients for one principal component, 
    and the columns are in descending order of component variance. 
    By default, pca centers the data and uses the singular value decomposition (SVD) algorithm.

    Features extracted from layer 8

https://www.mathworks.com/help/matlab/ref/colormap.html
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

target_subject = 1;
   
filename = dirname + model_name + num2str(target_subject) + ".mat";
load(filename);

%% Get Feature Vectors

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

%% PCA


X = feature_vectors;
algorithm = 'svd'; % 'eig', 'als'
num_components = 2;

[coeff,score,latent] = pca(X,'Algorithm',algorithm, 'NumComponents', num_components);


%% Graph

scatter(coeff(:,1), coeff(:,2), 36, feature_labels, 'filled');
colormap(turbo(40));
