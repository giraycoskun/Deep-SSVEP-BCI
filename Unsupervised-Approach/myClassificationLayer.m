classdef myClassificationLayer < nnet.layer.ClassificationLayer
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end
 
    methods
        function layer = myClassificationLayer(name)           
            % (Optional) Create a myClassificationLayer.

            % Layer constructor function goes here.
            layer.Name = name;
            layer.Description = 'Custom Classification absed on cross-entropy';

        end

        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the training 
            % targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here.
            loss = crossentropy(Y, T);
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % (Optional) Backward propagate the derivative of the loss 
            % function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the 
            %                 predictions Y

            % Layer backward loss function goes here.
        end
    end
end