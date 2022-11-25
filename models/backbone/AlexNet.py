
import torch

class AlexNet:

    def initialise(self, pretrained=True, requires_grad=True, remove_fc=True, show_params=False):
        
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=pretrained)

        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

        if show_params:
            for name, param in self.model.named_parameters():
                print(name, param.size())
        return self.model
    
