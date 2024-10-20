import torch
import torch.nn as nn
from torchvision import models

class CustomModel(nn.Module):
    def __init__(self, model_name='mobilenet_v2'):
        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.model = self._initialize_model(model_name)

    def _initialize_model(self, model_name):
        if 'efficientnet_b' in model_name:
            model = getattr(models, model_name)(pretrained=True)
            # Remove the classifier
            model_features = nn.Sequential(*list(model.children())[:-2])
            # Add custom classifier
            num_features = model.classifier[1].in_features
            custom_head = nn.Linear(num_features, 512)

        elif model_name in ['mobilenet_v3_large', 'mobilenet_v3_small']:
            model = getattr(models, model_name)(pretrained=True)
            # Remove the classifier
            model_features = model.features
            # Add custom classifier
            num_features = model.classifier[3].in_features
            custom_head = nn.Linear(num_features, 512)

        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            # Remove the classifier
            model_features = model.features
            # Add custom classifier
            num_features = model.classifier[1].in_features
            custom_head = nn.Linear(num_features, 512)

        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Combine the features extractor and the custom head
        model = nn.Sequential(model_features, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), custom_head)
        return model

    def forward(self, x):
        return self.model(x)

def get_model(model_name):
    return CustomModel(model_name)

# Example usage:
if __name__ == "__main__":
    # Replace 'efficientnet_b1' with the model you want to use
    model_name = 'efficientnet_b1'
    model = get_model(model_name)
    print(model)
