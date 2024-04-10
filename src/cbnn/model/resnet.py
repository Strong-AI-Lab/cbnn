
import pytorch_lightning as pl
import torchvision.models as models
import torch


class ResNet18(pl.LightningModule):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, pretrained: bool = True, learning_rate : float = 0.005, weight_decay : float = 0.0, **kwargs):
        super(ResNet18, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.resnet = models.resnet18(pretrained=pretrained)
        if num_classes != 1000:
            self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes) # Change the output layer to match the number of classes
        if in_channels > 3 or in_channels == 2:
            self.resnet.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Change the input layer to match the number of channels (no modifications needed if 3 channels; if 1 channel, repeat the channels during forward pass)

        self.save_hyperparameters()

    def forward(self, x):
        # If the input has 1 channel, repeat the channels to match the input of the model
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)


        # Process the input
        return self.resnet(x)
    
    def loss_function(self, logits, y):
        return torch.nn.functional.cross_entropy(logits, y)
    
    def accuracy(self, logits, y):
        return torch.sum(torch.argmax(logits, dim=1) == y).item() / len(y)
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group("ResNet18")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
        parser.add_argument('--pretrained', type=bool, default=True, help='Load pretrained weights.')
        parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the optimizer.')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer.')
        return parent_parser
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        acc = self.accuracy(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        acc = self.accuracy(logits, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)