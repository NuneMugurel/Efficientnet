from efficientnet import model
from efficientnet.model_service import EfficientNetService

ImageNetPath = r"./ImageNet/"
batch_size = 4
num_epochs = 16

if __name__ == '__main__':
    model = model.EfficientNetExperimentalResolution(load_weights=True, num_classes=23)
    EfficientNetService.compile_model(model)
    callbacks = EfficientNetService.get_callbacks()
    EfficientNetService.fit_model(model, ImageNetPath, batch_size, num_epochs, callbacks, 267)
    #open tensorboard cmd for powershell: tensorboard --logdir=./logs