from resnet_from_scratch.resnet import ResNet

# resnetX = (Num of channels, repetition, Bottleneck_expansion, Bottleneck_layer)
model_parameters = {}
model_parameters['resnet18'] = ([64, 128, 256, 512], [2, 2, 2, 2], 1, False)
model_parameters['resnet34'] = ([64, 128, 256, 512], [3, 4, 6, 3], 1, False)
model_parameters['resnet50'] = ([64, 128, 256, 512], [3, 4, 6, 3], 4, True)
model_parameters['resnet101'] = ([64, 128, 256, 512], [3, 4, 23, 3], 4, True)
model_parameters['resnet152'] = ([64, 128, 256, 512], [3, 8, 36, 3], 4, True)

# Instantiate the model

# resnet18
def resnet18(in_channels, num_classes):
    resnet18 = ResNet(
        resnet_variant=model_parameters['resnet18'],
        in_channels=in_channels,
        num_classes=num_classes
    )
    return resnet18