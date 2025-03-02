from model import *

def get_model(num_class, pretrained_path, device):
    model = poolformer_s12(num_classes=1000)
    model.load_state_dict(torch.load(pretrained_path, weights_only=True))
    model.head = torch.nn.Linear(model.head.in_features, num_class)
    self_model = model.to(device)

    # model = resnet34(num_classes=1000)
    # model.load_state_dict(torch.load(pretrained_path, weights_only=True))
    # model.fc = torch.nn.Linear(model.fc.in_features, num_class)  # 修改全连接层
    # model = model.to(device)


    # model = EyeSeeNet(num_class=7).to(device)

    # self_model = efficientnetv2_s(num_classes=1000)
    # self_model.load_state_dict(torch.load(pretrained_path, weights_only=True))
    # self_model.head.classifier = torch.nn.Linear(self_model.head.classifier.in_features, num_class)
    # self_model = self_model.to(device)
    # for name, para in model.named_parameters():
    #     # 除head外，其他权重全部冻结
    #     if "head" not in name:
    #         para.requires_grad_(False)
    #     else:
    #         print("training {}".format(name))


    return self_model






