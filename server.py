import torch.nn as nn
import math

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, 1000)
        self.classifier1 = nn.Linear(1000, 102)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.classifier1(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)


# Định nghĩa các nhãn cho các loại hoa
labels = ["fire lily", "canterbury bells", "bolero deep blue", "pink primrose", "mexican aster", "prince of wales feathers", "moon orchid", "globe-flower", "grape hyacinth", "corn poppy", "toad lily", "siam tulip", "red ginger", "spring crocus", "alpine sea holly", "garden phlox",  "globe thistle",  "tiger lily", "ball moss", "love in the mist",  "monkshood",  "blackberry lily",  "spear thistle",  "balloon flower",  "blanket flower",  "king protea",  "oxeye daisy",  "yellow iris",  "cautleya spicata",  "carnation", "silverbush", "bearded iris", "black-eyed susan", "windflower", "japanese anemone", "giant white arum lily", "great masterwort",  "sweet pea",  "tree mallow",  "trumpet creeper",  "daffodil",  "pincushion flower",  "hard-leaved pocket orchid",  "sunflower",  "osteospermum",  "tree poppy",  "desert-rose",  "bromelia",  "magnolia",  "english marigold",  "bee balm",  "stemless gentian",  "mallow",  "gaura",  "lenten rose",  "marigold",  "orange dahlia",  "buttercup",  "pelargonium",  "ruby-lipped cattleya",  "hippeastrum",  "artichoke",  "gazania",  "canna lily", "peruvian lily",  "mexican petunia", "bird of paradise", "sweet william",  "purple coneflower",  "wild pansy",  "columbine",  "colt's foot",  "snapdragon",  "camellia",  "fritillary",  "common dandelion",  "poinsettia", "primula",  "azalea", "californian poppy",  "anthurium",  "morning glory",  "cape flower", "bishop of llandaff",  "pink-yellow dahlia",  "clematis",  "geranium", "thorn apple",  "barbeton daisy",  "bougainvillea",  "sword lily", "hibiscus", "lotus lotus", "cyclamen",  "foxglove",  "frangipani",  "rose", "watercress", "water lily",  "wallflower",  "passion flower",  "petunia"]

# Load mô hình
model_path = "/Users/minhtrietpham/PycharmProjects/test_pbl/model_0.9882669413919414.pt"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()


# Hàm tiền xử lý ảnh
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # tạo batch với kích thước 1
    return input_batch


# Hàm dự đoán
def predict(image):
    input_batch = preprocess_image(image)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities


@app.route('/predict', methods=['POST'])
def predict_flower():
    try:
        file = request.files['image']
        image = Image.open(file.stream)
        probabilities = predict(image)

        # Tìm nhãn có xác suất cao nhất
        max_prob, max_idx = torch.max(probabilities, 0)
        result = {"label": labels[max_idx], "probability": float(max_prob)}
        # result = [{"label": labels[i], "probability": float(probabilities[i])} for i in range(len(labels))]

        print(result)
        return jsonify(result)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)