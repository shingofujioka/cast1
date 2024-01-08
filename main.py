from fastapi import FastAPI, File
from PIL import Image
import io
import torch
from pydantic import BaseModel
import pickle
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2
import pytorch_lightning as pl
import torch.nn as nn
#from model import Net

#インスタンス化
app = FastAPI()

#ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.feature = mobilenet_v2(weights=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

# 画像変換の設定
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ネットワークモデルのロード
net = Net().cpu().eval()

#学習済みモデルの読み込み
model_weights = torch.load('models/model_cast.pth')

# 読み込んだ重みをネットワークモデルに設定
net.load_state_dict(model_weights)

#トップページ
@app.get('/')
def index():
    return {"cast":'鋳造部品 AI'}

#POSTが送信されたとき（入力）と予測値（出力）の定義
@app.post("/prediction")
async def predict_image(image_bytes: bytes = File(...)):
    # 画像のバイトデータを受け取り、PIL Imageオブジェクトに変換
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)

    # もし画像がグレースケールの場合、RGBに変換
    if image.mode != "RGB":
        image = image.convert("RGB")
        

    # 画像をモデルに適した形に変換（リサイズ、グレースケール変換、テンソル変換）
    transformed_image = transform(image)

    # 変換された画像をモデルに入力し、生の予測値を取得
    prediction = net(transformed_image.unsqueeze(0))

    # Softmax関数を適用して、生の予測値を確率に変換
    probabilities = F.softmax(prediction, dim=1)


    # 最も確率が高いクラスを決定
    most_probable_class = torch.argmax(probabilities)
    print('test1:', most_probable_class)

    # 各クラスの確率をnumpy配列に変換し、4桁の小数点まで丸めて可読性を高める
    class_probabilities = probabilities.detach().numpy()[0].tolist()
    class_probabilities = [round(float(prob), 4) for prob in class_probabilities]
    print('test2:', class_probabilities)

    # 最も確率が高いクラスと各クラスの確率を含む辞書をレスポンスとして返す
    return {
        "most_probable_class": most_probable_class.item(),
        "class_probabilities": class_probabilities
    }
