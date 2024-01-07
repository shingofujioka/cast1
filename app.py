import streamlit as st
import pandas as pd
import requests
from PIL import Image
import base64


# ヘッダーセクション
st.title('Welcome to AI Quality Control!')
st.write("""
このページでは、画像を入力するだけで合否判定を行うことができます。
ユーザーは画像をドラッグアンドドロップまたはファイルを選択し、
「Start Analysis」ボタンをクリックするだけで、AIが部品の品質を評価します。
""")
# 画像ファイルの読み込み
image1 = Image.open('test2.jpeg')  # ここに画像ファイルのパスを入力
image2 = Image.open('test1.jpeg')  # ここに画像ファイルのパスを入力


# 2つの画像を横に並べて表示
col1, col2 = st.columns(2)
with col1:
    st.image(image1, width=300)
with col2:
    st.image(image2, width=300)

# 画像ファイルの読み込み
imageOK = Image.open('OK.jpeg')  # ここに画像ファイルのパスを入力
imageNG1 = Image.open('NG1.jpeg')  # ここに画像ファイルのパスを入力
imageNG2 = Image.open('NG2.jpeg')  # ここに画像ファイルのパスを入力
imageNG3 = Image.open('NG3.jpeg')  # ここに画像ファイルのパスを入力

# 画像を表示するためのコラムを作成
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image(imageOK, caption='適合例', width=150)

with col2:
    st.image(imageNG1, caption='不適合例1', width=150)

with col3:
    st.image(imageNG2, caption='不適合例2', width=150)

with col4:
    st.image(imageNG3, caption='不適合例3', width=150)

# ファイルアップロードセクション
st.subheader('↓検査する部品の画像をアップロードしてください↓')
# accept_multiple_filesをTrueに設定して複数のファイルをアップロードできるようにする
uploaded_file = st.file_uploader('Choose images of cast parts', type=['jpg', 'jpeg', 'png'])  #, accept_multiple_files=True

# アップロードされた画像の表示
if uploaded_file:
    # アップロードされた画像を表示する
    st.subheader('Uploaded Image')
    # PILライブラリを使用して画像を読み込む
    image = Image.open(uploaded_file)
    # 画像を表示
    st.image(image, caption=uploaded_file.name, width=300)


targets = ['不合格', '合格']

# 分析開始ボタン
if st.button('Start Analysis') and uploaded_file:
    
    # 入力された説明変数の表示
    #st.write('## Input Value')

    # FastAPIサーバーに画像を送信し、予測結果を取得
    response = requests.post(
        "https://ai-quality-control.onrender.com/prediction",
        files={"image_bytes": uploaded_file.getvalue()
               }
                )
    
    if response.status_code == 200:
        response_json = response.json()
        # 予測されたクラスと確率を安全に取得
        prediction = response_json.get("most_probable_class", "Unknown")
        class_probabilities = response_json.get("class_probabilities")
        max_probability = max(class_probabilities)*100

        # 予測結果の出力
        st.markdown('## Result', unsafe_allow_html=True)
        st.markdown(
        f'<div style="font-size: 24px;">この部品は、<span style="color: red;">{max_probability}%</span>の確率で、'
        f'<span style="color: red;">{str(targets[int(prediction)])}</span>です！</div>',
        unsafe_allow_html=True
         )
        
        # デバッグ用の応答内容の表示
        st.write('### Response JSON')
        st.json(response_json)


    else:
        # エラーが発生した場合の処理
        st.error("サーバーからの応答が無効です")






 # システムの動作説明
st.info('Just upload the image of your cast part and the AI will analyze it for quality control.')

# フッター
st.write('© 2023 AI Quality Control')   

# 注入するカスタム CSS
css = """
<style>
    .stApp {
        background-color: #F0FFF0;
    }
</style>
"""

# スタイルを注入
st.markdown(css, unsafe_allow_html=True)