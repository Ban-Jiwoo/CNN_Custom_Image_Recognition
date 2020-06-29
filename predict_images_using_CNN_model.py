import os
from kospiCode import *
from date import *
import pandas_datareader.data as web
import datetime
import plotly.graph_objects as go
import pandas as pd
from plotly.graph_objs import *
import plotly.io as pio
from PIL import Image
import tensorflow.keras.models as Models
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
import pandas as pd




#이미지데이터폴더만들기
os.system("mkdir pred_data_for_BMW; cd pred_data_for_BMW; mkdir pred_data_for_BMW")
PATH = os.getcwd()
data_path = PATH + '/pred_data_for_BMW/pred_data_for_BMW/'

# kospiList = ['049770']

# start = datetime.datetime(2019, 12, 26)
# end = datetime.datetime(2019, 12, 30)

print("Start Downloading the Images to "+data_path)

#KospiList 는 ['111222','222334', ] 구조로 이루어져 있음.
for code in kospiList:
    try :
        df = web.DataReader(code+".KS", "yahoo", start, end)

        #원하는 조건으로 종목 필터링
        if df['Volume'][0] < df['Volume'][1] and df['Volume'][1] < df['Volume'][2]:

            #커스텀 차트 이미지 만들기
            fig = go.Figure(data=[go.Candlestick(
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'])
                         ])

            fig.update_layout(
                yaxis=dict(
                    showgrid=False,
                    showticklabels=False
                ),

                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),

                xaxis_rangeslider_visible=False,

                paper_bgcolor='white',
                plot_bgcolor='white',
                showlegend=False,
                autosize=False,
                width=150,
                height=150,
                margin=go.layout.Margin(
                    l=0,
                    r=0,
                    b=0,
                    t=0,
                ),

            )
            # fig.show()
            pio.write_image(fig, data_path+''+code+'.jpg')

        else:
            pass


    except:
        print('{} 에러'.format(code))

print("The work has done.")


#cnn_making_model.py에서 만든 모델불러오기
from tensorflow.keras.models import load_model

model = load_model('your_model.h5')


#예측폴더 진입
PATH = os.getcwd()
data_path = PATH + '/pred_data_for_BMW/'
print("Entering to the pred data folder... " + data_path)
os.system("cd pred_data_for_BMW; find . -name '.DS_Store' -type f -delete")
pred_images, no_labels, imageNames = get_images(data_path)
pred_images = np.array(pred_images)

#dataframe만들기, 원하는 정확도 조건 설정하기.
iNames = []
pClass = []
pProb = []

for i in range(0, len(pred_images)):
    pred_image = np.array([pred_images[i]]).astype(np.float32)
    pred_class = get_classlabel(model.predict_classes(pred_image)[0])
    pred_probx = model.predict(pred_image)
    pred_prob = 100*np.max(pred_probx)

    print("{} ({}) {}%".format(imageNames[i][:6],
                                    pred_class,
                                    pred_prob))
    if pred_prob >= 90 and pred_class == 'BMW':
        iNames.append(imageNames[i].strip('.jpg'))
        pClass.append(pred_class)
        pProb.append(pred_prob)


raw_data = {'Code':iNames,
            'Class':pClass,
            'Prob':pProb
           }

data = pd.DataFrame(raw_data)

print("Complete the BMW dataframe")

#데이터프레임 정리 정렬
ascendingData = data.sort_values(['Prob'], ascending=[False])

#결과
print("The final BMW result has been made, ")
print(ascendingData)

#결과 리스트로 보내기.
results = ascendingData['Code'].values.tolist()
print("I've made the filtered code list\n")
print(results)
