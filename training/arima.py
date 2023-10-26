import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima import datasets
from pmdarima import utils
from pmdarima import arima
from pmdarima import model_selection
from sklearn.metrics import mean_absolute_error
from statistics import mean 
from matplotlib import pyplot as plt

def main():
    # グラフのスタイルとサイズ
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = [12, 9]
    data = datasets.load_wineind()
    fig_name = f"fig.png"

    # データ分割（train:学習データ、test:テストデータ）
    train, test = model_selection.train_test_split(data, train_size=140)
    # モデル構築（Auto ARIMA）
    arima_model = pm.auto_arima(train, 
                            seasonal=True,
                            m=12,
                            trace=True,
                            n_jobs=-1,
                            maxiter=10)
    # グラフのサイズ変更
    plt.rcParams['figure.figsize'] = [12, 9]
    # 予測
    preds, conf_int = arima_model.predict(n_periods=test.shape[0], 
                                        return_conf_int=True)
    # 予測精度
    print('MAE:')
    print(mean_absolute_error(test, preds)) 
    print('MAPE(%):')
    print(mean(abs(test - preds)/test) *100)
    # 予測と実測の比較（グラフ）
    x_axis = np.arange(preds.shape[0])
    plt.plot(x_axis,test,label="actual",color='r') 
    plt.plot(x_axis,preds,label="predicted",color='b')
    plt.fill_between(x_axis[-preds.shape[0]:],
                    conf_int[:, 0], conf_int[:, 1],
                    alpha=0.1, color='b')
    plt.legend()
    plt.savefig(fig_name)



if __name__ == "__main__":
    main()
    #a = np.array([1,2,3,2,5,8])
    #b = np.array([1,2,3,4,5,5])
    #print(np.linalg.norm(a-b)**2)