'''
Author: hushuwang hushuwang2019@ia.ac.cn
Date: 2022-07-06 15:03:36
LastEditors: hushuwang hushuwang2019@ia.ac.cn
LastEditTime: 2022-07-06 20:23:14
FilePath: /mahua/metrics.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def metrics(y_true, y_pred):
    """
    input:
        NdArray/Tensor pred(B,T,N,N):
        NdArray/Tensor label(B,T,N,N):
    output:
        float MAE, MSE, RMSE
    """
    assert y_pred.shape == y_true.shape
    b, _, _, _ = y_pred.shape
    y_pred = y_pred.reshape(b, -1).cpu()
    y_true = y_true.reshape(b, -1).cpu()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return mae, mse, rmse