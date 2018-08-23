import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

ridg = Ridge(alpha=0.01)

#svr = SVR(kernel='poly')
svr = GridSearchCV(SVR(kernel='linear', gamma=0.1), cv=5, param_grid={"C":[1e0,1e1,1e2,1e3],"gamma":np.logspace(-2,2,5)})
kr = GridSearchCV(KernelRidge(kernel='linear',gamma=0.1), cv=5, param_grid={"alpha":[1e0,0.1,1e-2, 1e-3], "gamma":np.logspace(-2,2,5)})

def get_coef(x,y,model=ridg):
#x=np.array([-10.1, -8.9, 0, 5.2, 10.1]).reshape(-1, 1)
#y=np.array([-19, -18.1, 2, 10, 21]).reshape(-1, 1)

    if model == ridg:
        y = np.array(y,dtype=float).reshape(-1, 1)
        x = np.array(x, dtype=float).reshape(-1, 1)
    else:
        x = np.array(x, dtype=float).T
        y = np.array(y, dtype=float)

    model.fit(x,y)

    #print('score: {}'.format(ridg.score(x,y)))

    # plt.scatter(x,y,color = 'black')
    # plt.plot(x,ridg.predict(x), color='blue',linewidth=1)
    # plt.xticks()
    # plt.yticks()
    # plt.show()
    if model == ridg:
        # print('Coef: {}'.format(model.coef_))
        # print('Intercept: {}'.format(model.intercept_))
        return model.coef_, model.intercept_

def get_pred(test, model=ridg):
    return model.predict(test)

if __name__ == '__main__':
    x = [-10.1, -8.9, 0, 5.2, 10.1]
    y = [-19, -18.1, 2, 10, 21]
    coef, intercept = get_coef(x, y)
    print('coef: {}'.format(coef))
    print('intercept: {}'.format(intercept))