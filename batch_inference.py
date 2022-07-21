import os
import joblib
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import KNNImputer

def handle_missing_cols(df):
    missing_df= df.isnull().sum().reset_index()
    na_cols = missing_df['index'][missing_df[0] > 0].tolist()
    n_neighbors = [5]
    for col_name in na_cols:
        for k in n_neighbors:
            knn_imp = KNNImputer()
            df[col_name] = knn_imp.fit_transform(df[[col_name]])
    return df



def identify_correlated_cols(df,threshold=0.7, mode="train"):
    """
    :param df : pandas dataframe with train/test data
    :param threshold: integer 
    :return: dataframe with new features
    """
    try:
        corr_cols = joblib.load(os.path.join('/app/models/' , 'corr_columns.pkl'))
        df.drop(corr_cols, axis=1, inplace=True)        
    except:
        raise
    return df


def preprocess(df, mode):
    try:
        df.pipe(handle_missing_cols)\
          .pipe(identify_correlated_cols, threshold='0.7', mode=mode)
    except:
        print("EXCEPTION CAUGHT IN preprocess FUNC")
        raise 
    return df


def predict(df, MODEL, fold=1):
    predictions= None

    for fold in range(fold):
        #load model
        clf = joblib.load(os.path.join('/app/models/' , f'{MODEL}.pkl'))
        X_test= preprocess(X_test, "test")
        preds = clf.predict(X_test)
        
        #append preds in each fold
        if fold == 0:
            predictions = preds
        else:
            predictions += preds
    if fold > 1:
        #avg of all folds preds
        predictions /= fold
    sub = pd.DataFrame(predictions, columns=['PREDICTIONS'])
    return sub



if __name__ == "__main__":
    data=[0.899922219,-0.161,-0.244,1.022,-0.304,-0.156,-0.31,12.064,-0.051,462.092,-0.732,1.14251,3,0.0034231,-0.262,33831,-0.12,-0.109,0.694306716,0.997923,88.1847211,74.7715,0.787,21,-0.005,-0.00457,0.801131,-0.143,-0.055,-0.005,-0.117,0.00336065,0.66388,-0.005,-0.054,3170.06,-0.784,-0.06,-0.686,-0.435,-3.350107803,0.889,96.3255,-0.564,-0.469,-0.028,-0.133,-21.4902,106.792,1.49544,704.956,0.351,-0.024,-0.098,0.178543,9.62,515.012,-0.146,5490.13,-0.028,105.219,0.995525,-0.00204,-1.604,0,0.0053249,67226,0.00372,-0.089,3.42085,0.00193191,0.181023,1.968,-212.318,0.00397,0.500891953,-0.0028,0.00466,-1.904040628,5190.98,0.00194778,-0.426,-0.103,1.19102,-0.032,0.927649417,-0.024,21.4711,0.00368,0.0045642,3138.67,0.01959,21,0.482589,-0.307,0,-0.12259,3.09698,-0.3607318,0.0051347,0.179927,1.49193,-0.012,3.10686,-0.00579,-0.18553,-0.122263,-0.12373,189.092,2052.68,-0.548,3.49045,-232.8,6,-0.021,-0.089,0.01107,-0.058,1.16655,21.4029,1.08095,0.0100732,1.837,0.255,0.385120596,6913.68,1.19125,1132,1.09259,0.204294,0.001265,-1.148,0.07,-0.035,1.08133,-0.031,-0.008,-0.458764352,0.481826,-0.062,66676.1,-0.658,764.3,-0.005,0.0603032,13.383,0.81990921,-0.18291,-0.12259,0.0606836,17.1094]
    MODEL ="randomforest"
    df = pd.DataFrame([data])
    f_cols = ["f_" + str(col) for col in range(150)]
    df.columns = f_cols
    print(df.shape)
    sub = predict(df, MODEL)
    print(sub)
    sub.to_csv(f"/app/models/", f'{MODEL}.csv', index=False)