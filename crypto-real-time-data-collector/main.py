import numpy as np
import pandas as pd
import requests
import json
import os
import sys
import torch
import torch.nn as nn
import time
import datetime

import firebase_admin
from firebase_admin import credentials, firestore

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb

# ---------------------------
# Debug: Check environment variable
# ---------------------------
kucoin_key_status = 'KEY_FOUND' if os.environ.get('KUCOIN_API_KEY') else 'KEY_NOT_FOUND'
print(f"DEBUG CHECK: KUCOIN_API_KEY status: {kucoin_key_status}", file=sys.stderr)

# ---------------------------
# Firebase Initialization
# ---------------------------
if not firebase_admin._apps:
    try:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)
        print("Firebase Admin initialized with ADC", file=sys.stderr)
    except Exception as e:
        print(f"Error initializing Firebase Admin: {e}", file=sys.stderr)

db = firestore.client()

# ---------------------------
# Constants
# ---------------------------
SYMBOL = 'BTC-USDT'
INTERVAL = '1h'
CANDLE_LIMIT = 1000
TARGET_TIMEFRAME = 1
K_FEATURES = 10
NUM_EPOCHS = 20
BATCH_SIZE = 32
NUM_BOOST_ROUNDS = 150
EARLY_STOPPING_ROUNDS = 10

KUCOIN_INTERVAL_MAP = {
    '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min',
    '30m': '30min', '1h': '1hour', '2h': '2hour', '4h': '4hour',
    '6h': '6hour', '8h': '8hour', '12h': '12hour', '1d': '1day',
    '1w': '1week', '1M': '1month'
}

APP_ID = 'default-app-id'
CANDLE_COLLECTION_PATH = f'artifacts/{APP_ID}/public/data/cryptoCandles'
METRICS_DOC_PATH = f'artifacts/{APP_ID}/public/data/predictionMetrics/btcUsdtML'
PREDICTIONS_COLLECTION_PATH = f'artifacts/{APP_ID}/public/data/predictions'

# ---------------------------
# Helper functions
# ---------------------------
def log_debug(message, extra=None):
    """Write debug logs to Firestore."""
    print(f"DEBUG: {message} | {extra if extra else ''}", file=sys.stderr)
    if db is None:
        return
    entry = {
        "timestamp": datetime.datetime.utcnow(),
        "message": message
    }
    if extra is not None:
        entry["extra"] = extra
    db.collection("artifacts/debugLogs/pythoncryptopredictor").add(entry)


class HybridNN(nn.Module):
    def __init__(self, input_size):
        super(HybridNN, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)
        return x


class TradingDataset(torch.utils.data.Dataset):
    def __init__(self, X_data, y_data=None):
        self.X_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).float().unsqueeze(1) if y_data is not None else None

    def __getitem__(self, index):
        if self.y_data is not None:
            return self.X_data[index], self.y_data[index]
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


def get_crypto_data(symbol, interval, limit):
    kucoin_interval = KUCOIN_INTERVAL_MAP.get(interval)
    if not kucoin_interval:
        log_debug("Invalid interval", {"interval": interval})
        return pd.DataFrame()
    base_url = "https://api.kucoin.com/api/v1/market/candles"
    all_raw_data, end_time_sec = [], None

    while len(all_raw_data) < limit:
        params = {'symbol': symbol, 'type': kucoin_interval}
        if end_time_sec:
            params['endAt'] = end_time_sec
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data_json = response.json()
        if data_json.get('code') != '200000':
            log_debug("KuCoin API error", {"code": data_json.get('code')})
            break
        new_data = data_json['data']
        if not new_data:
            break
        oldest_ts = int(new_data[-1][0])
        end_time_sec = oldest_ts - int(pd.Timedelta(interval).total_seconds())
        all_raw_data.extend(new_data)
        time.sleep(0.1)

    df = pd.DataFrame(all_raw_data, columns=['Open time','Open','Close','High','Low','Volume','Turnover'])
    df = df.iloc[::-1].tail(limit)
    df = df[['Open time','Open','High','Low','Close','Volume']].apply(pd.to_numeric, errors='coerce')
    df['Open time'] = pd.to_datetime(df['Open time'].astype(float), unit='s')
    df['Close time'] = df['Open time'] + pd.Timedelta(interval) - pd.Timedelta(milliseconds=1)
    df = df.set_index('Open time')
    log_debug("Fetched crypto data", {"rows": len(df)})
    return df


def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss
    rs[np.isinf(rs)] = 100
    data['RSI'] = 100 - (100 / (1 + rs))
    return data


def calculate_wick_features(df):
    df['BodyMax'] = df[['Open','Close']].max(axis=1)
    df['BodyMin'] = df[['Open','Close']].min(axis=1)
    df['UpperWick'] = df['High'] - df['BodyMax']
    df['LowerWick'] = df['BodyMin'] - df['Low']
    df['TotalRange'] = df['High'] - df['Low']
    zero_range = df['TotalRange'] == 0
    df['UpperWickProportion'] = np.where(zero_range,0,df['UpperWick']/df['TotalRange'])
    df['LowerWickProportion'] = np.where(zero_range,0,df['LowerWick']/df['TotalRange'])
    return df.drop(columns=['BodyMax','BodyMin','UpperWick','LowerWick','TotalRange'])


def create_all_features(df):
    df['Future Close'] = df['Close'].shift(-TARGET_TIMEFRAME)
    df['Target'] = (df['Future Close'] > df['Close']).astype(int)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df = calculate_rsi(df)
    df = calculate_wick_features(df)
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    df['Daily_Change'] = df['Close'].pct_change()
    df['High_Low_Range'] = (df['High']-df['Low'])/df['Close']
    log_debug("Features created", {"columns": list(df.columns)})
    return df


def save_candles_to_firestore(df_raw):
    if db is None or df_raw.empty:
        return
    batch = db.batch()
    collection_ref = db.collection(CANDLE_COLLECTION_PATH)
    for idx,row in df_raw.iterrows():
        doc_ref = collection_ref.document(str(int(idx.value//10**6)))
        batch.set(doc_ref, {
            'timestamp': idx.to_pydatetime(),
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume'],
        })
    batch.commit()
    log_debug("Candles written to Firestore", {"count": len(df_raw)})


def save_prediction_metrics(prediction_data, df_raw):
    """
    df_raw: the latest candle dataframe returned by get_crypto_data
    prediction_data: dict containing 'prediction', 'confidence_score', 'timestamp', 'latest_close', 'interval'
    """
    if db is None:
        return

    doc_ref = db.document(METRICS_DOC_PATH)
    doc = doc_ref.get()
    
    total_predictions = 1
    correct_predictions = 0

    # Determine if previous prediction was correct
    if doc.exists:
        existing_data = doc.to_dict()
        total_predictions = existing_data.get('totalPredictions', 0) + 1
        correct_predictions = existing_data.get('correctPredictions', 0)
        
        last_pred = existing_data.get('lastPrediction')
        last_ts = existing_data.get('lastPredictedCandleTimestamp')
        if last_pred and last_ts:
            # Convert Firestore timestamp to datetime if needed
            if hasattr(last_ts, "to_datetime"):
                last_ts = last_ts.to_datetime()
            
            # Find the candle corresponding to last prediction
            if last_ts in df_raw.index:
                actual_close = df_raw.loc[last_ts, 'Close']
                prev_close = df_raw.loc[last_ts, 'Close'] - df_raw.loc[last_ts, 'Close'].pct_change()  # optional: could use Open instead
                # Determine actual movement
                movement = 'UP' if actual_close > prev_close else 'DOWN'
                if movement == last_pred:
                    correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    ts = datetime.datetime.strptime(prediction_data['timestamp'], '%Y-%m-%d %H:%M:%S')
    data = {
        'lastPrediction': prediction_data['prediction'],
        'confidence': float(prediction_data['confidence_score']),
        'lastPredictedCandleTimestamp': ts,
        'predictionMadeAtTimestamp': datetime.datetime.utcnow(),
        'latest_close': prediction_data['latest_close'],
        'interval': prediction_data['interval'],
        'totalPredictions': total_predictions,
        'correctPredictions': correct_predictions,
        'accuracy': accuracy
    }

    try:
        doc_ref.set(data, merge=True)
        print("Firestore WRITE SUCCESS")
    except Exception as e:
        print("Firestore WRITE FAILURE:", str(e))


def log_prediction_record(prediction_data):
    """
    Write a per-prediction document for confidence vs accuracy analysis.
    prediction_data: dict containing 'prediction', 'confidence_score', 'timestamp', 'latest_close', 'interval'
    """
    if db is None:
        return

    try:
        ts = datetime.datetime.strptime(prediction_data['timestamp'], '%Y-%m-%d %H:%M:%S')
        doc_id = str(int(ts.timestamp() * 1000))

        doc_ref = db.collection(PREDICTIONS_COLLECTION_PATH).document(doc_id)
        doc_ref.set({
            'predictionMadeAt': datetime.datetime.utcnow(),
            'targetCandleTimestamp': ts,
            'predicted': prediction_data['prediction'],
            'confidence': float(prediction_data['confidence_score']),
            'actual': None,
            'correct': None,
            'labeled': False,
            'interval': prediction_data.get('interval', INTERVAL),
            'latest_close': prediction_data.get('latest_close'),
        }, merge=True)

        print(f"Prediction record logged: {doc_id}")
    except Exception as e:
        print("Prediction record WRITE FAILURE:", str(e))



# ---------------------------
# Main function
# ---------------------------
def pythoncryptopredictor(data=None, context=None):
    start_time = time.time()
    try:
        log_debug("Function started")

        df_raw = get_crypto_data(SYMBOL, INTERVAL, CANDLE_LIMIT)

        MIN_REQUIRED_CANDLES = 110
        if df_raw.empty or len(df_raw) < MIN_REQUIRED_CANDLES:
            log_debug("Insufficient candle data", {"count": len(df_raw)})
            return {"status":"failure","message":"Insufficient data","interval":INTERVAL}

        df_features = create_all_features(df_raw)
        df_train_test = df_features.iloc[:-TARGET_TIMEFRAME].dropna()
        df_predict = df_features.tail(1).drop(columns=['Target','Future Close']).iloc[0]

        # ---------------------------
        # Train/Test Split & Scaling
        # ---------------------------
        X = df_train_test.drop(['Future Close','Target','Close time'],axis=1)
        y = df_train_test['Target']
        feature_names = list(X.columns)

        X_train_raw,X_test_raw,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
        X_train_raw,X_val_raw,y_train,y_val = train_test_split(X_train_raw,y_train,test_size=0.15,shuffle=False)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled_full = scaler.transform(X_val_raw)
        X_predict_scaled = scaler.transform(df_predict.drop('Close time').to_numpy().reshape(1,-1))
        y_train_np,y_val_np = y_train.to_numpy(),y_val.to_numpy()

        log_debug("Train/test split & scaling done", {"train_rows": len(X_train_scaled),"val_rows": len(X_val_scaled_full)})

        # ---------------------------
        # XGBoost
        # ---------------------------
        log_debug("Training XGB")
        selector_stats = SelectKBest(score_func=f_classif,k=K_FEATURES)
        X_train_stats = selector_stats.fit_transform(X_train_scaled,y_train_np)
        stats_indices = selector_stats.get_support(indices=True)
        X_val_stats_scaled = selector_stats.transform(X_val_scaled_full)

        dtrain = xgb.DMatrix(X_train_stats,label=y_train_np)
        dval = xgb.DMatrix(X_val_stats_scaled,label=y_val_np)
        params = {'objective':'binary:logistic','eval_metric':'logloss','eta':0.1,'max_depth':5,'seed':42}
        xgb_bst = xgb.train(params,dtrain,num_boost_round=NUM_BOOST_ROUNDS,evals=[(dval,'eval')],
                            early_stopping_rounds=EARLY_STOPPING_ROUNDS,verbose_eval=False)

        xgb_prob = xgb_bst.predict(xgb.DMatrix(X_predict_scaled[:,stats_indices]))[0]
        log_debug("XGB prediction done", {"xgb_prob": float(xgb_prob)})

        # ---------------------------
        # Neural Network
        # ---------------------------
        log_debug("Training NN")
        tree_selector = ExtraTreesClassifier(n_estimators=50,random_state=42)
        tree_selector.fit(X_train_scaled,y_train_np)
        top_indices = np.argsort(tree_selector.feature_importances_)[-K_FEATURES:]
        X_train_tree = X_train_scaled[:,top_indices]
        X_val_tree = X_val_scaled_full[:,top_indices]

        train_dataset = TradingDataset(X_train_tree,y_train_np)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
        nn_model = HybridNN(X_train_tree.shape[1])
        optimizer = torch.optim.Adam(nn_model.parameters(),lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(NUM_EPOCHS):
            nn_model.train()
            for X_batch,y_batch in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(nn_model(X_batch),y_batch)
                loss.backward()
                optimizer.step()

        nn_model.eval()
        with torch.no_grad():
            nn_prob = torch.sigmoid(nn_model(torch.from_numpy(X_predict_scaled[:,top_indices]).float())[0]).item()
        log_debug("NN prediction done", {"nn_prob": float(nn_prob)})

        # ---------------------------
        # Ensemble & Results
        # ---------------------------
        ensemble_prob = (xgb_prob + nn_prob)/2
        prediction = 1 if ensemble_prob>0.5 else 0
        final_pred = "UP" if prediction==1 else "DOWN"
        log_debug("Ensemble complete", {"final_prob": float(ensemble_prob), "label": final_pred})

        result = {
            "status":"success",
            "timestamp":df_raw.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            "latest_close": df_raw['Close'].iloc[-1],
            "prediction": final_pred,
            "confidence_score": float(ensemble_prob),
            "xgb_prob": float(xgb_prob),
            "nn_prob": float(nn_prob),
            "interval": INTERVAL
        }

        # ---------------------------
        # Save to Firestore
        # ---------------------------
        save_candles_to_firestore(df_raw[['Open','High','Low','Close','Volume']])
        log_prediction_record(result)
        save_prediction_metrics(result, df_raw)

        log_debug("Function completed successfully", {"duration_seconds": time.time()-start_time})
        return result

    except Exception as e:
        log_debug("Critical failure", {"error": str(e)})
        return {"status":"critical_failure","message":str(e),"interval":INTERVAL}
