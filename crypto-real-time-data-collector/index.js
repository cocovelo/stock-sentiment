const { Storage } = require('@google-cloud/storage');
const storage = new Storage();
const { initializeApp, applicationDefault } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');
const tf = require('@tensorflow/tfjs-node'); 

initializeApp({
    credential: applicationDefault(),
});
const db = getFirestore();

const APP_ID = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id'; 

const CANDLE_COLLECTION_PATH = `artifacts/${APP_ID}/public/data/cryptoCandles`;
const METRICS_DOC_PATH = `artifacts/${APP_ID}/public/data/predictionMetrics/btcUsdtML`;
const PREDICTIONS_COLLECTION_PATH = `artifacts/${APP_ID}/public/data/predictions`;
const CANDLE_COLLECTION = db.collection(CANDLE_COLLECTION_PATH);
const COLLECTOR_SYMBOL = 'BTC-USDT'; 
const COLLECTOR_INTERVAL = '15min'; 
const COLLECTOR_LIMIT = 500; // Number of candles to keep in the rolling window
const COLLECTOR_API_URL_BASE = 'https://api.kucoin.com/api/v1/market/candles'; 
const DATA_FILE_GCS_PATH = 'gs://client-project-viewer.firebasestorage.app/crypto_data.jsonl';
const BUCKET_NAME = DATA_FILE_GCS_PATH.replace('gs://', '').split('/')[0];
const FILE_NAME = DATA_FILE_GCS_PATH.replace('gs://', '').substring(BUCKET_NAME.length + 1);
const PREDICTOR_LOOKBACK_WINDOW = 3; // For momentum features (N, N-1, N-2 candle changes)
const PREDICTOR_RSI_PERIOD = 14;
const PREDICTOR_SMA_SHORT_PERIOD = 5; 
const PREDICTOR_SMA_LONG_PERIOD = 20; 
const PREDICTOR_ATR_PERIOD = 14; 
const PREDICTOR_BB_PERIOD = 20; // Bollinger Bands period
const PREDICTOR_BB_STDDEV = 2.0; // Bollinger Bands standard deviation

const EPSILON = 1e-7; // Used to prevent NaN from division by zero during normalization and division by zero.

async function fetchWithRetry(url, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url);
            if (response.ok) {
                return response;
            }
            if (i < retries - 1) {
                console.warn(`Attempt ${i + 1} failed with status ${response.status}. Retrying in ${2 ** i} seconds...`);
            } else {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        } catch (error) {
            if (i === retries - 1) {
                throw error; 
            }
            console.warn(`Attempt ${i + 1} failed: ${error.message}. Retrying in ${2 ** i} seconds...`);
        }
        await new Promise(resolve => setTimeout(resolve, (2 ** i) * 1000));
    }
    throw new Error(`Failed to fetch ${url} after ${retries} attempts.`);
}

function mapKlineData(rawKline) {
    if (!rawKline || rawKline.length < 7) return null; 

    const timestamp = parseInt(rawKline[0]) * 1000;
    const open = parseFloat(rawKline[1]);
    const close = parseFloat(rawKline[2]); 
    const high = parseFloat(rawKline[3]);
    const low = parseFloat(rawKline[4]);
    const volume = parseFloat(rawKline[5]); 
    const turnover = parseFloat(rawKline[6]);

    if (!Number.isFinite(open) || !Number.isFinite(close) || 
        !Number.isFinite(high) || !Number.isFinite(low) || 
        !Number.isFinite(volume) || volume <= 0) {
        
        console.warn(`[DATA ERROR] Rejected candle at timestamp ${timestamp} due to non-finite or zero-volume data.`);
        return null;
    }

    return {
        timestamp: timestamp, 
        open: open,
        close: close, 
        high: high,
        low: low,
        volume: volume, 
        turnover: turnover, 
    };
}

async function loadFromCloudStorage() {
    try {
        const file = storage.bucket(BUCKET_NAME).file(FILE_NAME);
        const [contentBuffer] = await file.download();
        const jsonlContent = contentBuffer.toString();
        
        const data = jsonlContent.split('\n')
            .filter(line => line.trim().length > 0)
            .map(line => JSON.parse(line));

        console.log(`[SUCCESS] Loaded ${data.length} records from GCS.`);
        return data;
    } catch (e) {
        if (e.code === 404) {
            console.log('[INFO] Data file not found in GCS. Performing initial data fetch.');
            return [];
        }
        console.error('CRITICAL: Failed to load data from GCS:', e.message);
        throw e;
    }
}

async function fetchInitialData() {
    const fetchLimit = COLLECTOR_LIMIT + 1; 
    const url = `${COLLECTOR_API_URL_BASE}?symbol=${COLLECTOR_SYMBOL}&type=${COLLECTOR_INTERVAL}&limit=${fetchLimit}`;

    try {
        const response = await fetchWithRetry(url);
        const jsonResponse = await response.json(); 

        if (jsonResponse.code !== '200000' || !Array.isArray(jsonResponse.data)) {
            throw new Error(`KuCoin API returned non-success code or invalid data: ${jsonResponse.code || 'UNKNOWN'}`);
        }
        
        let structuredData = jsonResponse.data.map(mapKlineData).filter(Boolean);
        structuredData.reverse(); // Arrange chronologically (oldest first)

        if (structuredData.length > COLLECTOR_LIMIT) {
            structuredData = structuredData.slice(structuredData.length - COLLECTOR_LIMIT);
        }
        
        return structuredData;

    } catch (error) {
        console.error('Error fetching initial historical data:', error.message);
        throw error;
    }
}

async function fetchLatestClosedCandle() {
    const latestLimit = 2; // We need at least 2 to ensure the second one is closed.
    const url = `${COLLECTOR_API_URL_BASE}?symbol=${COLLECTOR_SYMBOL}&type=${COLLECTOR_INTERVAL}&limit=${latestLimit}`;

    try {
        const response = await fetchWithRetry(url);
        const jsonResponse = await response.json();

        if (jsonResponse.code !== '200000' || !Array.isArray(jsonResponse.data)) {
            throw new Error(`KuCoin API returned non-success code or invalid data: ${jsonResponse.code || 'UNKNOWN'}`);
        }
        
        const rawData = jsonResponse.data;
        
        if (rawData.length >= 2) {
            // rawData[0] is the current open candle; rawData[1] is the last closed candle.
            return mapKlineData(rawData[1]); 
        } else {
            console.warn('WARN: API response did not contain enough recent data (KuCoin).');
            return null;
        }
    } catch (error) {
        console.error('Error fetching latest candle:', error.message);
        throw error;
    }
}

async function saveToCloudStorage(data) {
    const jsonlContent = data.map(obj => JSON.stringify(obj)).join('\n');

    try {
        const file = storage.bucket(BUCKET_NAME).file(FILE_NAME);

        await file.save(jsonlContent, {
            contentType: 'application/x-jsonlines',
            resumable: false, 
            metadata: {
                cacheControl: 'no-cache', 
            }
        });

        console.log(`[SUCCESS] Data uploaded to GCS: gs://${BUCKET_NAME}/${FILE_NAME}. Total lines: ${data.length}`);

    } catch (e) {
        console.error('CRITICAL: Failed to upload data to Google Cloud Storage. Check service account permissions:', e.message);
        throw e;
    }
}

exports.cryptoCollector = async (req, res) => { 
    console.log(`\n--- Starting Cloud Function Execution at ${new Date().toISOString()} ---`);
    
    try {
        let ohlcData = await loadFromCloudStorage();

        if (ohlcData.length === 0) {
            const initialData = await fetchInitialData();
            if (initialData.length > 0) {
                ohlcData = initialData;
                console.log(`Initialized with ${ohlcData.length} records.`);
                await saveToCloudStorage(ohlcData);
            }
            res.status(200).send('Collector initialized data.');
            return; 
        }

        const latestCandle = await fetchLatestClosedCandle();

        if (latestCandle) {
            const lastLocalTimestamp = ohlcData.length > 0 ? ohlcData[ohlcData.length - 1].timestamp : 0;

            if (latestCandle.timestamp > lastLocalTimestamp) {
                // New candle received, add it
                ohlcData.push(latestCandle);
                console.log(`New candle received: O=${latestCandle.open}, C=${latestCandle.close}, T=${new Date(latestCandle.timestamp).toLocaleTimeString()}`);

                // Maintain the rolling window size
                if (ohlcData.length > COLLECTOR_LIMIT) {
                    const droppedCandle = ohlcData.shift();
                    console.log(`Dropped oldest candle (Timestamp: ${new Date(droppedCandle.timestamp).toLocaleTimeString()}).`);
                }
                
                await saveToCloudStorage(ohlcData);
                res.status(200).send('Collector saved new candle.'); 

            } else if (latestCandle.timestamp === lastLocalTimestamp) {
                console.log('No new candle closed since the last check. Skipping update.');
                res.status(200).send('No new candle, skipped update.'); 
            } else {
                console.warn('WARN: Fetched candle is older than the last one in GCS. Skipping update.');
                res.status(200).send('Fetched candle is older, skipped update.'); 
            }
        } else {
            console.log('Could not fetch a valid latest closed candle. No update performed.');
            res.status(200).send('Could not fetch valid candle, no update.'); 
        }
    } catch (e) {
        console.error('CRITICAL COLLECTOR ERROR:', e);
        res.status(500).send(`Collector failed: ${e.message}`); 
    }
};

function calculateMicrostructureFeatures(historicalData) {
    for (let i = 0; i < historicalData.length; i++) {
        const bar = historicalData[i];
        const { open, high, low, close, volume } = bar;
        
        if (!Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(open) || !Number.isFinite(close) || !Number.isFinite(volume)) {
             bar.UWP = bar.LWP = bar.VWBP = bar.RCPP = bar.IBPJ = 0;
             continue;
        }

        const range = high - low;
        const totalRangeDivisor = range > EPSILON ? range : EPSILON;
        const bodyTop = Math.max(open, close);
        const bodyBottom = Math.min(open, close);
        bar.UWP = (high - bodyTop) / totalRangeDivisor;
        bar.LWP = (bodyBottom - low) / totalRangeDivisor;
            
        let RCPP_val = (close - low) / totalRangeDivisor;
        bar.RCPP = Math.max(0, Math.min(1, RCPP_val)); 
        bar.VWBP = range * volume;

        if (i === 0) {
            bar.IBPJ = 0; // The first bar cannot have a jump
        } else {
            const prevClose = historicalData[i - 1].close;
            const prevCloseDivisor = prevClose > EPSILON ? prevClose : EPSILON;
            bar.IBPJ = (close - prevClose) / prevCloseDivisor;
        }
    }

    return historicalData;
}


function calculateRSI(closes, period) {
    if (closes.length < period + 1) return [];
    
    const changes = [];
    for (let i = 1; i < closes.length; i++) {
        changes.push(closes[i] - closes[i - 1]);
    }

    const RSIs = [];
    let avgGain = 0;
    let avgLoss = 0;

    for (let i = 0; i < period; i++) {
        const change = changes[i];
        if (change > 0) { avgGain += change; } else { avgLoss += Math.abs(change); }
    }
    avgGain /= period;
    avgLoss /= period;
    
    let RS = avgLoss === 0 ? 100 : avgGain / avgLoss; 
    RSIs.push(100 - (100 / (1 + RS)));

    for (let i = period; i < changes.length; i++) {
        const change = changes[i];
        let gain = 0;
        let loss = 0;
        if (change > 0) { gain = change; } else { loss = Math.abs(change); }
        avgGain = (avgGain * (period - 1) + gain) / period;
        avgLoss = (avgLoss * (period - 1) + loss) / period;
        
        RS = avgLoss === 0 ? 100 : avgGain / avgLoss;
        RSIs.push(100 - (100 / (1 + RS)));
    }
    return RSIs;
}

function calculateATR(highs, lows, closes, period) {
    if (highs.length < period) return [];

    const trueRanges = [];
    trueRanges.push(highs[0] - lows[0]); 

    for (let i = 1; i < highs.length; i++) {
        const tr = Math.max(
            highs[i] - lows[i], 
            Math.abs(highs[i] - closes[i - 1]), 
            Math.abs(lows[i] - closes[i - 1]) 
        );
        trueRanges.push(tr);
    }
    
    if (trueRanges.length < period) return [];
    
    const ATRs = [];
    
    let initialATR = trueRanges.slice(0, period).reduce((a, b) => a + b, 0) / period;
    ATRs.push(initialATR);
    
    for (let i = period; i < trueRanges.length; i++) {
        const currentTR = trueRanges[i];
        const prevATR = ATRs[ATRs.length - 1];
        const nextATR = ((prevATR * (period - 1)) + currentTR) / period;
        ATRs.push(nextATR);
    }
    return ATRs;
}

function calculateSMA(values, period) {
    if (values.length < period) return [];
    const SMAs = [];
    for (let i = period - 1; i < values.length; i++) {
        const slice = values.slice(i - period + 1, i + 1);
        const sum = slice.reduce((a, b) => a + b, 0);
        SMAs.push(sum / period);
    }
    return SMAs;
}

function calculateBollingerBands(closes, period, stdDevFactor) {
    const SMAs = calculateSMA(closes, period); 
    const BBs = [];

    for (let i = period - 1; i < closes.length; i++) {
        const currentSMA = SMAs[i - (period - 1)];
        const dataSlice = closes.slice(i - period + 1, i + 1);
        
        const variance = dataSlice.map(c => Math.pow(c - currentSMA, 2)).reduce((a, b) => a + b, 0) / period;
        const stdDev = Math.sqrt(variance);

        const upperBand = currentSMA + (stdDev * stdDevFactor);
        const lowerBand = currentSMA - (stdDev * stdDevFactor);
        
        const currentClose = closes[i];
        
        let percentB = 0.5; // Default if bands collapse
        
        const bandWidth = upperBand - lowerBand;
        
        if (bandWidth > EPSILON) {
            percentB = (currentClose - lowerBand) / bandWidth;
        } else if (currentClose > currentSMA) {
            percentB = 1.0; 
        } else if (currentClose < currentSMA) {
            percentB = 0.0; 
        }
        
        BBs.push({ percentB: percentB, sma: currentSMA, upper: upperBand, lower: lowerBand });
    }
    return BBs;
}


function prepareData(history) {
    const enrichedHistory = calculateMicrostructureFeatures(history);

    const requiredIndicatorLength = Math.max(
        PREDICTOR_RSI_PERIOD, 
        PREDICTOR_ATR_PERIOD, 
        PREDICTOR_SMA_LONG_PERIOD,
        PREDICTOR_BB_PERIOD
    ); 
    
    const requiredLength = requiredIndicatorLength + PREDICTOR_LOOKBACK_WINDOW + 1;

    console.log(`[PREP DEBUG] Total history length: ${history.length}. Required minimum for full indicators/lookback: ${requiredLength}.`);

    if (history.length < requiredLength) {
        console.warn(`[WARN] Data history length (${history.length}) is less than required minimum (${requiredLength}). Skipping preparation.`);
        return { features: null, labels: null, normalizationMin: [], normalizationMax: [], classCounts: { up: 0, down: 0 } };
    }

    const closes = enrichedHistory.map(c => c.close);
    const highs = enrichedHistory.map(c => c.high); 
    const lows = enrichedHistory.map(c => c.low); 
    
    const RSIs = calculateRSI(closes, PREDICTOR_RSI_PERIOD);
    const ATRs = calculateATR(highs, lows, closes, PREDICTOR_ATR_PERIOD); 
    const SMAShorts = calculateSMA(closes, PREDICTOR_SMA_SHORT_PERIOD);
    const SMALongs = calculateSMA(closes, PREDICTOR_SMA_LONG_PERIOD);
    const BBands = calculateBollingerBands(closes, PREDICTOR_BB_PERIOD, PREDICTOR_BB_STDDEV);
    const candleChanges = enrichedHistory.map(c => c.close - c.open);
    
    const features = [];
    const labels = [];
    let upCount = 0; 
    let downCount = 0; 
    
    const dataStartIndex = requiredIndicatorLength; 
    
    for (let i = dataStartIndex; i < enrichedHistory.length - 1; i++) {
        const rsiIndex = i - PREDICTOR_RSI_PERIOD; 
        const atrIndex = i - PREDICTOR_ATR_PERIOD; 
        const smaShortIndex = i - PREDICTOR_SMA_SHORT_PERIOD + 1; 
        const smaLongIndex = i - PREDICTOR_SMA_LONG_PERIOD + 1; 
        const bbIndex = i - PREDICTOR_BB_PERIOD + 1; 

        const currentRSI = RSIs[rsiIndex];
        const currentATR = ATRs[atrIndex]; 
        const currentBB = BBands[bbIndex];
        const currentClose = closes[i];
        
        const divisor = currentATR > EPSILON ? currentATR : currentClose; 

        const priceVsSMAShort_norm = (currentClose - SMAShorts[smaShortIndex]) / divisor; 
        const priceVsSMALong_norm = (currentClose - SMALongs[smaLongIndex]) / divisor; 
        
        const changeN_norm = candleChanges[i] / currentClose; 
        const changeNMinus1_norm = candleChanges[i - 1] / currentClose; 
        const changeNMinus2_norm = candleChanges[i - 2] / currentClose;

        const currentBar = enrichedHistory[i]; // Use the enriched bar
        const upperWickProportion = currentBar.UWP; 
        const lowerWickProportion = currentBar.LWP; 
        const volumeWeightedBarPower = currentBar.VWBP;
        const relativeClosingPricePosition = currentBar.RCPP;
        const interBarPriceJump = currentBar.IBPJ; 
        const featureVector = [
            currentRSI,
            currentATR,
            currentBB.percentB,
            priceVsSMAShort_norm,
            priceVsSMALong_norm,
            changeN_norm,
            changeNMinus1_norm,
            changeNMinus2_norm,
            (currentClose - currentBB.upper) / divisor,
            upperWickProportion,
            lowerWickProportion,
            volumeWeightedBarPower,
            relativeClosingPricePosition,
            interBarPriceJump
        ]; 
        
        if (featureVector.some(v => !Number.isFinite(v))) {
            console.warn(`[WARN] Skipping training data point at timestamp ${enrichedHistory[i].timestamp} due to non-finite feature value.`);
            continue;
        }

        features.push(featureVector);

        const nextCandleClose = closes[i + 1];
        const currentCandleClose = closes[i];
        const label = nextCandleClose > currentCandleClose ? 1 : 0; // 1 for UP, 0 for DOWN/FLAT
        labels.push(label);

        if (label === 1) {
            upCount++;
        } else {
            downCount++;
        }
    }
    
    if (features.length === 0) {
        console.error("CRITICAL: No valid training samples remain after filtering.");
        return { features: null, labels: null, normalizationMin: [], normalizationMax: [], classCounts: { up: 0, down: 0 } };
    }
    
    console.log(`[PREP DEBUG] Successfully created ${features.length} training samples with ${features[0].length} features.`);

    const featureTensor = tf.tensor2d(features);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
    
    const normalizationMin = featureTensor.min(0);
    const normalizationMax = featureTensor.max(0);
    
    const normalizedFeatures = featureTensor.sub(normalizationMin).div(normalizationMax.sub(normalizationMin).add(EPSILON));

    return { 
        features: normalizedFeatures, 
        labels: labelTensor, 
        normalizationMin: normalizationMin.arraySync(), 
        normalizationMax: normalizationMax.arraySync(),
        classCounts: { up: upCount, down: downCount }
    };
}


async function createAndTrainModel(features, labels, classWeight) {
    const model = tf.sequential();
    const inputShape = features.shape[1]; 

    model.add(tf.layers.dense({ inputShape: [inputShape], units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' })); 
    model.add(tf.layers.dense({ units: 64, activation: 'relu' })); 
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // Single output unit for binary classification

    model.compile({
        optimizer: tf.train.adam(0.005), 
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    console.log(`Training model on ${features.shape[0]} samples with ${inputShape} features...`);
    
    await model.fit(features, labels, {
        epochs: 75, 
        batchSize: 32,
        verbose: 0,
        classWeight: classWeight,
    });
    
    console.log('Model training complete.');
    return model;
}

async function executePredictionPipeline(fileMetadata) {
    const bucket = fileMetadata.bucket;
    const filename = fileMetadata.name;

    const file = storage.bucket(bucket).file(filename);
    const [contentBuffer] = await file.download();
    const jsonlContent = contentBuffer.toString();

    const dataHistory = jsonlContent.split('\n')
        .filter(line => line.trim().length > 0)
        .map(line => JSON.parse(line));
    
    console.log(`Successfully loaded ${dataHistory.length} clean records.`);

    const requiredIndicatorLength = Math.max(
        PREDICTOR_RSI_PERIOD, 
        PREDICTOR_ATR_PERIOD, 
        PREDICTOR_SMA_LONG_PERIOD,
        PREDICTOR_BB_PERIOD
    );
    const minLength = requiredIndicatorLength + PREDICTOR_LOOKBACK_WINDOW + 2; 

    if (dataHistory.length < minLength) {
        console.warn(`Not enough data (need >${minLength}) for ML prediction analysis. Waiting for more data.`);
        return; 
    }
    
    const enrichedHistory = calculateMicrostructureFeatures(dataHistory);

    const candleN = enrichedHistory[enrichedHistory.length - 1]; // Current candle for prediction input (T)
    const candleNMinus1 = enrichedHistory[enrichedHistory.length - 2]; // Candle whose outcome is validated (T-1)

    const batch = db.batch();
    dataHistory.forEach(candle => {
        const docRef = CANDLE_COLLECTION.doc(candle.timestamp.toString());
        const { UWP, LWP, VWBP, RCPP, IBPJ, ...baseCandle } = candle; 
        batch.set(docRef, baseCandle);
    });
    
    await batch.commit();
    console.log(`[FIRESTORE] Successfully updated ${dataHistory.length} candle documents in ${CANDLE_COLLECTION_PATH} via batch.`);

    
    const trainingHistory = enrichedHistory.slice(0, -1);
    console.log(`[PIPE DEBUG] DataHistory length: ${enrichedHistory.length}. TrainingHistory length: ${trainingHistory.length}.`);
    
    const { features, labels, normalizationMin, normalizationMax, classCounts } = prepareData(trainingHistory);

    if (!features) {
        console.warn('Features could not be prepared or no valid training samples were found. Skipping prediction.');
        return;
    }
    
    const totalSamples = classCounts.up + classCounts.down;
    const weightDown = classCounts.down > 0 ? totalSamples / (2 * classCounts.down) : 1.0; 
    const weightUp = classCounts.up > 0 ? totalSamples / (2 * classCounts.up) : 1.0; 
    const classWeight = { 0: weightDown, 1: weightUp };

    console.log(`[ML WEIGHT] Total Samples: ${totalSamples}. Down (0): ${classCounts.down}, Up (1): ${classCounts.up}`);
    console.log(`[ML WEIGHT] Class Weights (0: DOWN, 1: UP): {0: ${weightDown.toFixed(2)}, 1: ${weightUp.toFixed(2)}}`);

    let model;
    try {
        model = await createAndTrainModel(features, labels, classWeight); 
    } catch (e) {
        console.error("Model training failed:", e.message);
        tf.dispose([features, labels]); 
        return;
    }
    tf.dispose([features, labels]); 

    
    const metricsRef = db.doc(METRICS_DOC_PATH);
    const predictionsRef = db.collection(PREDICTIONS_COLLECTION_PATH);

    try {
        await db.runTransaction(async (t) => {
            const metricsSnapshot = await t.get(metricsRef);
            
            let metrics = {
                lastPredictedCandleTimestamp: 0,
                totalPredictions: 0,
                correctPredictions: 0,
                accuracy: 0,
                lastPrediction: 'N/A',
                confidence: 0,
                predictionMadeAtTimestamp: 0
            };
            if (metricsSnapshot.exists) {
                metrics = metricsSnapshot.data();
                metrics.totalPredictions = metrics.totalPredictions || 0;
                metrics.correctPredictions = metrics.correctPredictions || 0;
            }

            // --- T+1 ACCURACY CHECK (UPDATED LOGIC) ---
            const predictedInputTimestamp = metrics.lastPredictedCandleTimestamp;
            const validationInputTimestamp = candleNMinus1.timestamp;
            
            if (predictedInputTimestamp === validationInputTimestamp) {
                // T+1: This prediction was made using features from candleNMinus1 (the validation target).
                // The actual outcome is revealed by candleN.
                const actualResult = candleN.close > candleNMinus1.close ? 'UP' : 'DOWN';
                const prediction = metrics.lastPrediction;
                const isCorrect = actualResult === prediction;
                
                console.log(`--- PREVIOUS PREDICTION CHECK (T+1) ---`);
                console.log(`Predicted: ${prediction}, Actual: ${actualResult}, Result: ${isCorrect ? 'CORRECT' : 'INCORRECT'} (Input Candlestick: ${new Date(validationInputTimestamp).toLocaleTimeString()})`);
                
                // Only count towards accuracy if it was a T+1 check
                metrics.totalPredictions += 1;
                if (isCorrect) {
                    metrics.correctPredictions += 1;
                }
            } else if (predictedInputTimestamp > 0 && predictedInputTimestamp < validationInputTimestamp) {
                // T+2 or later: The data jump means we skipped validation for the prediction made at predictedInputTimestamp.
                console.log(`Skipping validation (T+2+): Last prediction was for the candle at ${new Date(predictedInputTimestamp).toLocaleTimeString()}, but current data jumped to ${new Date(validationInputTimestamp).toLocaleTimeString()}. Accuracy score ignored.`);
            } else {
                // N/A: No previous prediction, or first run.
                console.log(`Skipping validation: last prediction was for a different or non-existent candle. Current validation input: ${new Date(validationInputTimestamp).toLocaleTimeString()}.`);
            }
            // --- END T+1 ACCURACY CHECK ---

            const closes = enrichedHistory.map(c => c.close);
            const highs = enrichedHistory.map(c => c.high);
            const lows = enrichedHistory.map(c => c.low);
            
            const RSIs = calculateRSI(closes, PREDICTOR_RSI_PERIOD);
            const ATRs = calculateATR(highs, lows, closes, PREDICTOR_ATR_PERIOD); 
            const SMAShorts = calculateSMA(closes, PREDICTOR_SMA_SHORT_PERIOD);
            const SMALongs = calculateSMA(closes, PREDICTOR_SMA_LONG_PERIOD);
            const BBands = calculateBollingerBands(closes, PREDICTOR_BB_PERIOD, PREDICTOR_BB_STDDEV);
            const candleChanges = enrichedHistory.map(c => c.close - c.open);
            
            const latestRSI = RSIs[RSIs.length - 1];
            const latestATR = ATRs[ATRs.length - 1]; 
            const latestClose = closes[closes.length - 1];
            const latestSMAShort = SMAShorts[SMAShorts.length - 1];
            const latestSMALong = SMALongs[SMALongs.length - 1];
            const latestBB = BBands[BBands.length - 1];
            
            const divisor = latestATR > EPSILON ? latestATR : latestClose; 
            const priceVsSMAShort_norm = (latestClose - latestSMAShort) / divisor;
            const priceVsSMALong_norm = (latestClose - latestSMALong) / divisor;
            
            const changeN_norm = candleChanges[candleChanges.length - 1] / latestClose;
            const changeNMinus1_norm = candleChanges[candleChanges.length - 2] / latestClose;
            const changeNMinus2_norm = candleChanges[candleChanges.length - 3] / latestClose;
            
            const latestUpperWickProportion = candleN.UWP; 
            const latestLowerWickProportion = candleN.LWP; 
            const latestVolumeWeightedBarPower = candleN.VWBP;
            const latestRelativeClosingPricePosition = candleN.RCPP;
            const latestInterBarPriceJump = candleN.IBPJ;

            const inputFeatures = [
                latestRSI, 
                latestATR, 
                latestBB.percentB, 
                priceVsSMAShort_norm, 
                priceVsSMALong_norm, 
                changeN_norm, 
                changeNMinus1_norm, 
                changeNMinus2_norm, 
                (latestClose - latestBB.upper) / divisor,
                latestUpperWickProportion, 
                latestLowerWickProportion, 
                latestVolumeWeightedBarPower,
                latestRelativeClosingPricePosition,
                latestInterBarPriceJump,
            ];
            
            console.log(`[PIPE DEBUG] Prediction input features (raw, ${inputFeatures.length} features): [${inputFeatures.map(v => v.toFixed(4)).join(', ')}]`);


            const normalizedInputFeatures = inputFeatures.map((value, index) => {
                const min = normalizationMin[index];
                const max = normalizationMax[index];

                if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) < EPSILON) {
                    console.warn(`[WARN] Normalization failed for feature index ${index}. Max: ${max}, Min: ${min}. Setting to 0.`);
                    return 0; 
                }
                
                const normalizedValue = (value - min) / (max - min + EPSILON); 
                return Number.isFinite(normalizedValue) ? normalizedValue : 0;
            });
            
            const inputTensor = tf.tensor2d([normalizedInputFeatures]);
            const predictionOutput = model.predict(inputTensor);
            const predictionValue = predictionOutput.dataSync()[0]; 

            tf.dispose([predictionOutput, inputTensor]);
            
            let newPrediction = 'UNKNOWN';
            let confidence = 0;

            if (isNaN(predictionValue) || !isFinite(predictionValue)) {
                console.error("Prediction value resulted in NaN or Infinity. Defaulting to 'UNKNOWN'.");
            } else {
                newPrediction = predictionValue >= 0.5 ? 'UP' : 'DOWN';
                confidence = Math.max(predictionValue, 1 - predictionValue); 
            }

            metrics.lastPrediction = newPrediction;
            metrics.confidence = confidence;
            metrics.predictionMadeAtTimestamp = Date.now();
            metrics.lastPredictedCandleTimestamp = candleN.timestamp; // The features of this candle (T) were used to predict T+1
            
            if (metrics.totalPredictions > 0) {
                metrics.accuracy = (metrics.correctPredictions / metrics.totalPredictions) * 100;
            } else {
                metrics.accuracy = 0;
            }
            
            // Per-prediction record for confidence vs accuracy analysis
            const predictionDocId = candleN.timestamp.toString();
            t.set(predictionsRef.doc(predictionDocId), {
                predictionMadeAt: Date.now(),
                inputCandleTimestamp: candleNMinus1.timestamp,
                targetCandleTimestamp: candleN.timestamp,
                predicted: newPrediction,
                confidence: confidence,
                actual: null,
                correct: null,
                labeled: false,
                interval: COLLECTOR_INTERVAL,
            }, { merge: true });

            t.set(metricsRef, metrics);
            
            console.log(`[ML PREDICTION] Predicted '${metrics.lastPrediction}' for next candle with ${Math.floor(confidence * 100)}% confidence.`);
        });
        
        tf.dispose(model); 
    } catch (e) {
        console.error('CRITICAL FIRESTORE/PREDICTION ERROR:', e);
        if (model) tf.dispose(model);
        throw e;
    }
}


exports.cryptoPredictor = async (file, context) => {
    console.log(`\n--- Starting Predictor Execution for file ${file.name} at ${context.timestamp} ---`);
    if (file.resourceState === 'not_exists') {
        console.log('File deleted, nothing to do.');
        return;
    }
    if (file.bucket !== BUCKET_NAME || file.name !== FILE_NAME) {
            console.log(`Ignoring file change for non-target file: gs://${file.bucket}/${file.name}`);
            return;
    }
    
    try {
        await executePredictionPipeline(file);
    } catch (e) {
        console.error('Predictor failed execution:', e.message);
        throw e; 
    }
    console.log(`--- Predictor Execution Finished at ${new Date().toISOString()} ---`);
};
