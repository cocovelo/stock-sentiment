# Crypto Real-Time Data Collector

This project provides a simple Google Cloud Function (HTTP-triggered) to fetch crypto prices, append them to a Cloud Storage object (JSON Lines), perform lightweight analysis (simple moving averages), and write analysis back to the bucket.

Features
- Appends raw price records as JSONL to `crypto_data.jsonl` (configurable)
- Computes short and long simple moving averages and writes `crypto_analysis.json`
- Works in two modes:
  - Cloud mode: uses `BUCKET_NAME` env var and GCS
  - Local mode: no `BUCKET_NAME` -> writes to `local_data/` and supports `SIMULATE_PRICE=true`

Environment variables
- BUCKET_NAME: (optional) GCS bucket name to use
- RAW_OBJECT: object name for raw JSONL (default `crypto_data.jsonl`)
- ANALYSIS_OBJECT: object name for analysis JSON (default `crypto_analysis.json`)
- CRYPTO_PAIR: human-readable pair (default `BTC-USD`)
- PRICE_API: API endpoint to fetch price (default CoinCap for Bitcoin)
- SIMULATE_PRICE: when `true` the function simulates price locally for offline testing

Local testing

Install deps:

```powershell
cd path\to\crypto-real-time-data-collector; npm install
```

Run a short local simulation (5s interval, 12 runs):

```powershell
$env:SIMULATE_PRICE = 'true'; npm run run-local
```

Deployment (Google Cloud Functions)

- Ensure `package.json` and `index.js` are present.
- Deploy as an HTTP-triggered function with environment variables set (BUCKET_NAME should be set):

```powershell
gcloud functions deploy cryptoCollector --runtime=nodejs20 --trigger-http --allow-unauthenticated --set-env-vars "BUCKET_NAME=your-bucket,CRYPTO_PAIR=BTC-USD" --region=us-central1
```

- Use Cloud Scheduler to POST the function's HTTPS URL on your desired interval.

Notes
- This implementation keeps the raw object as a whole when appending (downloads, appends, re-uploads). For large datasets consider using object composition or writing individual files per run and using a lifecycle policy.
- The analysis is intentionally lightweight; you can extend it to compute indicators or stream to BigQuery.
