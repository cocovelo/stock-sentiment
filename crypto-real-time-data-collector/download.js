/**
 * GCS File Downloader
 *
 * This script downloads a specific file (crypto_data.jsonl) from the 
 * Google Cloud Storage bucket used by the crypto collector to your local
 * machine.
 *
 * Prerequisites:
 * 1. Node.js installed.
 * 2. The @google-cloud/storage package installed: npm install @google-cloud/storage
 * 3. Proper authentication set up for your local machine, typically via:
 * gcloud auth application-default login
 */
const { Storage } = require('@google-cloud/storage');
const path = require('path');
const fs = require('fs');

// --- Configuration ---
const BUCKET_NAME = 'client-project-viewer.firebasestorage.app';
const OBJECT_NAME = 'crypto_data.jsonl';
const DESTINATION_FILE = path.join(process.cwd(), 'crypto_data_local.jsonl');

// Initialize the GCS Client
// It uses Application Default Credentials (ADC) for authentication.
const storage = new Storage();

/**
 * Downloads a file from GCS to a local path.
 * The file is assumed to be publicly readable based on the application's context.
 */
async function downloadGCSFile() {
  console.log(`Attempting to download gs://${BUCKET_NAME}/${OBJECT_NAME}`);
  console.log(`Saving to local path: ${DESTINATION_FILE}`);

  try {
    const options = {
      destination: DESTINATION_FILE,
    };

    // Reference the file and execute the download command
    await storage.bucket(BUCKET_NAME).file(OBJECT_NAME).download(options);

    // Confirm success
    const stats = fs.statSync(DESTINATION_FILE);
    console.log('\n--- Download Successful ---');
    console.log(`File saved as: ${DESTINATION_FILE}`);
    console.log(`File size: ${Math.round(stats.size / 1024)} KB`);
    console.log('You can now inspect the raw data.');

  } catch (err) {
    console.error('\n--- Download Failed ---');
    console.error('An error occurred during download. Ensure you have the @google-cloud/storage library installed and have authenticated via `gcloud auth application-default login`.');
    // Log the full error for debugging purposes
    console.error(err.message || err);
  }
}

// Execute the function
downloadGCSFile();
