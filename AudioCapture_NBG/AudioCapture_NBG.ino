#include <driver/i2s.h>
#include <LittleFS.h>
#include <dsps_fft2r.h>
#include <dsps_wind.h>
#include <math.h>

// I2S Microphone pins (INMP441)
#define I2S_SD 41    // Serial Data
#define I2S_WS 7     // Word Select (LRCLK)
#define I2S_SCK 15   // Serial Clock (BCLK)

// Button pin
#define BUTTON_PIN 16

// I2S Configuration
#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define BITS_PER_SAMPLE I2S_BITS_PER_SAMPLE_32BIT
#define I2S_READ_LEN (1024)

// MFCC Configuration - Matching librosa parameters
// librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=4096, hop_length=512)
// Note: hop_length=512 is intentionally smaller than the librosa default (n_fft/4=1024)
//       to achieve finer temporal resolution for classification.
#define N_FFT 4096                     // n_fft parameter (what librosa uses)
#define HOP_LENGTH 512                 // hop_length parameter
#define FFT_SIZE 4096                  // ESP-DSP max supported size (equals N_FFT)
#define NUM_MFCC 13
#define NUM_MEL_FILTERS 128            // librosa default
#define MEL_LOW_FREQ 0                 // librosa default
#define MEL_HIGH_FREQ (SAMPLE_RATE / 2)

// Audio buffer management - decoupled from MFCC parameters
// Buffer holds N_FFT samples for the current frame plus HOP_LENGTH for the next hop
#define AUDIO_BUFFER_SIZE (N_FFT + HOP_LENGTH)

// Derived timing values (for display/metadata only, not buffer control)
#define FRAME_SIZE_MS ((N_FFT * 1000) / SAMPLE_RATE)      // ~256ms
#define FRAME_STRIDE_MS ((HOP_LENGTH * 1000) / SAMPLE_RATE)  // 32ms

// Classifier Configuration
#define NUM_CLASSES 5  // Change this to match your number of classes
#define ENABLE_REALTIME_CLASSIFICATION false  // Set to true to classify during recording

// WAV file configuration
#define WAV_FILE_PATH "/recording.wav"
#define MFCC_FILE_PATH "/mfcc_features.bin"
#define CLASSIFIER_MODEL_PATH "/classifier_model.bin"
#define WAV_HEADER_SIZE 44

// Recording state
bool isRecording = false;
bool lastButtonState = HIGH;
File wavFile;
File mfccFile;
uint32_t recordedSamples = 0;
uint32_t mfccFrameCount = 0;

// MFCC buffers - moved to PSRAM using pointers
int16_t* audioBuffer = nullptr;  // Will allocate in PSRAM
uint32_t bufferIndex = 0;

// FFT buffers (ESP-DSP requires 2*N for complex FFT) - moved to PSRAM
float* fft_input = nullptr;
float* fft_output = nullptr;
float* window = nullptr;

// Mel filterbank - moved to PSRAM
float** melFilterbank = nullptr;
float mfcc_coeffs[NUM_MFCC];

// MFCC accumulator for calculating mean across all frames
struct MFCCAccumulator {
  float sum[NUM_MFCC];
  uint32_t frameCount;
  
  void reset() {
    for (int i = 0; i < NUM_MFCC; i++) {
      sum[i] = 0.0;
    }
    frameCount = 0;
  }
  
  void addFrame(float* mfcc) {
    for (int i = 0; i < NUM_MFCC; i++) {
      sum[i] += mfcc[i];
    }
    frameCount++;
  }
  
  void getMean(float* mean) {
    if (frameCount > 0) {
      for (int i = 0; i < NUM_MFCC; i++) {
        mean[i] = sum[i] / frameCount;
      }
    } else {
      for (int i = 0; i < NUM_MFCC; i++) {
        mean[i] = 0.0;
      }
    }
  }
};

MFCCAccumulator mfccAccumulator;

// Naive Gaussian Classifier
struct GaussianClassifier {
  char classNames[NUM_CLASSES][32];  // Class labels
  float means[NUM_CLASSES][NUM_MFCC];  // Mean for each feature per class
  float variances[NUM_CLASSES][NUM_MFCC];  // Variance for each feature per class
  float priors[NUM_CLASSES];  // Prior probabilities (class frequencies)
  bool isLoaded;
};

GaussianClassifier classifier;

// WAV header structure
struct WAVHeader {
  // RIFF Header
  char riff[4] = {'R', 'I', 'F', 'F'};
  uint32_t fileSize;
  char wave[4] = {'W', 'A', 'V', 'E'};
  
  // Format chunk
  char fmt[4] = {'f', 'm', 't', ' '};
  uint32_t fmtSize = 16;
  uint16_t audioFormat = 1; // PCM
  uint16_t numChannels = 1; // Mono
  uint32_t sampleRate = SAMPLE_RATE;
  uint32_t byteRate;
  uint16_t blockAlign;
  uint16_t bitsPerSample = 16;
  
  // Data chunk
  char data[4] = {'d', 'a', 't', 'a'};
  uint32_t dataSize;
};

// ==================== Memory Allocation ====================

bool allocateBuffers() {
  Serial.println("Allocating buffers in PSRAM...");
  
  // Allocate audio buffer sized for sliding window (N_FFT + HOP_LENGTH samples)
  audioBuffer = (int16_t*)ps_malloc(AUDIO_BUFFER_SIZE * sizeof(int16_t));
  if (!audioBuffer) {
    Serial.println("Failed to allocate audioBuffer");
    return false;
  }
  memset(audioBuffer, 0, AUDIO_BUFFER_SIZE * sizeof(int16_t));
  
  // Allocate FFT buffers
  fft_input = (float*)ps_malloc(FFT_SIZE * 2 * sizeof(float));
  if (!fft_input) {
    Serial.println("Failed to allocate fft_input");
    return false;
  }
  
  fft_output = (float*)ps_malloc(FFT_SIZE * sizeof(float));
  if (!fft_output) {
    Serial.println("Failed to allocate fft_output");
    return false;
  }
  
  window = (float*)ps_malloc(N_FFT * sizeof(float));  // Window size is N_FFT
  if (!window) {
    Serial.println("Failed to allocate window");
    return false;
  }
  
  // Allocate Mel filterbank (2D array)
  // Filterbank is based on effective FFT bins
  int numBins = FFT_SIZE / 2 + 1;
  
  melFilterbank = (float**)ps_malloc(NUM_MEL_FILTERS * sizeof(float*));
  if (!melFilterbank) {
    Serial.println("Failed to allocate melFilterbank");
    return false;
  }
  
  for (int i = 0; i < NUM_MEL_FILTERS; i++) {
    melFilterbank[i] = (float*)ps_malloc(numBins * sizeof(float));
    if (!melFilterbank[i]) {
      Serial.println("Failed to allocate melFilterbank row");
      return false;
    }
    memset(melFilterbank[i], 0, numBins * sizeof(float));
  }
  
  Serial.println("All buffers allocated successfully");
  
  // Print memory stats
  Serial.print("Free heap: ");
  Serial.print(ESP.getFreeHeap());
  Serial.println(" bytes");
  Serial.print("Free PSRAM: ");
  Serial.print(ESP.getFreePsram());
  Serial.println(" bytes");
  
  return true;
}

void freeBuffers() {
  if (audioBuffer) free(audioBuffer);
  if (fft_input) free(fft_input);
  if (fft_output) free(fft_output);
  if (window) free(window);
  
  if (melFilterbank) {
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
      if (melFilterbank[i]) free(melFilterbank[i]);
    }
    free(melFilterbank);
  }
}

// ==================== Classifier Functions ====================

// Load classifier model from file
bool loadClassifierModel() {
  File modelFile = LittleFS.open(CLASSIFIER_MODEL_PATH, FILE_READ);
  if (!modelFile) {
    Serial.println("Classifier model file not found.");
    Serial.println("Please upload 'classifier_model.bin' to LittleFS.");
    classifier.isLoaded = false;
    return false;
  }
  
  // Read number of classes
  uint32_t numClasses;
  modelFile.read((uint8_t*)&numClasses, sizeof(uint32_t));
  
  if (numClasses != NUM_CLASSES) {
    Serial.print("Model class count mismatch! Expected ");
    Serial.print(NUM_CLASSES);
    Serial.print(" but got ");
    Serial.println(numClasses);
    modelFile.close();
    classifier.isLoaded = false;
    return false;
  }
  
  // Read class names
  for (int i = 0; i < NUM_CLASSES; i++) {
    modelFile.read((uint8_t*)classifier.classNames[i], 32);
  }
  
  // Read means
  for (int i = 0; i < NUM_CLASSES; i++) {
    modelFile.read((uint8_t*)classifier.means[i], NUM_MFCC * sizeof(float));
  }
  
  // Read variances
  for (int i = 0; i < NUM_CLASSES; i++) {
    modelFile.read((uint8_t*)classifier.variances[i], NUM_MFCC * sizeof(float));
  }
  
  // Read priors
  modelFile.read((uint8_t*)classifier.priors, NUM_CLASSES * sizeof(float));
  
  modelFile.close();
  classifier.isLoaded = true;
  
  Serial.println("Classifier model loaded successfully!");
  printClassifierModel();
  return true;
}

// Print classifier model for debugging
void printClassifierModel() {
  Serial.println("\n=== Classifier Model ===");
  Serial.print("Number of classes: "); Serial.println(NUM_CLASSES);
  
  for (int c = 0; c < NUM_CLASSES; c++) {
    Serial.print("\nClass "); Serial.print(c); Serial.print(": ");
    Serial.println(classifier.classNames[c]);
    Serial.print("  Prior probability: "); Serial.println(classifier.priors[c], 4);
    
    Serial.print("  Means: [");
    for (int i = 0; i < min(5, NUM_MFCC); i++) {
      Serial.print(classifier.means[c][i], 3);
      if (i < 4) Serial.print(", ");
    }
    Serial.println("...]");
    
    Serial.print("  Variances: [");
    for (int i = 0; i < min(5, NUM_MFCC); i++) {
      Serial.print(classifier.variances[c][i], 3);
      if (i < 4) Serial.print(", ");
    }
    Serial.println("...]");
  }
  Serial.println("========================\n");
}

// Gaussian probability density function
float gaussianPDF(float x, float mean, float variance) {
  if (variance <= 0) variance = 1e-6;
  float exponent = -0.5 * pow((x - mean), 2) / variance;
  float coefficient = 1.0 / sqrt(2.0 * M_PI * variance);
  return coefficient * exp(exponent);
}

// Predict class for a single MFCC frame using Naive Bayes
int predictClass(float* mfcc_frame, float* probabilities) {
  float logProbs[NUM_CLASSES];
  
  for (int c = 0; c < NUM_CLASSES; c++) {
    logProbs[c] = log(classifier.priors[c] + 1e-10);
    
    for (int i = 0; i < NUM_MFCC; i++) {
      float prob = gaussianPDF(mfcc_frame[i], classifier.means[c][i], classifier.variances[c][i]);
      logProbs[c] += log(prob + 1e-10);
    }
  }
  
  int maxClass = 0;
  float maxLogProb = logProbs[0];
  
  for (int c = 1; c < NUM_CLASSES; c++) {
    if (logProbs[c] > maxLogProb) {
      maxLogProb = logProbs[c];
      maxClass = c;
    }
  }
  
  if (probabilities != NULL) {
    float sumExp = 0.0;
    for (int c = 0; c < NUM_CLASSES; c++) {
      probabilities[c] = exp(logProbs[c] - maxLogProb);
      sumExp += probabilities[c];
    }
    for (int c = 0; c < NUM_CLASSES; c++) {
      probabilities[c] /= sumExp;
    }
  }
  
  return maxClass;
}

// Classify stored MFCC file using mean of all frames
void classifyStoredMFCC() {
  if (!classifier.isLoaded) {
    Serial.println("ERROR: Classifier model not loaded!");
    return;
  }
  
  File f = LittleFS.open(MFCC_FILE_PATH, FILE_READ);
  if (!f) {
    Serial.println("Failed to open MFCC file");
    return;
  }
  
  uint32_t numCoeffs, frameSizeMs, frameStrideMs, sampleRate;
  f.read((uint8_t*)&numCoeffs, sizeof(uint32_t));
  f.read((uint8_t*)&frameSizeMs, sizeof(uint32_t));
  f.read((uint8_t*)&frameStrideMs, sizeof(uint32_t));
  f.read((uint8_t*)&sampleRate, sizeof(uint32_t));
  
  Serial.println("\n=== Classifying Stored MFCC Features ===");
  Serial.print("Frame Size: "); Serial.print(N_FFT); Serial.print(" samples ("); 
  Serial.print(frameSizeMs); Serial.println(" ms)");
  Serial.print("Hop Length: "); Serial.print(HOP_LENGTH); Serial.print(" samples (");
  Serial.print(frameStrideMs); Serial.println(" ms)");
  Serial.print("Frame Overlap: "); Serial.print(N_FFT - HOP_LENGTH); Serial.println(" samples");
  Serial.println();
  
  // Accumulate all MFCC frames
  MFCCAccumulator accumulator;
  accumulator.reset();
  
  float coeffs[NUM_MFCC];
  int frameNum = 0;
  
  Serial.println("Reading all frames and computing mean MFCC...");
  while (f.available() >= NUM_MFCC * sizeof(float)) {
    f.read((uint8_t*)coeffs, NUM_MFCC * sizeof(float));
    accumulator.addFrame(coeffs);
    
    // Show progress every 10 frames
    if (frameNum % 10 == 0) {
      Serial.print("  Processed ");
      Serial.print(frameNum);
      Serial.println(" frames...");
    }
    
    frameNum++;
  }
  
  f.close();
  
  Serial.print("Total frames read: ");
  Serial.println(frameNum);
  
  if (frameNum == 0) {
    Serial.println("ERROR: No MFCC frames found!");
    return;
  }
  
  // Calculate mean MFCC across all frames
  float meanMFCC[NUM_MFCC];
  accumulator.getMean(meanMFCC);
  
  Serial.println("\nMean MFCC coefficients across all frames:");
  Serial.print("  [");
  for (int i = 0; i < NUM_MFCC; i++) {
    Serial.print(meanMFCC[i], 4);
    if (i < NUM_MFCC - 1) Serial.print(", ");
  }
  Serial.println("]");
  
  // Classify using mean MFCC
  float probabilities[NUM_CLASSES];
  int predictedClass = predictClass(meanMFCC, probabilities);
  
  // Print results
  Serial.println("\n=== Classification Result ===");
  Serial.print("Total duration: ");
  Serial.print((frameNum * frameStrideMs) / 1000.0, 2);
  Serial.println(" seconds");
  Serial.print("Number of frames averaged: ");
  Serial.println(frameNum);
  Serial.println();
  
  Serial.print("Predicted class: ");
  Serial.println(classifier.classNames[predictedClass]);
  Serial.print("Confidence: ");
  Serial.print(probabilities[predictedClass] * 100, 2);
  Serial.println("%");
  Serial.println();
  
  Serial.println("Class probabilities:");
  for (int c = 0; c < NUM_CLASSES; c++) {
    Serial.print("  ");
    Serial.print(classifier.classNames[c]);
    Serial.print(": ");
    Serial.print(probabilities[c] * 100, 2);
    Serial.println("%");
  }
  Serial.println("==============================\n");
}

// ==================== MFCC Functions (Librosa-compatible with FFT size limitation) ====================

// Hz to Mel conversion (HTK formula - used by librosa by default)
float hzToMel(float hz) {
  return 2595.0 * log10(1.0 + hz / 700.0);
}

// Mel to Hz conversion
float melToHz(float mel) {
  return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

// Initialize Mel filterbank (librosa-compatible, scaled for actual FFT size)
void initMelFilterbank() {
  float melLow = hzToMel(MEL_LOW_FREQ);
  float melHigh = hzToMel(MEL_HIGH_FREQ);
  
  // Create mel points (linearly spaced in mel scale)
  float melPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    melPoints[i] = melLow + (melHigh - melLow) * i / (NUM_MEL_FILTERS + 1);
  }
  
  // Convert mel points to Hz
  float hzPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    hzPoints[i] = melToHz(melPoints[i]);
  }
  
  // Convert Hz to FFT bin indices
  // Use FFT_SIZE instead of N_FFT to match actual FFT performed
  int binPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    binPoints[i] = (int)floor((FFT_SIZE + 1) * hzPoints[i] / SAMPLE_RATE);
  }
  
  int numBins = FFT_SIZE / 2 + 1;
  
  // Create triangular filters (librosa style)
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    int leftBin = binPoints[m];
    int centerBin = binPoints[m + 1];
    int rightBin = binPoints[m + 2];
    
    // Initialize all to zero
    for (int k = 0; k < numBins; k++) {
      melFilterbank[m][k] = 0.0;
    }
    
    // Left slope
    for (int k = leftBin; k < centerBin && k < numBins; k++) {
      if (centerBin != leftBin) {
        melFilterbank[m][k] = (float)(k - leftBin) / (centerBin - leftBin);
      }
    }
    
    // Right slope
    for (int k = centerBin; k < rightBin && k < numBins; k++) {
      if (rightBin != centerBin) {
        melFilterbank[m][k] = (float)(rightBin - k) / (rightBin - centerBin);
      }
    }
  }
  
  // Normalize by mel bandwidth (librosa does this)
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    float enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m]);
    for (int k = 0; k < numBins; k++) {
      melFilterbank[m][k] *= enorm;
    }
  }
}

// Apply Hann window (librosa uses Hann by default)
void applyHannWindow(int16_t* input, float* output, int length) {
  for (int i = 0; i < length; i++) {
    // Normalize int16 to float [-1.0, 1.0]
    float normalized = input[i] / 32768.0f;
    output[i] = normalized * window[i];
  }
}

// Compute power spectrum using ESP-DSP FFT
// Since ESP-DSP only supports up to 4096, we'll use the first 4096 samples
void computePowerSpectrum(float* input, float* output, int length) {
  for (int i = 0; i < length; i++) {
    fft_input[i * 2] = input[i];
    fft_input[i * 2 + 1] = 0.0;
  }
  
  for (int i = length; i < FFT_SIZE; i++) {
    fft_input[i * 2] = 0.0;
    fft_input[i * 2 + 1] = 0.0;
  }
  
  dsps_fft2r_fc32(fft_input, FFT_SIZE);
  dsps_bit_rev_fc32(fft_input, FFT_SIZE);
  
  int numBins = FFT_SIZE / 2 + 1;
  for (int i = 0; i < numBins; i++) {
    float real = fft_input[i * 2];
    float imag = fft_input[i * 2 + 1];
    output[i] = (real * real + imag * imag) / (float)FFT_SIZE;  // Normalize by FFT size
  }
}

void applyMelFilters(float* powerSpectrum, float* melEnergies) {
  int numBins = FFT_SIZE / 2 + 1;
  
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    double energy = 0.0;  // Use double for accumulation
    for (int k = 0; k < numBins; k++) {
      energy += (double)powerSpectrum[k] * (double)melFilterbank[m][k];
    }
    // librosa uses natural log (ln), not log10!
    melEnergies[m] = log(energy + 1e-10);
  }
}

// Compute DCT Type-II (librosa uses scipy.fftpack.dct with norm='ortho')
void computeDCT(float* melEnergies, float* mfcc, int numFilters, int numCoeffs) {
  for (int i = 0; i < numCoeffs; i++) {
    float sum = 0.0;
    for (int j = 0; j < numFilters; j++) {
      sum += melEnergies[j] * cos(M_PI * i * (j + 0.5) / numFilters);
    }
    
    // Apply orthonormal normalization (librosa default)
    if (i == 0) {
      mfcc[i] = sum * sqrt(1.0 / numFilters);
    } else {
      mfcc[i] = sum * sqrt(2.0 / numFilters);
    }
  }
}

// Main MFCC extraction function (librosa-compatible with FFT limitation workaround)
void extractMFCC(int16_t* frame) {
  static float* windowedFrame = nullptr;
  static float* powerSpectrum = nullptr;
  static float* melEnergies = nullptr;
  
  // Allocate temporary buffers on first call
  if (!windowedFrame) {
    windowedFrame = (float*)ps_malloc(N_FFT * sizeof(float));
    powerSpectrum = (float*)ps_malloc((FFT_SIZE / 2 + 1) * sizeof(float));
    melEnergies = (float*)ps_malloc(NUM_MEL_FILTERS * sizeof(float));
  }
  
  // 1. Apply Hann window (librosa default) - apply to full N_FFT samples
  applyHannWindow(frame, windowedFrame, N_FFT);
  
  // 2. Compute power spectrum using ESP-DSP FFT (limited to 4096 samples)
  computePowerSpectrum(windowedFrame, powerSpectrum, N_FFT);
  
  // 3. Apply Mel filterbank
  applyMelFilters(powerSpectrum, melEnergies);
  
  // 4. Compute DCT to get MFCC coefficients (with orthonormal normalization)
  computeDCT(melEnergies, mfcc_coeffs, NUM_MEL_FILTERS, NUM_MFCC);
  
  // 5. Write MFCC coefficients to file
  if (mfccFile) {
    mfccFile.write((uint8_t*)mfcc_coeffs, NUM_MFCC * sizeof(float));
    mfccFrameCount++;
  }
  
  // Accumulate for mean calculation during recording
  if (isRecording) {
    mfccAccumulator.addFrame(mfcc_coeffs);
  }
  
  #if ENABLE_REALTIME_CLASSIFICATION
  if (classifier.isLoaded) {
    float probabilities[NUM_CLASSES];
    int predictedClass = predictClass(mfcc_coeffs, probabilities);
    
    float currentTime = (mfccFrameCount * FRAME_STRIDE_MS) / 1000.0;
    Serial.print("["); Serial.print(currentTime, 2); Serial.print("s] ");
    Serial.print(classifier.classNames[predictedClass]);
    Serial.print(" ("); Serial.print(probabilities[predictedClass] * 100, 1);
    Serial.println("%)");
  }
  #endif
}

void listFiles() {
  Serial.println("\n=== LittleFS Files ===");
  
  File root = LittleFS.open("/");
  if (!root) {
    Serial.println("Failed to open root directory");
    return;
  }
  
  File file = root.openNextFile();
  while (file) {
    Serial.print("  ");
    Serial.print(file.name());
    Serial.print(" - ");
    Serial.print(file.size());
    Serial.println(" bytes");
    file = root.openNextFile();
  }
  
  Serial.println("======================\n");
}

void dumpWavFile() {
  File f = LittleFS.open(WAV_FILE_PATH, FILE_READ);
  if (!f) {
    Serial.println("Failed to open WAV file");
    return;
  }
  
  // Dump raw bytes directly to serial
  uint8_t buffer[512];
  while (f.available()) {
    size_t bytesRead = f.read(buffer, sizeof(buffer));
    Serial.write(buffer, bytesRead);
  }
  
  f.close();
}

void dumpMfccFile() {
  File f = LittleFS.open(MFCC_FILE_PATH, FILE_READ);
  if (!f) {
    Serial.println("Failed to open MFCC file");
    return;
  }
  
  // Read and print header
  uint32_t numCoeffs, frameSizeMs, frameStrideMs, sampleRate;
  f.read((uint8_t*)&numCoeffs, sizeof(uint32_t));
  f.read((uint8_t*)&frameSizeMs, sizeof(uint32_t));
  f.read((uint8_t*)&frameStrideMs, sizeof(uint32_t));
  f.read((uint8_t*)&sampleRate, sizeof(uint32_t));
  
  Serial.println("\n=== MFCC FILE DUMP ===");
  Serial.print("Num Coefficients: "); Serial.println(numCoeffs);
  Serial.print("Frame Size: "); Serial.print(N_FFT); Serial.print(" samples (");
  Serial.print(frameSizeMs); Serial.println(" ms)");
  Serial.print("Hop Length: "); Serial.print(HOP_LENGTH); Serial.print(" samples (");
  Serial.print(frameStrideMs); Serial.println(" ms)");
  Serial.print("Sample Rate: "); Serial.println(sampleRate);
  Serial.println("\nLibrosa-compatible parameters:");
  Serial.print("  n_mfcc: "); Serial.println(NUM_MFCC);
  Serial.print("  n_fft: "); Serial.println(N_FFT);
  Serial.print("  FFT size: "); Serial.println(FFT_SIZE);
  Serial.print("  hop_length: "); Serial.println(HOP_LENGTH);
  Serial.print("  n_mels: "); Serial.println(NUM_MEL_FILTERS);
  Serial.println("\nMFCC Coefficients (frame by frame):");
  
  // Accumulator to calculate mean while reading
  MFCCAccumulator accumulator;
  accumulator.reset();
  
  int frameNum = 0;
  float coeffs[NUM_MFCC];
  
  while (f.available() >= NUM_MFCC * sizeof(float)) {
    f.read((uint8_t*)coeffs, NUM_MFCC * sizeof(float));
    
    // Add to accumulator for mean calculation
    accumulator.addFrame(coeffs);
    
    Serial.print("Frame "); Serial.print(frameNum); Serial.print(": ");
    for (int i = 0; i < NUM_MFCC; i++) {
      Serial.print(coeffs[i], 4);
      if (i < NUM_MFCC - 1) Serial.print(", ");
    }
    Serial.println();
    frameNum++;
  }
  
  f.close();
  
  // Calculate and display mean MFCC
  if (frameNum > 0) {
    float meanMFCC[NUM_MFCC];
    accumulator.getMean(meanMFCC);
    
    Serial.println("\n--- AVERAGED MFCC ---");
    Serial.print("Mean over ");
    Serial.print(frameNum);
    Serial.println(" frames:");
    Serial.print("  [");
    for (int i = 0; i < NUM_MFCC; i++) {
      Serial.print(meanMFCC[i], 4);
      if (i < NUM_MFCC - 1) Serial.print(", ");
    }
    Serial.println("]");
    Serial.println("---------------------");
  }
  
  Serial.println("\n=== END DUMP ===\n");
}

// ==================== Main Setup and Loop ====================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("ESP32-S3 INMP441 Recorder with MFCC and Classifier");
  Serial.println("===================================================");
  Serial.println("Librosa-compatible MFCC extraction:");
  Serial.print("  n_mfcc="); Serial.println(NUM_MFCC);
  Serial.print("  n_fft="); Serial.println(N_FFT);
  Serial.print("  FFT size="); Serial.println(FFT_SIZE);
  Serial.print("  hop_length="); Serial.println(HOP_LENGTH);
  Serial.print("  n_mels="); Serial.println(NUM_MEL_FILTERS);
  Serial.println("Classification method: Mean of all MFCC frames");
  
  // Check PSRAM
  if (!psramFound()) {
    Serial.println("ERROR: PSRAM not found!");
    Serial.println("Please enable PSRAM in Tools > PSRAM");
    while(1) delay(1000);
  }
  
  Serial.print("PSRAM size: ");
  Serial.print(ESP.getPsramSize() / 1024);
  Serial.println(" KB");
  
  // Allocate buffers
  if (!allocateBuffers()) {
    Serial.println("ERROR: Failed to allocate buffers!");
    while(1) delay(1000);
  }
  
  Serial.print("Frame size (n_fft): "); Serial.print(N_FFT); 
  Serial.print(" samples (~"); Serial.print(FRAME_SIZE_MS); Serial.println(" ms)");
  Serial.print("Hop length: "); Serial.print(HOP_LENGTH); 
  Serial.print(" samples ("); Serial.print(FRAME_STRIDE_MS); Serial.println(" ms)");
  Serial.print("Overlap: "); Serial.print(N_FFT - HOP_LENGTH); Serial.println(" samples");
  
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  if (!LittleFS.begin(true)) {
    Serial.println("LittleFS Mount Failed");
    return;
  }
  Serial.println("LittleFS mounted successfully");

  listFiles();
  
  esp_err_t ret = dsps_fft2r_init_fc32(NULL, FFT_SIZE);
  if (ret != ESP_OK) {
    Serial.println("Failed to initialize ESP-DSP FFT");
    return;
  }
  Serial.println("ESP-DSP FFT initialized (4096 point)");
  
  // Generate Hann window for the full N_FFT size
  dsps_wind_hann_f32(window, N_FFT);
  
  initMelFilterbank();
  Serial.println("Mel filterbank initialized");
  
  // Initialize MFCC accumulator
  mfccAccumulator.reset();
  
  loadClassifierModel();
  
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = BITS_PER_SAMPLE,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  
  if (i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL) != ESP_OK) {
    Serial.println("Failed to install I2S driver");
    return;
  }
  
  if (i2s_set_pin(I2S_PORT, &pin_config) != ESP_OK) {
    Serial.println("Failed to set I2S pins");
    return;
  }
  
  Serial.println("I2S initialized successfully");
  Serial.println("\n=== Commands ===");
  Serial.println("  Button on GPIO16 - start/stop recording");
  Serial.println("  'dump'     - dump WAV file to serial");
  Serial.println("  'mfcc'     - dump MFCC features + averaged MFCC");
  Serial.println("  'classify' - classify using mean MFCC");
  Serial.println("  'model'    - print classifier model");
  Serial.println("================\n");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "dump") {
      dumpWavFile();
    } else if (command == "mfcc") {
      dumpMfccFile();
    } else if (command == "classify") {
      classifyStoredMFCC();
    } else if (command == "model") {
      if (classifier.isLoaded) {
        printClassifierModel();
      } else {
        Serial.println("Classifier model not loaded!");
      }
    }
  }
  
  bool currentButtonState = digitalRead(BUTTON_PIN);
  
  if (currentButtonState == LOW && lastButtonState == HIGH) {
    toggleRecording();
    delay(50);
  }
  
  lastButtonState = currentButtonState;
  
  if (isRecording) {
    recordAudio();
  }
  
  delay(10);
}

void toggleRecording() {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
}

void startRecording() {
  Serial.println("Starting recording...");
  
  if (LittleFS.exists(WAV_FILE_PATH)) LittleFS.remove(WAV_FILE_PATH);
  if (LittleFS.exists(MFCC_FILE_PATH)) LittleFS.remove(MFCC_FILE_PATH);
  
  wavFile = LittleFS.open(WAV_FILE_PATH, FILE_WRITE);
  if (!wavFile) {
    Serial.println("Failed to open WAV file");
    return;
  }
  
  mfccFile = LittleFS.open(MFCC_FILE_PATH, FILE_WRITE);
  if (!mfccFile) {
    Serial.println("Failed to open MFCC file");
    wavFile.close();
    return;
  }
  
  WAVHeader header;
  wavFile.write((uint8_t*)&header, WAV_HEADER_SIZE);
  
  uint32_t numMfcc = NUM_MFCC;
  uint32_t frameSizeMs = FRAME_SIZE_MS;
  uint32_t frameStrideMs = FRAME_STRIDE_MS;
  uint32_t sampleRate = SAMPLE_RATE;
  
  mfccFile.write((uint8_t*)&numMfcc, sizeof(uint32_t));
  mfccFile.write((uint8_t*)&frameSizeMs, sizeof(uint32_t));
  mfccFile.write((uint8_t*)&frameStrideMs, sizeof(uint32_t));
  mfccFile.write((uint8_t*)&sampleRate, sizeof(uint32_t));
  
  recordedSamples = 0;
  mfccFrameCount = 0;
  bufferIndex = 0;
  memset(audioBuffer, 0, AUDIO_BUFFER_SIZE * sizeof(int16_t));
  
  // Reset MFCC accumulator for this recording
  mfccAccumulator.reset();
  
  isRecording = true;
  Serial.println("Recording started");
}

void stopRecording() {
  Serial.println("Stopping recording...");
  isRecording = false;
  
  if (!wavFile) {
    Serial.println("No active recording");
    return;
  }
  
  updateWavHeader();
  wavFile.close();
  mfccFile.close();
  
  Serial.print("Recording stopped. Samples recorded: ");
  Serial.println(recordedSamples);
  Serial.print("MFCC frames computed: ");
  Serial.println(mfccFrameCount);
  Serial.print("Duration: ");
  Serial.print(recordedSamples / (float)SAMPLE_RATE, 2);
  Serial.println(" seconds");
  
  // Print file sizes
  File f = LittleFS.open(WAV_FILE_PATH, FILE_READ);
  if (f) {
    Serial.print("WAV file size: ");
    Serial.print(f.size());
    Serial.println(" bytes");
    f.close();
  }
  
  f = LittleFS.open(MFCC_FILE_PATH, FILE_READ);
  if (f) {
    Serial.print("MFCC file size: ");
    Serial.print(f.size());
    Serial.println(" bytes");
    f.close();
  }
  
  // Auto-classify if classifier is loaded
  if (classifier.isLoaded && mfccAccumulator.frameCount > 0) {
    Serial.println("\n--- Auto-classifying recording ---");
    
    float meanMFCC[NUM_MFCC];
    mfccAccumulator.getMean(meanMFCC);
    
    Serial.print("Computed mean over ");
    Serial.print(mfccAccumulator.frameCount);
    Serial.println(" frames");
    
    float probabilities[NUM_CLASSES];
    int predictedClass = predictClass(meanMFCC, probabilities);
    
    Serial.print("\nPredicted: ");
    Serial.print(classifier.classNames[predictedClass]);
    Serial.print(" (");
    Serial.print(probabilities[predictedClass] * 100, 1);
    Serial.println("%)");
    
    Serial.println("All probabilities:");
    for (int c = 0; c < NUM_CLASSES; c++) {
      Serial.print("  ");
      Serial.print(classifier.classNames[c]);
      Serial.print(": ");
      Serial.print(probabilities[c] * 100, 1);
      Serial.println("%");
    }
  }
  
  Serial.println("\nType 'classify' for detailed classification");
  Serial.println("Type 'mfcc' to see all frames + averaged MFCC");
}

void recordAudio() {
  int32_t i2s_buffer[I2S_READ_LEN];
  size_t bytes_read = 0;
  
  esp_err_t result = i2s_read(I2S_PORT, i2s_buffer, I2S_READ_LEN * sizeof(int32_t), &bytes_read, portMAX_DELAY);
  
  if (result == ESP_OK && bytes_read > 0) {
    int samples_read = bytes_read / sizeof(int32_t);
    
    for (int i = 0; i < samples_read; i++) {
      int16_t sample = (i2s_buffer[i] >> 14) & 0xFFFF;
      wavFile.write((uint8_t*)&sample, sizeof(int16_t));
      recordedSamples++;
      
      audioBuffer[bufferIndex] = sample;
      bufferIndex++;
      
      // Extract an MFCC frame every HOP_LENGTH samples once the buffer holds N_FFT samples
      if (bufferIndex >= N_FFT) {
        extractMFCC(audioBuffer);
        
        // Slide the buffer forward by HOP_LENGTH, keeping the overlap region
        memmove(audioBuffer, audioBuffer + HOP_LENGTH, 
                (N_FFT - HOP_LENGTH) * sizeof(int16_t));
        bufferIndex = N_FFT - HOP_LENGTH;
      }
    }
  }
}

void updateWavHeader() {
  WAVHeader header;
  
  uint32_t dataSize = recordedSamples * sizeof(int16_t);
  header.dataSize = dataSize;
  header.fileSize = dataSize + WAV_HEADER_SIZE - 8;
  header.byteRate = SAMPLE_RATE * sizeof(int16_t);
  header.blockAlign = sizeof(int16_t);
  
  wavFile.seek(0);
  wavFile.write((uint8_t*)&header, WAV_HEADER_SIZE);
}