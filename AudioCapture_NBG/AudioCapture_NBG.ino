#include <driver/i2s.h>
#include <LittleFS.h>
#include <dsps_fft2r.h>
#include <dsps_wind.h>
#include <math.h>

// I2S Microphone pins (INMP441)
#define I2S_SD 41
#define I2S_WS 7
#define I2S_SCK 15
#define BUTTON_PIN 16
#include <driver/i2s.h>
#include <LittleFS.h>
#include <dsps_fft2r.h>
#include <dsps_wind.h>
#include <math.h>

// I2S Microphone pins (INMP441)
#define I2S_SD 41
#define I2S_WS 7
#define I2S_SCK 15
#define BUTTON_PIN 16

// I2S Configuration
#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define BITS_PER_SAMPLE I2S_BITS_PER_SAMPLE_32BIT
#define I2S_READ_LEN (1024)

// MFCC Configuration
#define N_FFT 4096
#define HOP_LENGTH 512
#define NUM_MFCC 13
#define NUM_MEL_FILTERS 128
#define MEL_LOW_FREQ 0
#define MEL_HIGH_FREQ (SAMPLE_RATE / 2)

#define FRAME_SIZE_MS ((N_FFT * 1000) / SAMPLE_RATE)
#define FRAME_STRIDE_MS ((HOP_LENGTH * 1000) / SAMPLE_RATE)

#define NUM_CLASSES 5
#define ENABLE_REALTIME_CLASSIFICATION false

#define WAV_FILE_PATH "/recording.wav"
#define MFCC_FILE_PATH "/mfcc_features.bin"
#define CLASSIFIER_MODEL_PATH "/classifier_model.bin"
#define WAV_HEADER_SIZE 44

// Recording state
bool isRecording = false;
bool lastButtonState = HIGH;
File wavFile;
uint32_t recordedSamples = 0;

// FFT buffers - PSRAM
float* fft_input = nullptr;
float* fft_output = nullptr;
float* window = nullptr;

// MFCC processing buffers - PSRAM
float* windowedFrame = nullptr;
float* powerSpectrum = nullptr;
float* melEnergies = nullptr;

// Frame buffer - PSRAM (THIS WAS THE PROBLEM!)
int16_t* frame = nullptr;
int16_t* nextSamples = nullptr;

// Mel filterbank
float** melFilterbank = nullptr;
float mfcc_coeffs[NUM_MFCC];

// MFCC accumulator
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

// Classifier
struct GaussianClassifier {
  char classNames[NUM_CLASSES][32];
  float means[NUM_CLASSES][NUM_MFCC];
  float variances[NUM_CLASSES][NUM_MFCC];
  float priors[NUM_CLASSES];
  bool isLoaded;
};

GaussianClassifier classifier;

// WAV header
struct WAVHeader {
  char riff[4] = {'R', 'I', 'F', 'F'};
  uint32_t fileSize;
  char wave[4] = {'W', 'A', 'V', 'E'};
  
  char fmt[4] = {'f', 'm', 't', ' '};
  uint32_t fmtSize = 16;
  uint16_t audioFormat = 1;
  uint16_t numChannels = 1;
  uint32_t sampleRate = SAMPLE_RATE;
  uint32_t byteRate;
  uint16_t blockAlign;
  uint16_t bitsPerSample = 16;
  
  char data[4] = {'d', 'a', 't', 'a'};
  uint32_t dataSize;
};

// ==================== Memory Allocation ====================

bool allocateBuffers() {
  Serial.println("Allocating buffers in PSRAM...");
  
  // FFT buffers
  fft_input = (float*)ps_malloc(N_FFT * 2 * sizeof(float));
  if (!fft_input) {
    Serial.println("Failed to allocate fft_input");
    return false;
  }
  Serial.println("  ✓ FFT input allocated");
  
  fft_output = (float*)ps_malloc((N_FFT / 2 + 1) * sizeof(float));
  if (!fft_output) {
    Serial.println("Failed to allocate fft_output");
    return false;
  }
  Serial.println("  ✓ FFT output allocated");
  
  window = (float*)ps_malloc(N_FFT * sizeof(float));
  if (!window) {
    Serial.println("Failed to allocate window");
    return false;
  }
  Serial.println("  ✓ Window allocated");
  
  // MFCC processing buffers
  windowedFrame = (float*)ps_malloc(N_FFT * sizeof(float));
  if (!windowedFrame) {
    Serial.println("Failed to allocate windowedFrame");
    return false;
  }
  Serial.println("  ✓ Windowed frame allocated");
  
  powerSpectrum = (float*)ps_malloc((N_FFT / 2 + 1) * sizeof(float));
  if (!powerSpectrum) {
    Serial.println("Failed to allocate powerSpectrum");
    return false;
  }
  Serial.println("  ✓ Power spectrum allocated");
  
  melEnergies = (float*)ps_malloc(NUM_MEL_FILTERS * sizeof(float));
  if (!melEnergies) {
    Serial.println("Failed to allocate melEnergies");
    return false;
  }
  Serial.println("  ✓ Mel energies allocated");
  
  // Frame buffers - THIS IS THE KEY FIX!
  frame = (int16_t*)ps_malloc(N_FFT * sizeof(int16_t));
  if (!frame) {
    Serial.println("Failed to allocate frame buffer");
    return false;
  }
  Serial.println("  ✓ Frame buffer allocated");
  
  nextSamples = (int16_t*)ps_malloc(HOP_LENGTH * sizeof(int16_t));
  if (!nextSamples) {
    Serial.println("Failed to allocate nextSamples buffer");
    return false;
  }
  Serial.println("  ✓ Next samples buffer allocated");
  
  // Mel filterbank
  int numBins = N_FFT / 2 + 1;
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
  Serial.println("  ✓ Mel filterbank allocated");
  
  Serial.println("\nAll buffers allocated successfully!");
  Serial.print("Free PSRAM: ");
  Serial.print(ESP.getFreePsram());
  Serial.println(" bytes\n");
  
  return true;
}

void freeBuffers() {
  if (fft_input) free(fft_input);
  if (fft_output) free(fft_output);
  if (window) free(window);
  if (windowedFrame) free(windowedFrame);
  if (powerSpectrum) free(powerSpectrum);
  if (melEnergies) free(melEnergies);
  if (frame) free(frame);
  if (nextSamples) free(nextSamples);
  
  if (melFilterbank) {
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
      if (melFilterbank[i]) free(melFilterbank[i]);
    }
    free(melFilterbank);
  }
}

// ==================== Classifier Functions ====================

bool loadClassifierModel() {
  File modelFile = LittleFS.open(CLASSIFIER_MODEL_PATH, FILE_READ);
  if (!modelFile) {
    Serial.println("Classifier model file not found.");
    classifier.isLoaded = false;
    return false;
  }
  
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
  
  for (int i = 0; i < NUM_CLASSES; i++) {
    modelFile.read((uint8_t*)classifier.classNames[i], 32);
  }
  
  for (int i = 0; i < NUM_CLASSES; i++) {
    modelFile.read((uint8_t*)classifier.means[i], NUM_MFCC * sizeof(float));
  }
  
  for (int i = 0; i < NUM_CLASSES; i++) {
    modelFile.read((uint8_t*)classifier.variances[i], NUM_MFCC * sizeof(float));
  }
  
  modelFile.read((uint8_t*)classifier.priors, NUM_CLASSES * sizeof(float));
  
  modelFile.close();
  classifier.isLoaded = true;
  
  Serial.println("Classifier model loaded successfully!");
  printClassifierModel();
  return true;
}

void printClassifierModel() {
  Serial.println("\n=== Classifier Model ===");
  Serial.print("Number of classes: "); Serial.println(NUM_CLASSES);
  
  for (int c = 0; c < NUM_CLASSES; c++) {
    Serial.print("\nClass "); Serial.print(c); Serial.print(": ");
    Serial.println(classifier.classNames[c]);
    Serial.print("  Prior: "); Serial.println(classifier.priors[c], 4);
    
    Serial.print("  Means: [");
    for (int i = 0; i < min(5, NUM_MFCC); i++) {
      Serial.print(classifier.means[c][i], 3);
      if (i < 4) Serial.print(", ");
    }
    Serial.println("...]");
  }
  Serial.println("========================\n");
}

float gaussianPDF(float x, float mean, float variance) {
  if (variance <= 0) variance = 1e-6;
  float exponent = -0.5 * pow((x - mean), 2) / variance;
  float coefficient = 1.0 / sqrt(2.0 * M_PI * variance);
  return coefficient * exp(exponent);
}

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

// ==================== MFCC Processing ====================

float hzToMel(float hz) {
  return 2595.0 * log10(1.0 + hz / 700.0);
}

float melToHz(float mel) {
  return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

void initMelFilterbank() {
  float melLow = hzToMel(MEL_LOW_FREQ);
  float melHigh = hzToMel(MEL_HIGH_FREQ);
  
  float melPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    melPoints[i] = melLow + (melHigh - melLow) * i / (NUM_MEL_FILTERS + 1);
  }
  
  float hzPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    hzPoints[i] = melToHz(melPoints[i]);
  }
  
  int binPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    binPoints[i] = (int)floor((N_FFT + 1) * hzPoints[i] / SAMPLE_RATE);
  }
  
  int numBins = N_FFT / 2 + 1;
  
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    int leftBin = binPoints[m];
    int centerBin = binPoints[m + 1];
    int rightBin = binPoints[m + 2];
    
    for (int k = 0; k < numBins; k++) {
      melFilterbank[m][k] = 0.0;
    }
    
    for (int k = leftBin; k < centerBin && k < numBins; k++) {
      if (centerBin != leftBin) {
        melFilterbank[m][k] = (float)(k - leftBin) / (centerBin - leftBin);
      }
    }
    
    for (int k = centerBin; k < rightBin && k < numBins; k++) {
      if (rightBin != centerBin) {
        melFilterbank[m][k] = (float)(rightBin - k) / (rightBin - centerBin);
      }
    }
  }
  
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    float enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m]);
    for (int k = 0; k < numBins; k++) {
      melFilterbank[m][k] *= enorm;
    }
  }
}

void applyHannWindow(int16_t* input, float* output, int length) {
  for (int i = 0; i < length; i++) {
    float normalized = input[i] / 32768.0f;
    output[i] = normalized * window[i];
  }
}

void computePowerSpectrum(float* input, float* output, int length) {
  for (int i = 0; i < length; i++) {
    fft_input[i * 2] = input[i];
    fft_input[i * 2 + 1] = 0.0;
  }
  
  for (int i = length; i < N_FFT; i++) {
    fft_input[i * 2] = 0.0;
    fft_input[i * 2 + 1] = 0.0;
  }
  
  dsps_fft2r_fc32(fft_input, N_FFT);
  dsps_bit_rev_fc32(fft_input, N_FFT);
  
  int numBins = N_FFT / 2 + 1;
  for (int i = 0; i < numBins; i++) {
    float real = fft_input[i * 2];
    float imag = fft_input[i * 2 + 1];
    output[i] = (real * real + imag * imag) / (float)N_FFT;
  }
}

void applyMelFilters(float* powerSpectrumIn, float* melEnergiesOut) {
  int numBins = N_FFT / 2 + 1;
  
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    double energy = 0.0;
    for (int k = 0; k < numBins; k++) {
      energy += (double)powerSpectrumIn[k] * (double)melFilterbank[m][k];
    }
    melEnergiesOut[m] = log(energy + 1e-10);
  }
}

void computeDCT(float* melEnergiesIn, float* mfccOut, int numFilters, int numCoeffs) {
  for (int i = 0; i < numCoeffs; i++) {
    float sum = 0.0;
    for (int j = 0; j < numFilters; j++) {
      sum += melEnergiesIn[j] * cos(M_PI * i * (j + 0.5) / numFilters);
    }
    
    if (i == 0) {
      mfccOut[i] = sum * sqrt(1.0 / numFilters);
    } else {
      mfccOut[i] = sum * sqrt(2.0 / numFilters);
    }
  }
}

void extractMFCC(int16_t* frameIn) {
  applyHannWindow(frameIn, windowedFrame, N_FFT);
  computePowerSpectrum(windowedFrame, powerSpectrum, N_FFT);
  applyMelFilters(powerSpectrum, melEnergies);
  computeDCT(melEnergies, mfcc_coeffs, NUM_MEL_FILTERS, NUM_MFCC);
}

// ==================== WAV File Post-Processing ====================

void processWavFileToMFCC() {
  File wavFileInput = LittleFS.open(WAV_FILE_PATH, FILE_READ);
  if (!wavFileInput) {
    Serial.println("Failed to open WAV file for MFCC processing");
    return;
  }
  
  wavFileInput.seek(WAV_HEADER_SIZE);
  
  if (LittleFS.exists(MFCC_FILE_PATH)) LittleFS.remove(MFCC_FILE_PATH);
  File mfccFileOutput = LittleFS.open(MFCC_FILE_PATH, FILE_WRITE);
  if (!mfccFileOutput) {
    Serial.println("Failed to create MFCC file");
    wavFileInput.close();
    return;
  }
  
  uint32_t numMfcc = NUM_MFCC;
  uint32_t frameSizeMs = FRAME_SIZE_MS;
  uint32_t frameStrideMs = FRAME_STRIDE_MS;
  uint32_t sampleRate = SAMPLE_RATE;
  
  mfccFileOutput.write((uint8_t*)&numMfcc, sizeof(uint32_t));
  mfccFileOutput.write((uint8_t*)&frameSizeMs, sizeof(uint32_t));
  mfccFileOutput.write((uint8_t*)&frameStrideMs, sizeof(uint32_t));
  mfccFileOutput.write((uint8_t*)&sampleRate, sizeof(uint32_t));
  
  uint32_t frameCount = 0;
  
  Serial.println("\n=== Processing WAV file to MFCC ===");
  Serial.print("Frame size: "); Serial.print(N_FFT); Serial.println(" samples");
  Serial.print("Hop length: "); Serial.print(HOP_LENGTH); Serial.println(" samples");
  Serial.println();
  
  // Read first full frame
  size_t bytesRead = wavFileInput.read((uint8_t*)frame, N_FFT * sizeof(int16_t));
  uint32_t totalSamples = bytesRead / sizeof(int16_t);
  
  if (totalSamples < N_FFT) {
    Serial.println("WAV file too short for MFCC extraction");
    wavFileInput.close();
    mfccFileOutput.close();
    return;
  }
  
  mfccAccumulator.reset();
  
  Serial.println("Extracting MFCC from first frame...");
  extractMFCC(frame);
  mfccFileOutput.write((uint8_t*)mfcc_coeffs, NUM_MFCC * sizeof(float));
  mfccAccumulator.addFrame(mfcc_coeffs);
  frameCount++;
  Serial.println("Frame 1 processed");
  
  // Process remaining frames - NOW USING PSRAM BUFFERS!
  int frameProgress = 0;
  
  while (true) {
    bytesRead = wavFileInput.read((uint8_t*)nextSamples, HOP_LENGTH * sizeof(int16_t));
    if (bytesRead < HOP_LENGTH * sizeof(int16_t)) {
      break;
    }
    
    // These now work with PSRAM buffers - no stack overflow!
    memmove(frame, frame + HOP_LENGTH, (N_FFT - HOP_LENGTH) * sizeof(int16_t));
    memcpy(frame + (N_FFT - HOP_LENGTH), nextSamples, HOP_LENGTH * sizeof(int16_t));
    
    extractMFCC(frame);
    mfccFileOutput.write((uint8_t*)mfcc_coeffs, NUM_MFCC * sizeof(float));
    mfccAccumulator.addFrame(mfcc_coeffs);
    frameCount++;
    
    frameProgress++;
    if (frameProgress % 10 == 0) {
      Serial.print("Frame ");
      Serial.print(frameCount);
      Serial.println(" processed");
      delay(5);
    }
  }
  
  wavFileInput.close();
  mfccFileOutput.close();
  
  Serial.print("\nTotal frames: ");
  Serial.println(frameCount);
  Serial.print("Duration: ");
  Serial.print((frameCount * HOP_LENGTH) / (float)SAMPLE_RATE, 2);
  Serial.println(" seconds");
  Serial.println("=== Processing complete ===\n");
}

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
  
  MFCCAccumulator accumulator;
  accumulator.reset();
  
  float coeffs[NUM_MFCC];
  int frameNum = 0;
  
  Serial.println("Reading all frames...");
  while (f.available() >= NUM_MFCC * sizeof(float)) {
    f.read((uint8_t*)coeffs, NUM_MFCC * sizeof(float));
    accumulator.addFrame(coeffs);
    frameNum++;
  }
  
  f.close();
  
  if (frameNum == 0) {
    Serial.println("ERROR: No MFCC frames found!");
    return;
  }
  
  float meanMFCC[NUM_MFCC];
  accumulator.getMean(meanMFCC);
  
  Serial.print("Total frames: ");
  Serial.println(frameNum);
  Serial.println("\nMean MFCC: [");
  for (int i = 0; i < NUM_MFCC; i++) {
    Serial.print(meanMFCC[i], 4);
    if (i < NUM_MFCC - 1) Serial.print(", ");
  }
  Serial.println("]");
  
  float probabilities[NUM_CLASSES];
  int predictedClass = predictClass(meanMFCC, probabilities);
  
  Serial.println("\n=== Classification Result ===");
  Serial.print("Predicted: ");
  Serial.println(classifier.classNames[predictedClass]);
  Serial.print("Confidence: ");
  Serial.print(probabilities[predictedClass] * 100, 2);
  Serial.println("%\n");
  
  Serial.println("All probabilities:");
  for (int c = 0; c < NUM_CLASSES; c++) {
    Serial.print("  ");
    Serial.print(classifier.classNames[c]);
    Serial.print(": ");
    Serial.print(probabilities[c] * 100, 2);
    Serial.println("%");
  }
  Serial.println("==============================\n");
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
  
  // Serial.println("\n=== WAV FILE DUMP ===");
  WAVHeader header;
  f.read((uint8_t*)&header, WAV_HEADER_SIZE);
  f.seek(0);
  uint8_t buffer[512];
  while (f.available()) {
    size_t bytesRead = f.read(buffer, sizeof(buffer));
    Serial.write(buffer, bytesRead);
  }
  
  // Serial.println("\n\n=== END DUMP ===\n");
  f.close();
}

void dumpMfccFile() {
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
  
  Serial.println("\n=== MFCC FILE DUMP ===");
  Serial.print("Num Coefficients: "); Serial.println(numCoeffs);
  Serial.print("Frame Size: "); Serial.print(frameSizeMs); Serial.println(" ms");
  Serial.print("Hop Length: "); Serial.print(frameStrideMs); Serial.println(" ms");
  Serial.println("\nLibrosa parameters:");
  Serial.print("  n_mfcc="); Serial.println(NUM_MFCC);
  Serial.print("  n_fft="); Serial.println(N_FFT);
  Serial.print("  hop_length="); Serial.println(HOP_LENGTH);
  Serial.println("\nFrames:");
  
  MFCCAccumulator accumulator;
  accumulator.reset();
  int frameNum = 0;
  float coeffs[NUM_MFCC];
  
  while (f.available() >= NUM_MFCC * sizeof(float)) {
    f.read((uint8_t*)coeffs, NUM_MFCC * sizeof(float));
    accumulator.addFrame(coeffs);
    
    Serial.print("  Frame ");
    Serial.print(frameNum);
    Serial.print(": [");
    for (int i = 0; i < NUM_MFCC; i++) {
      Serial.print(coeffs[i], 3);
      if (i < NUM_MFCC - 1) Serial.print(", ");
    }
    Serial.println("]");
    frameNum++;
  }
  
  f.close();
  
  if (frameNum > 0) {
    float meanMFCC[NUM_MFCC];
    accumulator.getMean(meanMFCC);
    
    Serial.print("\nMean MFCC over ");
    Serial.print(frameNum);
    Serial.println(" frames:");
    Serial.print("  [");
    for (int i = 0; i < NUM_MFCC; i++) {
      Serial.print(meanMFCC[i], 4);
      if (i < NUM_MFCC - 1) Serial.print(", ");
    }
    Serial.println("]");
  }
  
  Serial.println("\n=== END DUMP ===\n");
}

// ==================== Main ====================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\nESP32-S3 Librosa-Compatible MFCC Processor");
  Serial.println("==========================================");
  Serial.println("Parameters:");
  Serial.print("  N_FFT: "); Serial.println(N_FFT);
  Serial.print("  HOP_LENGTH: "); Serial.println(HOP_LENGTH);
  Serial.print("  NUM_MFCC: "); Serial.println(NUM_MFCC);
  Serial.print("  SAMPLE_RATE: "); Serial.println(SAMPLE_RATE);
  Serial.println();
  
  if (!psramFound()) {
    Serial.println("ERROR: PSRAM not found!");
    while(1) delay(1000);
  }
  
  Serial.print("PSRAM available: ");
  Serial.print(ESP.getPsramSize() / 1024);
  Serial.println(" KB");
  
  if (!allocateBuffers()) {
    Serial.println("FATAL: Buffer allocation failed!");
    while(1) delay(1000);
  }
  
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  if (!LittleFS.begin(true)) {
    Serial.println("LittleFS mount failed");
    return;
  }
  Serial.println("LittleFS mounted\n");
  
  esp_err_t ret = dsps_fft2r_init_fc32(NULL, N_FFT);
  if (ret != ESP_OK) {
    Serial.println("FFT init failed");
    return;
  }
  
  dsps_wind_hann_f32(window, N_FFT);
  initMelFilterbank();
  mfccAccumulator.reset();
  
  Serial.println("Initialization complete!\n");
  listFiles();
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
    Serial.println("I2S install failed");
    return;
  }
  
  if (i2s_set_pin(I2S_PORT, &pin_config) != ESP_OK) {
    Serial.println("I2S set pin failed");
    return;
  }
  
  Serial.println("I2S configured\n");
  Serial.println("=== Commands ===");
  Serial.println("  Button GPIO16 - record");
  Serial.println("  'process'     - extract MFCC from WAV");
  Serial.println("  'classify'    - classify MFCC");
  Serial.println("  'dump'        - dump WAV file");
  Serial.println("  'mfcc'        - dump MFCC file");
  Serial.println("  'files'       - list files");
  Serial.println("================\n");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "process") {
      processWavFileToMFCC();
    } else if (command == "classify") {
      classifyStoredMFCC();
    } else if (command == "dump") {
      dumpWavFile();
    } else if (command == "mfcc") {
      dumpMfccFile();
    } else if (command == "files") {
      listFiles();
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
  Serial.println("Recording...");
  if (LittleFS.exists(WAV_FILE_PATH)) LittleFS.remove(WAV_FILE_PATH);
  
  wavFile = LittleFS.open(WAV_FILE_PATH, FILE_WRITE);
  if (!wavFile) {
    Serial.println("Failed to open WAV file");
    return;
  }
  
  WAVHeader header;
  wavFile.write((uint8_t*)&header, WAV_HEADER_SIZE);
  
  recordedSamples = 0;
  isRecording = true;
}

void stopRecording() {
  isRecording = false;
  updateWavHeader();
  wavFile.close();
  
  Serial.print("Stopped. Samples: ");
  Serial.print(recordedSamples);
  Serial.print(" (");
  Serial.print(recordedSamples / (float)SAMPLE_RATE, 2);
  Serial.println("s)");
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
// I2S Configuration
#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define BITS_PER_SAMPLE I2S_BITS_PER_SAMPLE_32BIT
#define I2S_READ_LEN (1024)

// MFCC Configuration
#define N_FFT 4096
#define HOP_LENGTH 512
#define NUM_MFCC 13
#define NUM_MEL_FILTERS 128
#define MEL_LOW_FREQ 0
#define MEL_HIGH_FREQ (SAMPLE_RATE / 2)

#define FRAME_SIZE_MS ((N_FFT * 1000) / SAMPLE_RATE)
#define FRAME_STRIDE_MS ((HOP_LENGTH * 1000) / SAMPLE_RATE)

#define NUM_CLASSES 5
#define ENABLE_REALTIME_CLASSIFICATION false

#define WAV_FILE_PATH "/recording.wav"
#define MFCC_FILE_PATH "/mfcc_features.bin"
#define CLASSIFIER_MODEL_PATH "/classifier_model.bin"
#define WAV_HEADER_SIZE 44

// Recording state
bool isRecording = false;
bool lastButtonState = HIGH;
File wavFile;
uint32_t recordedSamples = 0;

// FFT buffers - PSRAM
float* fft_input = nullptr;
float* fft_output = nullptr;
float* window = nullptr;

// MFCC processing buffers - PSRAM
float* windowedFrame = nullptr;
float* powerSpectrum = nullptr;
float* melEnergies = nullptr;

// Frame buffer - PSRAM (THIS WAS THE PROBLEM!)
int16_t* frame = nullptr;
int16_t* nextSamples = nullptr;

// Mel filterbank
float** melFilterbank = nullptr;
float mfcc_coeffs[NUM_MFCC];

// MFCC accumulator
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

// Classifier
struct GaussianClassifier {
  char classNames[NUM_CLASSES][32];
  float means[NUM_CLASSES][NUM_MFCC];
  float variances[NUM_CLASSES][NUM_MFCC];
  float priors[NUM_CLASSES];
  bool isLoaded;
};

GaussianClassifier classifier;

// WAV header
struct WAVHeader {
  char riff[4] = {'R', 'I', 'F', 'F'};
  uint32_t fileSize;
  char wave[4] = {'W', 'A', 'V', 'E'};
  
  char fmt[4] = {'f', 'm', 't', ' '};
  uint32_t fmtSize = 16;
  uint16_t audioFormat = 1;
  uint16_t numChannels = 1;
  uint32_t sampleRate = SAMPLE_RATE;
  uint32_t byteRate;
  uint16_t blockAlign;
  uint16_t bitsPerSample = 16;
  
  char data[4] = {'d', 'a', 't', 'a'};
  uint32_t dataSize;
};

// ==================== Memory Allocation ====================

bool allocateBuffers() {
  Serial.println("Allocating buffers in PSRAM...");
  
  // FFT buffers
  fft_input = (float*)ps_malloc(N_FFT * 2 * sizeof(float));
  if (!fft_input) {
    Serial.println("Failed to allocate fft_input");
    return false;
  }
  Serial.println("  ✓ FFT input allocated");
  
  fft_output = (float*)ps_malloc((N_FFT / 2 + 1) * sizeof(float));
  if (!fft_output) {
    Serial.println("Failed to allocate fft_output");
    return false;
  }
  Serial.println("  ✓ FFT output allocated");
  
  window = (float*)ps_malloc(N_FFT * sizeof(float));
  if (!window) {
    Serial.println("Failed to allocate window");
    return false;
  }
  Serial.println("  ✓ Window allocated");
  
  // MFCC processing buffers
  windowedFrame = (float*)ps_malloc(N_FFT * sizeof(float));
  if (!windowedFrame) {
    Serial.println("Failed to allocate windowedFrame");
    return false;
  }
  Serial.println("  ✓ Windowed frame allocated");
  
  powerSpectrum = (float*)ps_malloc((N_FFT / 2 + 1) * sizeof(float));
  if (!powerSpectrum) {
    Serial.println("Failed to allocate powerSpectrum");
    return false;
  }
  Serial.println("  ✓ Power spectrum allocated");
  
  melEnergies = (float*)ps_malloc(NUM_MEL_FILTERS * sizeof(float));
  if (!melEnergies) {
    Serial.println("Failed to allocate melEnergies");
    return false;
  }
  Serial.println("  ✓ Mel energies allocated");
  
  // Frame buffers - THIS IS THE KEY FIX!
  frame = (int16_t*)ps_malloc(N_FFT * sizeof(int16_t));
  if (!frame) {
    Serial.println("Failed to allocate frame buffer");
    return false;
  }
  Serial.println("  ✓ Frame buffer allocated");
  
  nextSamples = (int16_t*)ps_malloc(HOP_LENGTH * sizeof(int16_t));
  if (!nextSamples) {
    Serial.println("Failed to allocate nextSamples buffer");
    return false;
  }
  Serial.println("  ✓ Next samples buffer allocated");
  
  // Mel filterbank
  int numBins = N_FFT / 2 + 1;
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
  Serial.println("  ✓ Mel filterbank allocated");
  
  Serial.println("\nAll buffers allocated successfully!");
  Serial.print("Free PSRAM: ");
  Serial.print(ESP.getFreePsram());
  Serial.println(" bytes\n");
  
  return true;
}

void freeBuffers() {
  if (fft_input) free(fft_input);
  if (fft_output) free(fft_output);
  if (window) free(window);
  if (windowedFrame) free(windowedFrame);
  if (powerSpectrum) free(powerSpectrum);
  if (melEnergies) free(melEnergies);
  if (frame) free(frame);
  if (nextSamples) free(nextSamples);
  
  if (melFilterbank) {
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
      if (melFilterbank[i]) free(melFilterbank[i]);
    }
    free(melFilterbank);
  }
}

// ==================== Classifier Functions ====================

bool loadClassifierModel() {
  File modelFile = LittleFS.open(CLASSIFIER_MODEL_PATH, FILE_READ);
  if (!modelFile) {
    Serial.println("Classifier model file not found.");
    classifier.isLoaded = false;
    return false;
  }
  
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
  
  for (int i = 0; i < NUM_CLASSES; i++) {
    modelFile.read((uint8_t*)classifier.classNames[i], 32);
  }
  
  for (int i = 0; i < NUM_CLASSES; i++) {
    modelFile.read((uint8_t*)classifier.means[i], NUM_MFCC * sizeof(float));
  }
  
  for (int i = 0; i < NUM_CLASSES; i++) {
    modelFile.read((uint8_t*)classifier.variances[i], NUM_MFCC * sizeof(float));
  }
  
  modelFile.read((uint8_t*)classifier.priors, NUM_CLASSES * sizeof(float));
  
  modelFile.close();
  classifier.isLoaded = true;
  
  Serial.println("Classifier model loaded successfully!");
  printClassifierModel();
  return true;
}

void printClassifierModel() {
  Serial.println("\n=== Classifier Model ===");
  Serial.print("Number of classes: "); Serial.println(NUM_CLASSES);
  
  for (int c = 0; c < NUM_CLASSES; c++) {
    Serial.print("\nClass "); Serial.print(c); Serial.print(": ");
    Serial.println(classifier.classNames[c]);
    Serial.print("  Prior: "); Serial.println(classifier.priors[c], 4);
    
    Serial.print("  Means: [");
    for (int i = 0; i < min(5, NUM_MFCC); i++) {
      Serial.print(classifier.means[c][i], 3);
      if (i < 4) Serial.print(", ");
    }
    Serial.println("...]");
  }
  Serial.println("========================\n");
}

float gaussianPDF(float x, float mean, float variance) {
  if (variance <= 0) variance = 1e-6;
  float exponent = -0.5 * pow((x - mean), 2) / variance;
  float coefficient = 1.0 / sqrt(2.0 * M_PI * variance);
  return coefficient * exp(exponent);
}

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

// ==================== MFCC Processing ====================

float hzToMel(float hz) {
  return 2595.0 * log10(1.0 + hz / 700.0);
}

float melToHz(float mel) {
  return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

void initMelFilterbank() {
  float melLow = hzToMel(MEL_LOW_FREQ);
  float melHigh = hzToMel(MEL_HIGH_FREQ);
  
  float melPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    melPoints[i] = melLow + (melHigh - melLow) * i / (NUM_MEL_FILTERS + 1);
  }
  
  float hzPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    hzPoints[i] = melToHz(melPoints[i]);
  }
  
  int binPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    binPoints[i] = (int)floor((N_FFT + 1) * hzPoints[i] / SAMPLE_RATE);
  }
  
  int numBins = N_FFT / 2 + 1;
  
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    int leftBin = binPoints[m];
    int centerBin = binPoints[m + 1];
    int rightBin = binPoints[m + 2];
    
    for (int k = 0; k < numBins; k++) {
      melFilterbank[m][k] = 0.0;
    }
    
    for (int k = leftBin; k < centerBin && k < numBins; k++) {
      if (centerBin != leftBin) {
        melFilterbank[m][k] = (float)(k - leftBin) / (centerBin - leftBin);
      }
    }
    
    for (int k = centerBin; k < rightBin && k < numBins; k++) {
      if (rightBin != centerBin) {
        melFilterbank[m][k] = (float)(rightBin - k) / (rightBin - centerBin);
      }
    }
  }
  
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    float enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m]);
    for (int k = 0; k < numBins; k++) {
      melFilterbank[m][k] *= enorm;
    }
  }
}

void applyHannWindow(int16_t* input, float* output, int length) {
  for (int i = 0; i < length; i++) {
    float normalized = input[i] / 32768.0f;
    output[i] = normalized * window[i];
  }
}

void computePowerSpectrum(float* input, float* output, int length) {
  for (int i = 0; i < length; i++) {
    fft_input[i * 2] = input[i];
    fft_input[i * 2 + 1] = 0.0;
  }
  
  for (int i = length; i < N_FFT; i++) {
    fft_input[i * 2] = 0.0;
    fft_input[i * 2 + 1] = 0.0;
  }
  
  dsps_fft2r_fc32(fft_input, N_FFT);
  dsps_bit_rev_fc32(fft_input, N_FFT);
  
  int numBins = N_FFT / 2 + 1;
  for (int i = 0; i < numBins; i++) {
    float real = fft_input[i * 2];
    float imag = fft_input[i * 2 + 1];
    output[i] = (real * real + imag * imag) / (float)N_FFT;
  }
}

void applyMelFilters(float* powerSpectrumIn, float* melEnergiesOut) {
  int numBins = N_FFT / 2 + 1;
  
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    double energy = 0.0;
    for (int k = 0; k < numBins; k++) {
      energy += (double)powerSpectrumIn[k] * (double)melFilterbank[m][k];
    }
    melEnergiesOut[m] = log(energy + 1e-10);
  }
}

void computeDCT(float* melEnergiesIn, float* mfccOut, int numFilters, int numCoeffs) {
  for (int i = 0; i < numCoeffs; i++) {
    float sum = 0.0;
    for (int j = 0; j < numFilters; j++) {
      sum += melEnergiesIn[j] * cos(M_PI * i * (j + 0.5) / numFilters);
    }
    
    if (i == 0) {
      mfccOut[i] = sum * sqrt(1.0 / numFilters);
    } else {
      mfccOut[i] = sum * sqrt(2.0 / numFilters);
    }
  }
}

void extractMFCC(int16_t* frameIn) {
  applyHannWindow(frameIn, windowedFrame, N_FFT);
  computePowerSpectrum(windowedFrame, powerSpectrum, N_FFT);
  applyMelFilters(powerSpectrum, melEnergies);
  computeDCT(melEnergies, mfcc_coeffs, NUM_MEL_FILTERS, NUM_MFCC);
}

// ==================== WAV File Post-Processing ====================

void processWavFileToMFCC() {
  File wavFileInput = LittleFS.open(WAV_FILE_PATH, FILE_READ);
  if (!wavFileInput) {
    Serial.println("Failed to open WAV file for MFCC processing");
    return;
  }
  
  wavFileInput.seek(WAV_HEADER_SIZE);
  
  if (LittleFS.exists(MFCC_FILE_PATH)) LittleFS.remove(MFCC_FILE_PATH);
  File mfccFileOutput = LittleFS.open(MFCC_FILE_PATH, FILE_WRITE);
  if (!mfccFileOutput) {
    Serial.println("Failed to create MFCC file");
    wavFileInput.close();
    return;
  }
  
  uint32_t numMfcc = NUM_MFCC;
  uint32_t frameSizeMs = FRAME_SIZE_MS;
  uint32_t frameStrideMs = FRAME_STRIDE_MS;
  uint32_t sampleRate = SAMPLE_RATE;
  
  mfccFileOutput.write((uint8_t*)&numMfcc, sizeof(uint32_t));
  mfccFileOutput.write((uint8_t*)&frameSizeMs, sizeof(uint32_t));
  mfccFileOutput.write((uint8_t*)&frameStrideMs, sizeof(uint32_t));
  mfccFileOutput.write((uint8_t*)&sampleRate, sizeof(uint32_t));
  
  uint32_t frameCount = 0;
  
  Serial.println("\n=== Processing WAV file to MFCC ===");
  Serial.print("Frame size: "); Serial.print(N_FFT); Serial.println(" samples");
  Serial.print("Hop length: "); Serial.print(HOP_LENGTH); Serial.println(" samples");
  Serial.println();
  
  // Read first full frame
  size_t bytesRead = wavFileInput.read((uint8_t*)frame, N_FFT * sizeof(int16_t));
  uint32_t totalSamples = bytesRead / sizeof(int16_t);
  
  if (totalSamples < N_FFT) {
    Serial.println("WAV file too short for MFCC extraction");
    wavFileInput.close();
    mfccFileOutput.close();
    return;
  }
  
  mfccAccumulator.reset();
  
  Serial.println("Extracting MFCC from first frame...");
  extractMFCC(frame);
  mfccFileOutput.write((uint8_t*)mfcc_coeffs, NUM_MFCC * sizeof(float));
  mfccAccumulator.addFrame(mfcc_coeffs);
  frameCount++;
  Serial.println("Frame 1 processed");
  
  // Process remaining frames - NOW USING PSRAM BUFFERS!
  int frameProgress = 0;
  
  while (true) {
    bytesRead = wavFileInput.read((uint8_t*)nextSamples, HOP_LENGTH * sizeof(int16_t));
    if (bytesRead < HOP_LENGTH * sizeof(int16_t)) {
      break;
    }
    
    // These now work with PSRAM buffers - no stack overflow!
    memmove(frame, frame + HOP_LENGTH, (N_FFT - HOP_LENGTH) * sizeof(int16_t));
    memcpy(frame + (N_FFT - HOP_LENGTH), nextSamples, HOP_LENGTH * sizeof(int16_t));
    
    extractMFCC(frame);
    mfccFileOutput.write((uint8_t*)mfcc_coeffs, NUM_MFCC * sizeof(float));
    mfccAccumulator.addFrame(mfcc_coeffs);
    frameCount++;
    
    frameProgress++;
    if (frameProgress % 10 == 0) {
      Serial.print("Frame ");
      Serial.print(frameCount);
      Serial.println(" processed");
      delay(5);
    }
  }
  
  wavFileInput.close();
  mfccFileOutput.close();
  
  Serial.print("\nTotal frames: ");
  Serial.println(frameCount);
  Serial.print("Duration: ");
  Serial.print((frameCount * HOP_LENGTH) / (float)SAMPLE_RATE, 2);
  Serial.println(" seconds");
  Serial.println("=== Processing complete ===\n");
}

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
  
  MFCCAccumulator accumulator;
  accumulator.reset();
  
  float coeffs[NUM_MFCC];
  int frameNum = 0;
  
  Serial.println("Reading all frames...");
  while (f.available() >= NUM_MFCC * sizeof(float)) {
    f.read((uint8_t*)coeffs, NUM_MFCC * sizeof(float));
    accumulator.addFrame(coeffs);
    frameNum++;
  }
  
  f.close();
  
  if (frameNum == 0) {
    Serial.println("ERROR: No MFCC frames found!");
    return;
  }
  
  float meanMFCC[NUM_MFCC];
  accumulator.getMean(meanMFCC);
  
  Serial.print("Total frames: ");
  Serial.println(frameNum);
  Serial.println("\nMean MFCC: [");
  for (int i = 0; i < NUM_MFCC; i++) {
    Serial.print(meanMFCC[i], 4);
    if (i < NUM_MFCC - 1) Serial.print(", ");
  }
  Serial.println("]");
  
  float probabilities[NUM_CLASSES];
  int predictedClass = predictClass(meanMFCC, probabilities);
  
  Serial.println("\n=== Classification Result ===");
  Serial.print("Predicted: ");
  Serial.println(classifier.classNames[predictedClass]);
  Serial.print("Confidence: ");
  Serial.print(probabilities[predictedClass] * 100, 2);
  Serial.println("%\n");
  
  Serial.println("All probabilities:");
  for (int c = 0; c < NUM_CLASSES; c++) {
    Serial.print("  ");
    Serial.print(classifier.classNames[c]);
    Serial.print(": ");
    Serial.print(probabilities[c] * 100, 2);
    Serial.println("%");
  }
  Serial.println("==============================\n");
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
  
  // Serial.println("\n=== WAV FILE DUMP ===");
  WAVHeader header;
  f.read((uint8_t*)&header, WAV_HEADER_SIZE);
  f.seek(0);
  uint8_t buffer[512];
  while (f.available()) {
    size_t bytesRead = f.read(buffer, sizeof(buffer));
    Serial.write(buffer, bytesRead);
  }
  
  // Serial.println("\n\n=== END DUMP ===\n");
  f.close();
}

void dumpMfccFile() {
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
  
  Serial.println("\n=== MFCC FILE DUMP ===");
  Serial.print("Num Coefficients: "); Serial.println(numCoeffs);
  Serial.print("Frame Size: "); Serial.print(frameSizeMs); Serial.println(" ms");
  Serial.print("Hop Length: "); Serial.print(frameStrideMs); Serial.println(" ms");
  Serial.println("\nLibrosa parameters:");
  Serial.print("  n_mfcc="); Serial.println(NUM_MFCC);
  Serial.print("  n_fft="); Serial.println(N_FFT);
  Serial.print("  hop_length="); Serial.println(HOP_LENGTH);
  Serial.println("\nFrames:");
  
  MFCCAccumulator accumulator;
  accumulator.reset();
  int frameNum = 0;
  float coeffs[NUM_MFCC];
  
  while (f.available() >= NUM_MFCC * sizeof(float)) {
    f.read((uint8_t*)coeffs, NUM_MFCC * sizeof(float));
    accumulator.addFrame(coeffs);
    
    Serial.print("  Frame ");
    Serial.print(frameNum);
    Serial.print(": [");
    for (int i = 0; i < NUM_MFCC; i++) {
      Serial.print(coeffs[i], 3);
      if (i < NUM_MFCC - 1) Serial.print(", ");
    }
    Serial.println("]");
    frameNum++;
  }
  
  f.close();
  
  if (frameNum > 0) {
    float meanMFCC[NUM_MFCC];
    accumulator.getMean(meanMFCC);
    
    Serial.print("\nMean MFCC over ");
    Serial.print(frameNum);
    Serial.println(" frames:");
    Serial.print("  [");
    for (int i = 0; i < NUM_MFCC; i++) {
      Serial.print(meanMFCC[i], 4);
      if (i < NUM_MFCC - 1) Serial.print(", ");
    }
    Serial.println("]");
  }
  
  Serial.println("\n=== END DUMP ===\n");
}

// ==================== Main ====================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\nESP32-S3 Librosa-Compatible MFCC Processor");
  Serial.println("==========================================");
  Serial.println("Parameters:");
  Serial.print("  N_FFT: "); Serial.println(N_FFT);
  Serial.print("  HOP_LENGTH: "); Serial.println(HOP_LENGTH);
  Serial.print("  NUM_MFCC: "); Serial.println(NUM_MFCC);
  Serial.print("  SAMPLE_RATE: "); Serial.println(SAMPLE_RATE);
  Serial.println();
  
  if (!psramFound()) {
    Serial.println("ERROR: PSRAM not found!");
    while(1) delay(1000);
  }
  
  Serial.print("PSRAM available: ");
  Serial.print(ESP.getPsramSize() / 1024);
  Serial.println(" KB");
  
  if (!allocateBuffers()) {
    Serial.println("FATAL: Buffer allocation failed!");
    while(1) delay(1000);
  }
  
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  if (!LittleFS.begin(true)) {
    Serial.println("LittleFS mount failed");
    return;
  }
  Serial.println("LittleFS mounted\n");
  
  esp_err_t ret = dsps_fft2r_init_fc32(NULL, N_FFT);
  if (ret != ESP_OK) {
    Serial.println("FFT init failed");
    return;
  }
  
  dsps_wind_hann_f32(window, N_FFT);
  initMelFilterbank();
  mfccAccumulator.reset();
  
  Serial.println("Initialization complete!\n");
  listFiles();
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
    Serial.println("I2S install failed");
    return;
  }
  
  if (i2s_set_pin(I2S_PORT, &pin_config) != ESP_OK) {
    Serial.println("I2S set pin failed");
    return;
  }
  
  Serial.println("I2S configured\n");
  Serial.println("=== Commands ===");
  Serial.println("  Button GPIO16 - record");
  Serial.println("  'process'     - extract MFCC from WAV");
  Serial.println("  'classify'    - classify MFCC");
  Serial.println("  'dump'        - dump WAV file");
  Serial.println("  'mfcc'        - dump MFCC file");
  Serial.println("  'files'       - list files");
  Serial.println("================\n");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "process") {
      processWavFileToMFCC();
    } else if (command == "classify") {
      classifyStoredMFCC();
    } else if (command == "dump") {
      dumpWavFile();
    } else if (command == "mfcc") {
      dumpMfccFile();
    } else if (command == "files") {
      listFiles();
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
  Serial.println("Recording...");
  if (LittleFS.exists(WAV_FILE_PATH)) LittleFS.remove(WAV_FILE_PATH);
  
  wavFile = LittleFS.open(WAV_FILE_PATH, FILE_WRITE);
  if (!wavFile) {
    Serial.println("Failed to open WAV file");
    return;
  }
  
  WAVHeader header;
  wavFile.write((uint8_t*)&header, WAV_HEADER_SIZE);
  
  recordedSamples = 0;
  isRecording = true;
}

void stopRecording() {
  isRecording = false;
  updateWavHeader();
  wavFile.close();
  
  Serial.print("Stopped. Samples: ");
  Serial.print(recordedSamples);
  Serial.print(" (");
  Serial.print(recordedSamples / (float)SAMPLE_RATE, 2);
  Serial.println("s)");
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