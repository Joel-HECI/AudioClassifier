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

// MFCC Configuration
#define FRAME_SIZE_MS 25
#define FRAME_STRIDE_MS 10
#define FRAME_SIZE (SAMPLE_RATE * FRAME_SIZE_MS / 1000)  // 400 samples
#define FRAME_STRIDE (SAMPLE_RATE * FRAME_STRIDE_MS / 1000)  // 160 samples
#define FFT_SIZE 512  // Next power of 2 >= FRAME_SIZE
#define NUM_MFCC 13
#define NUM_MEL_FILTERS 26
#define MEL_LOW_FREQ 0
#define MEL_HIGH_FREQ (SAMPLE_RATE / 2)

// WAV file configuration
#define WAV_FILE_PATH "/recording.wav"
#define MFCC_FILE_PATH "/mfcc_features.bin"
#define WAV_HEADER_SIZE 44

// Recording state
bool isRecording = false;
bool lastButtonState = HIGH;
File wavFile;
File mfccFile;
uint32_t recordedSamples = 0;
uint32_t mfccFrameCount = 0;

// MFCC buffers
int16_t audioBuffer[FRAME_SIZE + FRAME_STRIDE];  // Sliding window buffer
int16_t* frameBuffer = audioBuffer;  // Points to current frame
uint32_t bufferIndex = 0;

// FFT buffers (ESP-DSP requires 2*N for complex FFT)
__attribute__((aligned(16))) float fft_input[FFT_SIZE * 2];
__attribute__((aligned(16))) float fft_output[FFT_SIZE];
float window[FFT_SIZE];

// Mel filterbank
float melFilterbank[NUM_MEL_FILTERS][FFT_SIZE / 2 + 1];
float mfcc_coeffs[NUM_MFCC];

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

// Helper function: Convert Hz to Mel scale
float hzToMel(float hz) {
  return 2595.0 * log10(1.0 + hz / 700.0);
}

// Helper function: Convert Mel to Hz scale
float melToHz(float mel) {
  return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

// Initialize Mel filterbank
void initMelFilterbank() {
  float melLow = hzToMel(MEL_LOW_FREQ);
  float melHigh = hzToMel(MEL_HIGH_FREQ);
  float melStep = (melHigh - melLow) / (NUM_MEL_FILTERS + 1);
  
  // Calculate center frequencies in Mel scale
  float melPoints[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
    melPoints[i] = melLow + i * melStep;
  }
  
  // Convert back to Hz and then to FFT bin indices
  float fftFreqs[FFT_SIZE / 2 + 1];
  for (int i = 0; i < FFT_SIZE / 2 + 1; i++) {
    fftFreqs[i] = (float)i * SAMPLE_RATE / FFT_SIZE;
  }
  
  // Create triangular filters
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    float leftMel = melPoints[m];
    float centerMel = melPoints[m + 1];
    float rightMel = melPoints[m + 2];
    
    float leftHz = melToHz(leftMel);
    float centerHz = melToHz(centerMel);
    float rightHz = melToHz(rightMel);
    
    for (int k = 0; k < FFT_SIZE / 2 + 1; k++) {
      float freq = fftFreqs[k];
      
      if (freq >= leftHz && freq <= centerHz) {
        melFilterbank[m][k] = (freq - leftHz) / (centerHz - leftHz);
      } else if (freq > centerHz && freq <= rightHz) {
        melFilterbank[m][k] = (rightHz - freq) / (rightHz - centerHz);
      } else {
        melFilterbank[m][k] = 0.0;
      }
    }
  }
}

// Apply Hamming window
void applyHammingWindow(int16_t* input, float* output, int length) {
  for (int i = 0; i < length; i++) {
    output[i] = (float)input[i] * window[i];
  }
}

// Compute power spectrum using ESP-DSP FFT
void computePowerSpectrum(float* input, float* output, int length) {
  // Prepare complex input for FFT (real, imag, real, imag, ...)
  for (int i = 0; i < length; i++) {
    fft_input[i * 2] = input[i];      // Real part
    fft_input[i * 2 + 1] = 0.0;       // Imaginary part
  }
  
  // Zero-pad if necessary
  for (int i = length; i < FFT_SIZE; i++) {
    fft_input[i * 2] = 0.0;
    fft_input[i * 2 + 1] = 0.0;
  }
  
  // Perform FFT using ESP-DSP
  dsps_fft2r_fc32(fft_input, FFT_SIZE);
  dsps_bit_rev_fc32(fft_input, FFT_SIZE);
  
  // Compute power spectrum: |X[k]|^2
  for (int i = 0; i < FFT_SIZE / 2 + 1; i++) {
    float real = fft_input[i * 2];
    float imag = fft_input[i * 2 + 1];
    output[i] = real * real + imag * imag;
  }
}

// Apply Mel filterbank and compute log energy
void applyMelFilters(float* powerSpectrum, float* melEnergies) {
  for (int m = 0; m < NUM_MEL_FILTERS; m++) {
    float energy = 0.0;
    for (int k = 0; k < FFT_SIZE / 2 + 1; k++) {
      energy += powerSpectrum[k] * melFilterbank[m][k];
    }
    // Add small epsilon to avoid log(0)
    melEnergies[m] = log(energy + 1e-10);
  }
}

// Compute DCT (Discrete Cosine Transform) for MFCC
void computeDCT(float* melEnergies, float* mfcc, int numFilters, int numCoeffs) {
  for (int i = 0; i < numCoeffs; i++) {
    float sum = 0.0;
    for (int j = 0; j < numFilters; j++) {
      sum += melEnergies[j] * cos(M_PI * i * (j + 0.5) / numFilters);
    }
    mfcc[i] = sum;
  }
}

// Main MFCC extraction function
void extractMFCC(int16_t* frame) {
  static float windowedFrame[FFT_SIZE];
  static float powerSpectrum[FFT_SIZE / 2 + 1];
  static float melEnergies[NUM_MEL_FILTERS];
  
  // 1. Apply Hamming window
  applyHammingWindow(frame, windowedFrame, FRAME_SIZE);
  
  // 2. Compute power spectrum using ESP-DSP FFT
  computePowerSpectrum(windowedFrame, powerSpectrum, FRAME_SIZE);
  
  // 3. Apply Mel filterbank
  applyMelFilters(powerSpectrum, melEnergies);
  
  // 4. Compute DCT to get MFCC coefficients
  computeDCT(melEnergies, mfcc_coeffs, NUM_MEL_FILTERS, NUM_MFCC);
  
  // 5. Write MFCC coefficients to file
  if (mfccFile) {
    mfccFile.write((uint8_t*)mfcc_coeffs, NUM_MFCC * sizeof(float));
    mfccFrameCount++;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("ESP32-S3 INMP441 Recorder with MFCC");
  
  // Initialize button
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Initialize LittleFS
  if (!LittleFS.begin(true)) {
    Serial.println("LittleFS Mount Failed");
    return;
  }
  Serial.println("LittleFS mounted successfully");
  
  // Initialize ESP-DSP FFT
  esp_err_t ret = dsps_fft2r_init_fc32(NULL, FFT_SIZE);
  if (ret != ESP_OK) {
    Serial.println("Failed to initialize ESP-DSP FFT");
    return;
  }
  Serial.println("ESP-DSP FFT initialized");
  
  // Generate Hamming window
  dsps_wind_hann_f32(window, FFT_SIZE);
  
  // Initialize Mel filterbank
  initMelFilterbank();
  Serial.println("Mel filterbank initialized");
  
  // Initialize I2S
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
  Serial.println("Press button on GPIO16 to start/stop recording");
  Serial.println("Commands:");
  Serial.println("  'dump' - dump WAV file");
  Serial.println("  'mfcc' - dump MFCC features");
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "dump") {
      dumpWavFile();
    } else if (command == "mfcc") {
      dumpMfccFile();
    }
  }
  
  // Check button state (hardware debounced)
  bool currentButtonState = digitalRead(BUTTON_PIN);
  
  if (currentButtonState == LOW && lastButtonState == HIGH) {
    // Button pressed
    toggleRecording();
    delay(50); // Small delay for stability
  }
  
  lastButtonState = currentButtonState;
  
  // Record audio if recording is active
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
  
  // Delete existing files if they exist
  if (LittleFS.exists(WAV_FILE_PATH)) {
    LittleFS.remove(WAV_FILE_PATH);
  }
  if (LittleFS.exists(MFCC_FILE_PATH)) {
    LittleFS.remove(MFCC_FILE_PATH);
  }
  
  // Open WAV file for writing
  wavFile = LittleFS.open(WAV_FILE_PATH, FILE_WRITE);
  if (!wavFile) {
    Serial.println("Failed to open WAV file for writing");
    return;
  }
  
  // Open MFCC file for writing
  mfccFile = LittleFS.open(MFCC_FILE_PATH, FILE_WRITE);
  if (!mfccFile) {
    Serial.println("Failed to open MFCC file for writing");
    wavFile.close();
    return;
  }
  
  // Write placeholder WAV header (will be updated when recording stops)
  WAVHeader header;
  wavFile.write((uint8_t*)&header, WAV_HEADER_SIZE);
  
  // Write MFCC file header (metadata) - create temporary variables for #define constants
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
  memset(audioBuffer, 0, sizeof(audioBuffer));
  
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
  
  // Update WAV header with actual sizes
  updateWavHeader();
  
  wavFile.close();
  mfccFile.close();
  
  Serial.print("Recording stopped. Samples recorded: ");
  Serial.println(recordedSamples);
  Serial.print("MFCC frames computed: ");
  Serial.println(mfccFrameCount);
  
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
}

void recordAudio() {
  int32_t i2s_buffer[I2S_READ_LEN];
  size_t bytes_read = 0;
  
  // Read from I2S
  esp_err_t result = i2s_read(I2S_PORT, i2s_buffer, I2S_READ_LEN * sizeof(int32_t), &bytes_read, portMAX_DELAY);
  
  if (result == ESP_OK && bytes_read > 0) {
    int samples_read = bytes_read / sizeof(int32_t);
    
    // Convert 32-bit samples to 16-bit and write to file
    for (int i = 0; i < samples_read; i++) {
      // INMP441 outputs 24-bit data in 32-bit container, shift right to get 16-bit
      int16_t sample = (i2s_buffer[i] >> 14) & 0xFFFF;
      wavFile.write((uint8_t*)&sample, sizeof(int16_t));
      recordedSamples++;
      
      // Add sample to sliding window buffer
      audioBuffer[bufferIndex] = sample;
      bufferIndex++;
      
      // When we have enough samples for a frame, extract MFCC
      if (bufferIndex >= FRAME_SIZE) {
        extractMFCC(audioBuffer);
        
        // Shift buffer by FRAME_STRIDE samples (overlap)
        memmove(audioBuffer, audioBuffer + FRAME_STRIDE, 
                (FRAME_SIZE - FRAME_STRIDE) * sizeof(int16_t));
        bufferIndex = FRAME_SIZE - FRAME_STRIDE;
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
  
  // Seek to beginning and write updated header
  wavFile.seek(0);
  wavFile.write((uint8_t*)&header, WAV_HEADER_SIZE);
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
  Serial.print("Frame Size (ms): "); Serial.println(frameSizeMs);
  Serial.print("Frame Stride (ms): "); Serial.println(frameStrideMs);
  Serial.print("Sample Rate: "); Serial.println(sampleRate);
  Serial.println("\nMFCC Coefficients (frame by frame):");
  
  int frameNum = 0;
  float coeffs[NUM_MFCC];
  
  while (f.available() >= NUM_MFCC * sizeof(float)) {
    f.read((uint8_t*)coeffs, NUM_MFCC * sizeof(float));
    
    Serial.print("Frame "); Serial.print(frameNum); Serial.print(": ");
    for (int i = 0; i < NUM_MFCC; i++) {
      Serial.print(coeffs[i], 4);
      if (i < NUM_MFCC - 1) Serial.print(", ");
    }
    Serial.println();
    frameNum++;
  }
  
  Serial.println("=== END DUMP ===\n");
  f.close();
}