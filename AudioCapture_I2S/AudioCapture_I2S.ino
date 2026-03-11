#include <driver/i2s.h>
#include <LittleFS.h>

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

// WAV file configuration
#define WAV_FILE_PATH "/recording.wav"
#define WAV_HEADER_SIZE 44

// Recording state
bool isRecording = false;
bool lastButtonState = HIGH;
File wavFile;
uint32_t recordedSamples = 0;

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

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("ESP32-S3 INMP441 Recorder");
  
  // Initialize button
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Initialize LittleFS
  if (!LittleFS.begin(true)) {
    Serial.println("LittleFS Mount Failed");
    return;
  }
  Serial.println("LittleFS mounted successfully");
  
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
  Serial.println("Send 'dump' command to print WAV file to serial");
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "dump") {
      dumpWavFile();
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
  
  // Delete existing file if it exists
  if (LittleFS.exists(WAV_FILE_PATH)) {
    LittleFS.remove(WAV_FILE_PATH);
  }
  
  // Open file for writing
  wavFile = LittleFS.open(WAV_FILE_PATH, FILE_WRITE);
  if (!wavFile) {
    Serial.println("Failed to open file for writing");
    return;
  }
  
  // Write placeholder WAV header (will be updated when recording stops)
  WAVHeader header;
  wavFile.write((uint8_t*)&header, WAV_HEADER_SIZE);
  
  recordedSamples = 0;
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
  
  Serial.print("Recording stopped. Samples recorded: ");
  Serial.println(recordedSamples);
  
  // Print file size
  File f = LittleFS.open(WAV_FILE_PATH, FILE_READ);
  if (f) {
    Serial.print("File size: ");
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
  // Serial.println("\n=== WAV FILE DUMP START ===");
  
  File f = LittleFS.open(WAV_FILE_PATH, FILE_READ);
  if (!f) {
    Serial.println("Failed to open WAV file");
    return;
  }
  
  // // Print file size first
  // size_t fileSize = f.size();
  // Serial.print("SIZE:");
  // Serial.println(fileSize);
  
  // Dump raw bytes directly to serial
  uint8_t buffer[512];
  while (f.available()) {
    size_t bytesRead = f.read(buffer, sizeof(buffer));
    Serial.write(buffer, bytesRead);
  }
  
  f.close();
  // Serial.println("\n=== WAV FILE DUMP END ===");
}