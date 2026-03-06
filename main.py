from machine import Pin, I2S
import struct
import time
import sys

# === I2S MICROPHONE SETTINGS ===
SCK_PIN = 15   # Serial Clock (BCLK)
WS_PIN = 7     # Word Select (LRCLK/WS)
SD_PIN = 41    # Serial Data (DOUT)

# === AUDIO SETTINGS ===
SAMPLE_RATE = 16000
RECORDING_DURATION = 3
BITS_PER_SAMPLE = 32  # INMP441 requires 32-bit mode, but uses 24-bit data
TOTAL_SAMPLES = SAMPLE_RATE * RECORDING_DURATION
BUFFER_SIZE = 8192

# === BUTTON SETTINGS ===
RECORD_BUTTON_PIN = 16

# === GLOBAL VARIABLES ===
audio_buffer = bytearray(TOTAL_SAMPLES * 2)  # Final 16-bit samples
recording = False
i2s = None
button = None

# === SETUP ===
def setup():
    global i2s, button
    
    print("Initializing I2S microphone...")
    
    # Initialize I2S for INMP441 microphone
    # INMP441 requires 32-bit format even though it outputs 24-bit data
    try:
        i2s = I2S(
            0,  # I2S ID
            sck=Pin(SCK_PIN),
            ws=Pin(WS_PIN),
            sd=Pin(SD_PIN),
            mode=I2S.RX,
            bits=32,  # INMP441 needs 32-bit clock cycles
            format=I2S.MONO,
            rate=SAMPLE_RATE,
            ibuf=BUFFER_SIZE * 2
        )
        print("I2S initialized successfully")
    except Exception as e:
        print(f"I2S initialization error: {e}")
        return False
    
    # Initialize button (hardware debounced, active LOW)
    button = Pin(RECORD_BUTTON_PIN, Pin.IN, Pin.PULL_UP)
    
    print("Setup complete.")
    print("Press button to start/stop recording.")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Recording Duration: {RECORDING_DURATION} seconds")
    return True

# === AUDIO RECORDING ===
def record_audio():
    global audio_buffer
    
    print("Recording...")
    
    # Buffer for reading 32-bit samples from I2S
    read_buffer = bytearray(BUFFER_SIZE)
    total_bytes_needed = TOTAL_SAMPLES * 4  # 4 bytes per 32-bit sample
    bytes_read = 0
    sample_index = 0
    
    # Read audio samples from I2S microphone
    while bytes_read < total_bytes_needed and sample_index < TOTAL_SAMPLES:
        try:
            # Read chunk from I2S
            num_bytes = i2s.readinto(read_buffer)
            
            if num_bytes > 0:
                # Process 32-bit samples, extract 16-bit data
                for i in range(0, num_bytes, 4):
                    if sample_index >= TOTAL_SAMPLES:
                        break
                    
                    # INMP441 outputs 24-bit data in 32-bit format
                    # The 24-bit data is in the upper 24 bits
                    # We extract the middle 16 bits for better quality
                    byte0 = read_buffer[i]
                    byte1 = read_buffer[i + 1]
                    byte2 = read_buffer[i + 2]
                    byte3 = read_buffer[i + 3]
                    
                    # Reconstruct 32-bit signed integer
                    sample_32 = struct.unpack('<i', bytes([byte0, byte1, byte2, byte3]))[0]
                    
                    # Shift right by 16 to get the most significant 16 bits
                    sample_16 = (sample_32 >> 16) & 0xFFFF
                    
                    # Convert to signed 16-bit
                    if sample_16 > 32767:
                        sample_16 -= 65536
                    
                    # Store as 16-bit sample
                    audio_buffer[sample_index * 2] = sample_16 & 0xFF
                    audio_buffer[sample_index * 2 + 1] = (sample_16 >> 8) & 0xFF
                    
                    sample_index += 1
                
                bytes_read += num_bytes
                
                # Progress indicator
                if sample_index % (SAMPLE_RATE // 2) == 0:
                    print(f"Progress: {sample_index // SAMPLE_RATE}s")
        
        except Exception as e:
            print(f"Error reading I2S: {e}")
            break
    
    print(f"Recording complete. Captured {sample_index} samples ({bytes_read} bytes read).")
    return sample_index

# === DISPLAY SUMMARY ===
def display_summary():
    min_val = 32767
    max_val = -32768
    sum_val = 0
    
    # Parse 16-bit signed samples
    num_samples = len(audio_buffer) // 2
    for i in range(0, len(audio_buffer), 2):
        sample = struct.unpack('<h', audio_buffer[i:i+2])[0]
        min_val = min(min_val, sample)
        max_val = max(max_val, sample)
        sum_val += sample
    
    avg = sum_val / num_samples if num_samples > 0 else 0
    
    # Calculate RMS for volume indication
    sum_squares = 0
    for i in range(0, len(audio_buffer), 2):
        sample = struct.unpack('<h', audio_buffer[i:i+2])[0]
        sum_squares += sample * sample
    
    rms = (sum_squares / num_samples) ** 0.5 if num_samples > 0 else 0
    
    print("=== Recording Summary ===")
    print(f"Min: {min_val}, Max: {max_val}")
    print(f"Avg: {avg:.2f}")
    print(f"RMS: {rms:.2f}")
    print(f"Total Samples: {num_samples}")
    print(f"Peak-to-Peak: {max_val - min_val}")
    print("=========================")

# === SAVE WAV FILE ===
def save_audio_to_wav():
    filename = "/audio.wav"
    
    print(f"Saving WAV file to {filename}...")
    
    try:
        with open(filename, "wb") as f:
            # WAV file parameters
            num_channels = 1
            bits_per_sample = 16
            byte_rate = SAMPLE_RATE * num_channels * bits_per_sample // 8
            block_align = num_channels * bits_per_sample // 8
            data_size = len(audio_buffer)
            file_size = data_size + 36
            
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', file_size))
            f.write(b'WAVE')
            
            # fmt subchunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # Subchunk size
            f.write(struct.pack('<H', 1))   # Audio format (PCM)
            f.write(struct.pack('<H', num_channels))
            f.write(struct.pack('<I', SAMPLE_RATE))
            f.write(struct.pack('<I', byte_rate))
            f.write(struct.pack('<H', block_align))
            f.write(struct.pack('<H', bits_per_sample))
            
            # data subchunk
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            f.write(audio_buffer)
        
        print(f"WAV file saved successfully ({file_size + 8} bytes)")
        
    except Exception as e:
        print(f"Error saving WAV file: {e}")

# === DUMP FILE TO SERIAL ===
def dump_file():
    filename = "/audio.wav"
    
    try:
        # print(f"Dumping {filename}...")
        with open(filename, "rb") as f:
            data = f.read()
            sys.stdout.buffer.write(data)
        # print("\n--- End of file ---")
        
    except Exception as e:
        print(f"Error reading file: {e}")

# === CHECK SERIAL INPUT ===
def check_serial():
    if sys.stdin in sys.stdin.buffer.read():
        return False
    return False

# === MAIN LOOP ===
def main():
    global recording
    
    if not setup():
        print("Setup failed!")
        return
    
    last_button_state = 1
    
    # Test I2S by reading a small sample
    print("Testing I2S microphone...")
    test_buf = bytearray(128)
    try:
        test_bytes = i2s.readinto(test_buf)
        print(f"I2S test read: {test_bytes} bytes")
        print(f"First few bytes: {list(test_buf[:16])}")
    except Exception as e:
        print(f"I2S test failed: {e}")
    
    print("\nReady. Press button to record.")
    
    while True:
        # Simple serial command check
        try:
            # Try to read without blocking (won't work perfectly but simple)
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip()
                if cmd == "dump":
                    dump_file()
        except:
            # If select not available, skip serial checking
            pass
        
        # Button handling (hardware debounced)
        current_button_state = button.value()
        
        # Detect button press (falling edge, active LOW)
        if last_button_state == 1 and current_button_state == 0:
            if not recording:
                recording = True
                print("\n=== RECORDING STARTED ===")
                samples_recorded = record_audio()
                display_summary()
                save_audio_to_wav()
                print("=== RECORDING COMPLETE ===\n")
                recording = False
        
        last_button_state = current_button_state
        time.sleep_ms(10)

# === RUN ===
if __name__ == "__main__":
    main()