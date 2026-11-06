// Resample the device's actual rate to 16 kHz and send 32 ms PCM16 frames.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.phase = 0; this.previous = 0; this.frame = new Int16Array(512); this.index = 0;
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    const step = sampleRate / 16000;
    for (let i = 0; i < input.length; i++) {
      while (this.phase <= 1) {
        const value = this.previous + (input[i] - this.previous) * this.phase;
        this.frame[this.index++] = Math.round(Math.max(-1, Math.min(1, value)) * 32767);
        if (this.index === 512) {
          this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
          this.frame = new Int16Array(512); this.index = 0;
        }
        this.phase += step;
      }
      this.phase -= 1; this.previous = input[i];
    }
    return true;
  }
}
registerProcessor('capture', Capture);
