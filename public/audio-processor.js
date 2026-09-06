class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.nextOutputSample = 0
    this.inputOffset = 0
    this.frame = new Int16Array(320) // 20 ms at 16 kHz.
    this.frameOffset = 0
    this.threshold = 0.015
    this.port.onmessage = ({ data }) => {
      if (data.type === 'threshold') this.threshold = data.value
    }
  }

  process(inputs, outputs) {
    // Keep the graph pulled without ever playing microphone audio back to speakers.
    outputs[0]?.[0]?.fill(0)
    const input = inputs[0]?.[0]
    if (!input?.length) return true

    const end = this.inputOffset + input.length - 1
    const interval = sampleRate / 16_000
    while (this.nextOutputSample <= end) {
      const local = Math.max(0, this.nextOutputSample - this.inputOffset)
      const index = Math.min(input.length - 1, Math.floor(local))
      const next = Math.min(input.length - 1, index + 1)
      const value = input[index] + (input[next] - input[index]) * (local - index)
      this.frame[this.frameOffset] = Math.max(-32768, Math.min(32767, Math.round(value * 32767)))
      this.frameOffset += 1
      if (this.frameOffset === this.frame.length) {
        const level = Math.sqrt(this.frame.reduce((sum, sample) => sum + sample * sample, 0) / this.frame.length) / 32768
        this.port.postMessage({ type: 'level', level })
        if (level < this.threshold) this.frame.fill(0)
        const pcm = this.frame.buffer
        this.port.postMessage({ type: 'pcm', pcm }, [pcm])
        this.frame = new Int16Array(320)
        this.frameOffset = 0
      }
      this.nextOutputSample += interval
    }
    this.inputOffset += input.length
    return true
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor)
