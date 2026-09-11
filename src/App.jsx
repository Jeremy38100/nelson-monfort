import { useEffect, useRef, useState } from 'react'

const LANGUAGES = [
  { code: 'fr', name: 'Francais', flag: '🇫🇷' },
  { code: 'en', name: 'English', flag: '🇬🇧' },
  { code: 'es', name: 'Espanol', flag: '🇪🇸' },
  { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
  { code: 'it', name: 'Italiano', flag: '🇮🇹' },
  { code: 'pt', name: 'Portugues', flag: '🇵🇹' },
  { code: 'nl', name: 'Nederlands', flag: '🇳🇱' },
  { code: 'ja', name: '日本語', flag: '🇯🇵' },
  { code: 'ko', name: 'Korean', flag: '🇰🇷' },
  { code: 'zh', name: 'Chinese', flag: '🇨🇳' },
  { code: 'ar', name: 'Arabic', flag: '🇸🇦' },
  { code: 'ru', name: 'Russian', flag: '🇷🇺' },
]

const language = (code) => LANGUAGES.find((item) => item.code === code)

export default function App() {
  const [targets, setTargets] = useState(['fr', 'en'])
  const [model, setModel] = useState('large-v3-turbo')
  const [translationPreset, setTranslationPreset] = useState('fast')
  const [models, setModels] = useState([])
  const [translationPresets, setTranslationPresets] = useState([])
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [segments, setSegments] = useState([])
  const [maxLines, setMaxLines] = useState(4)
  const [fontScale, setFontScale] = useState(1)
  const [volume, setVolume] = useState(0)
  const [threshold, setThreshold] = useState(0.015)
  const socketRef = useRef(null)
  const streamRef = useRef(null)
  const contextRef = useRef(null)
  const workletRef = useRef(null)
  const historyRefs = useRef({})
  const lastVolumeUpdateRef = useRef(0)

  useEffect(() => {
    fetch('/api/models')
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then((data) => {
        setModels(data.models)
        setTranslationPresets(data.translationPresets)
        if (data.models.some((item) => item.id === data.default)) setModel(data.default)
      })
      .catch(() => setError('Le serveur local est indisponible.'))
  }, [])

  useEffect(() => () => stop(), [])

  useEffect(() => {
    const behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
    Object.values(historyRefs.current).forEach((history) => {
      if (history) history.scrollTo({ top: history.scrollHeight, behavior })
    })
  }, [segments])

  function toggleLanguage(code) {
    setTargets((current) => current.includes(code)
      ? current.filter((target) => target !== code)
      : [...current, code])
  }

  function updateSegment(id, update) {
    setSegments((current) => {
      const index = current.findIndex((segment) => segment.id === id)
      const next = index < 0
        ? [...current, { id, translations: {}, ...update }]
        : current.map((segment) => segment.id === id ? { ...segment, ...update } : segment)
      const finals = next.filter((segment) => segment.isFinal).slice(-maxLines)
      const partials = next.filter((segment) => !segment.isFinal).slice(-1)
      return [...finals, ...partials]
    })
  }

  function changeThreshold({ target }) {
    const value = Number(target.value)
    setThreshold(value)
    workletRef.current?.port.postMessage({ type: 'threshold', value })
  }

  function stop() {
    workletRef.current?.disconnect()
    contextRef.current?.close()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    socketRef.current?.close()
    workletRef.current = null
    contextRef.current = null
    streamRef.current = null
    socketRef.current = null
    setVolume(0)
    setStatus((current) => current === 'error' ? current : 'idle')
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {})
    } else {
      document.exitFullscreen().catch(() => {})
    }
  }

  async function start() {
    if (!targets.length) {
      setError('Selectionnez au moins une langue de sortie.')
      return
    }
    setError('')
    setSegments([])
    setStatus('connecting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } })
      const context = new AudioContext({ sampleRate: 16_000 })
      await context.audioWorklet.addModule('/audio-processor.js')
      const source = context.createMediaStreamSource(stream)
      const worklet = new AudioWorkletNode(context, 'pcm-capture', { channelCount: 1 })
      const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/transcribe`)
      socket.binaryType = 'arraybuffer'
      socket.onopen = () => socket.send(JSON.stringify({ type: 'configure', targets, model, translationPreset }))
      socket.onmessage = ({ data }) => {
        const message = JSON.parse(data)
        if (message.type === 'session.ready') setStatus('listening')
        if (message.type === 'transcript.partial' || message.type === 'transcript.final') {
          updateSegment(message.segmentId, {
            text: message.text,
            language: message.language,
            isFinal: message.isFinal,
          })
        }
        if (message.type === 'translation') {
          setSegments((current) => current.map((segment) => segment.id === message.segmentId
            ? { ...segment, translations: { ...segment.translations, [message.targetLanguage]: message.text } }
            : segment))
        }
        if (message.type === 'error') setError(message.message)
      }
      socket.onerror = () => setError('La connexion au serveur local a echoue.')
      socket.onclose = () => {
        if (socketRef.current === socket) stop()
      }
      worklet.port.onmessage = ({ data }) => {
        if (data.type === 'pcm' && socket.readyState === WebSocket.OPEN && socket.bufferedAmount < 64_000) socket.send(data.pcm)
        if (data.type === 'level' && performance.now() - lastVolumeUpdateRef.current > 80) {
          lastVolumeUpdateRef.current = performance.now()
          setVolume(data.level)
        }
      }
      worklet.port.postMessage({ type: 'threshold', value: threshold })
      source.connect(worklet)
      worklet.connect(context.destination)
      streamRef.current = stream
      socketRef.current = socket
      contextRef.current = context
      workletRef.current = worklet
    } catch (cause) {
      stop()
      setError(cause.name === 'NotAllowedError' ? 'L acces au microphone est necessaire.' : 'Impossible de demarrer le microphone.')
      setStatus('error')
    }
  }

  const active = ['connecting', 'listening'].includes(status)
  const blocks = targets.map((code) => ({
    code,
    lines: segments.map((segment) => ({
      text: segment.language === code
        ? segment.text
        : segment.isFinal ? segment.translations[code] ?? 'Traduction...' : '',
      partial: !segment.isFinal,
    })).filter((line) => line.text),
  })).map((block) => ({
    ...block,
    text: block.lines.map((line) => line.text).join(' '),
    partial: block.lines.at(-1)?.partial,
  }))

  if (active) {
    return (
      <main className="dictation">
        <section
          className="dictation-panel"
          aria-live="polite"
          style={{
            gridTemplateRows: `repeat(${blocks.length}, minmax(0, 1fr))`,
            '--block-count': blocks.length,
            '--font-scale': fontScale,
          }}
        >
          <nav className="dictation-controls" aria-label="Controles">
            <button
              className="dictation-btn"
              onClick={() => setFontScale((s) => Math.max(0.5, Number((s - 0.1).toFixed(1))))}
              title="Diminuer la taille du texte"
              type="button"
            >
              A-
            </button>
            <button
              className="dictation-btn"
              onClick={() => setFontScale((s) => Math.min(3, Number((s + 0.1).toFixed(1))))}
              title="Agrandir la taille du texte"
              type="button"
            >
              A+
            </button>
            <button
              className="dictation-btn"
              onClick={toggleFullscreen}
              title="Plein ecran"
              type="button"
            >
              ⛶
            </button>
            <button
              className="dictation-btn dictation-stop"
              onClick={stop}
              type="button"
            >
              Arreter
            </button>
          </nav>
          {blocks.map((block) => {
            const item = language(block.code)
            return <article className="language-block" key={block.code} lang={block.code}>
              <header className="block-header">
                <span aria-hidden="true" className="flag">{item.flag}</span>
                <span className="row-label">{item.name}</span>
              </header>
              <div className="language-history" ref={(element) => { historyRefs.current[block.code] = element }}>
                {block.text
                  ? <p className={block.partial ? 'partial' : ''}>{block.text}</p>
                  : <p className="waiting">En attente...</p>}
              </div>
            </article>
          })}
        </section>
        <footer className="live-meter">
          <span>Niveau micro</span>
          <div aria-label="Niveau et seuil du microphone" className="meter-track">
            <i style={{ width: `${Math.min(100, volume / 0.05 * 100)}%` }} />
            <b style={{ left: `${threshold / 0.05 * 100}%` }} />
          </div>
          <label>Seuil
            <input aria-label="Seuil de filtrage microphone" max="0.05" min="0.003" onChange={changeThreshold} step="0.001" type="range" value={threshold} />
          </label>
          <output>{(volume * 100).toFixed(1)}%</output>
        </footer>
        {error && <p className="dictation-error" role="alert">{error}</p>}
      </main>
    )
  }

  return (
    <main className="setup">
      <header><h1>Traduction locale</h1><p>ASR et traduction restent sur cet ordinateur.</p></header>
      <section className="settings" aria-label="Configuration de la traduction">
        <label className="field">Modele ASR
          <select disabled={!models.length} onChange={({ target }) => setModel(target.value)} value={model}>
            {!models.length && <option>Chargement des modeles...</option>}
            {models.map((item) => <option key={item.id} value={item.id}>{item.label} - {item.description}</option>)}
          </select>
        </label>
        <label className="field">Preset de traduction
          <select disabled={!translationPresets.length} onChange={({ target }) => setTranslationPreset(target.value)} value={translationPreset}>
            {translationPresets.map((item) => <option key={item.id} value={item.id}>{item.label} - {item.model}</option>)}
          </select>
        </label>
        <fieldset>
          <legend>Langues de sortie</legend>
          <div className="languages">
            {LANGUAGES.map((item) => <label className="language" key={item.code}>
              <input checked={targets.includes(item.code)} onChange={() => toggleLanguage(item.code)} type="checkbox" />
              <span>{item.flag} {item.name}</span>
            </label>)}
          </div>
        </fieldset>
        <label className="field">Lignes conservees
          <input max="20" min="1" onChange={({ target }) => setMaxLines(Math.max(1, Math.min(20, Number(target.value) || 1)))} type="number" value={maxLines} />
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="start" disabled={!models.length || !translationPresets.length} onClick={start} type="button">Demarrer</button>
      </section>
    </main>
  )
}
