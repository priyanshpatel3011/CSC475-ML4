/* ── app.js ─────────────────────────────────────────────────────── */
let wavesurfer = null;
let wsRegions = null;
let currentAudioFile = null;

// Live spectrum analyser state
let analyzerCtx = null;
let analyserNode = null;
let sourceNode = null;
let spectrumCanvas, spectrumCtx;
let animFrameId = null;

// Algorithm display names
const ALGO_NAMES = {
    autocorrelation: 'Autocorrelation',
    state_space: 'State-Space Model',
    realtime: 'Realtime BPM Analyzer'
};

document.addEventListener('DOMContentLoaded', () => {
    // ── Canvas setup ────────────────────────────────────────────
    spectrumCanvas = document.getElementById('live-analyzer');
    spectrumCtx = spectrumCanvas.getContext('2d');

    function resizeCanvas() {
        const container = spectrumCanvas.parentElement;
        spectrumCanvas.width = container.clientWidth;
        spectrumCanvas.height = container.clientHeight;
    }
    window.addEventListener('resize', resizeCanvas);
    // Initial size after a tick to let CSS render
    setTimeout(resizeCanvas, 50);

    // ── WaveSurfer (Panel 1 — waveform + beat markers) ──────────
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: 'rgba(34, 211, 238, 0.45)',
        progressColor: 'rgba(99, 102, 241, 0.75)',
        cursorColor: '#ffffff',
        barWidth: 2,
        barGap: 1,
        barRadius: 3,
        cursorWidth: 1,
        height: 120,
        normalize: true,
        responsive: true,
        fillParent: true,
        hideScrollbar: true,    // WaveSurfer v7 built-in flag
        plugins: []
    });

    wsRegions = wavesurfer.registerPlugin(WaveSurfer.Regions.create());

    // ── File selection → auto-analyse ───────────────────────────
    document.getElementById('audio-input').addEventListener('change', (e) => {
        if (e.target.files.length === 0) return;
        currentAudioFile = e.target.files[0];
        const url = URL.createObjectURL(currentAudioFile);

        wavesurfer.load(url);
        wsRegions.clearRegions();
        document.getElementById('tempo-val').innerText = '--';
        document.getElementById('algo-label').innerText = '—';

        // Auto-trigger analysis and kill scrollbar after render
        wavesurfer.once('ready', () => {
            suppressWaveformScroll();
            runAnalysis();
        });
    });

    // ── Playback controls ───────────────────────────────────────
    wavesurfer.on('play', () => startSpectrum());
    wavesurfer.on('pause', () => stopSpectrum());

    document.getElementById('play-btn').addEventListener('click', () => {
        if (currentAudioFile) wavesurfer.playPause();
    });

    document.getElementById('zoom-slider').addEventListener('input', (e) => {
        if (wavesurfer) {
            wavesurfer.zoom(Number(e.target.value));
            // Re-apply after each zoom because WaveSurfer rebuilds the canvas
            suppressWaveformScroll();
        }
    });

    // ── Analyze button (re-analysis) ────────────────────────────
    document.getElementById('analyze-btn').addEventListener('click', () => {
        if (!currentAudioFile) {
            alert('Please select an audio file first.');
            return;
        }
        runAnalysis();
    });
});

/* ── Kill WaveSurfer's internal scrollbar completely ────────────── */
function suppressWaveformScroll() {
    const container = document.getElementById('waveform');
    if (!container) return;

    // Inject one-time global style (works on WaveSurfer's dynamically-added elements)
    if (!document.getElementById('ws-noscroll-style')) {
        const style = document.createElement('style');
        style.id = 'ws-noscroll-style';
        style.textContent = [
            '#waveform *::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }',
            '#waveform * { scrollbar-width: none !important; overflow: hidden !important; }'
        ].join('\n');
        document.head.appendChild(style);
    }

    // Belt-and-suspenders: also forcibly set inline style on all current children
    container.querySelectorAll('*').forEach(el => {
        el.style.overflow = 'hidden';
        el.style.scrollbarWidth = 'none';
    });
}

/* ── Live Frequency Spectrum (Panel 2) ────────────────────────── */
function ensureAudioGraph() {
    if (analyzerCtx) return;    // already wired

    analyzerCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyserNode = analyzerCtx.createAnalyser();
    analyserNode.fftSize = 256;
    analyserNode.smoothingTimeConstant = 0.8;

    const mediaEl = wavesurfer.getMediaElement();
    sourceNode = analyzerCtx.createMediaElementSource(mediaEl);
    sourceNode.connect(analyserNode);
    analyserNode.connect(analyzerCtx.destination);
}

function startSpectrum() {
    ensureAudioGraph();
    if (analyzerCtx.state === 'suspended') analyzerCtx.resume();
    drawSpectrum();
}

function stopSpectrum() {
    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }
}

function drawSpectrum() {
    animFrameId = requestAnimationFrame(drawSpectrum);

    const bufLen = analyserNode.frequencyBinCount;
    const data = new Uint8Array(bufLen);
    analyserNode.getByteFrequencyData(data);

    const W = spectrumCanvas.width;
    const H = spectrumCanvas.height;

    // Fade previous frame for trailing effect
    spectrumCtx.fillStyle = 'rgba(0, 0, 0, 0.25)';
    spectrumCtx.fillRect(0, 0, W, H);

    const barW = Math.max(1, (W / bufLen) * 2);
    let x = 0;

    for (let i = 0; i < bufLen; i++) {
        const v = data[i] / 255;
        const barH = v * H;

        // Gradient per bar: cyan → pink
        const grad = spectrumCtx.createLinearGradient(0, H, 0, H - barH);
        grad.addColorStop(0, 'rgba(34, 211, 238, 0.6)');
        grad.addColorStop(1, 'rgba(236, 72, 153, 0.9)');

        spectrumCtx.fillStyle = grad;
        spectrumCtx.fillRect(x, H - barH, barW, barH);

        x += barW + 1;
    }
}

/* ── Analysis Router ──────────────────────────────────────────── */
async function runAnalysis() {
    const algorithm = document.getElementById('algorithm-select').value;

    document.getElementById('loader').style.display = 'flex';
    document.getElementById('analyze-btn').disabled = true;
    document.getElementById('tempo-val').innerText = '…';
    document.getElementById('algo-label').innerText = ALGO_NAMES[algorithm] || algorithm;
    wsRegions.clearRegions();

    try {
        if (algorithm === 'realtime') {
            await analyzeWithRealtimeBPM();
        } else {
            await analyzeWithBackend(algorithm);
        }
    } catch (err) {
        console.error('Analysis failed:', err);
        alert('Analysis failed: ' + err.message);
        document.getElementById('tempo-val').innerText = '--';
    } finally {
        document.getElementById('loader').style.display = 'none';
        document.getElementById('analyze-btn').disabled = false;
    }
}

/* ── Realtime BPM (browser-side, offline buffer) ──────────────── */
async function analyzeWithRealtimeBPM() {
    const { analyzeFullBuffer } = await import('https://esm.run/realtime-bpm-analyzer');

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuf = await currentAudioFile.arrayBuffer();
    const audioBuf = await ctx.decodeAudioData(arrayBuf);

    const result = await analyzeFullBuffer(audioBuf);

    if (result && result.length > 0 && result[0].tempo !== undefined) {
        document.getElementById('tempo-val').innerText = result[0].tempo.toFixed(1);
    } else {
        document.getElementById('tempo-val').innerText = '--';
    }
}

/* ── Backend algorithms (Autocorrelation / State-Space) ────────── */
async function analyzeWithBackend(algorithm) {
    const fd = new FormData();
    fd.append('file', currentAudioFile);

    const res = await fetch(`/analyze/${algorithm}`, { method: 'POST', body: fd });

    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || res.statusText);
    }

    const data = await res.json();
    document.getElementById('tempo-val').innerText = data.tempo.toFixed(1);

    // Plot beat markers on the waveform
    data.beat_times.forEach(t => {
        wsRegions.addRegion({
            start: t,
            end: t + 0.04,
            color: 'rgba(236, 72, 153, 0.55)',
            drag: false,
            resize: false
        });
    });
}
