// ==============================
// ELEMENTS
// ==============================
const video        = document.getElementById('video');
const canvas       = document.getElementById('canvas');
const ctx          = canvas.getContext('2d');
const predictionEl = document.getElementById('prediction');
const wordEl       = document.getElementById('word');
const statusTextEl = document.getElementById('statusText');
const hudLetterEl  = document.getElementById('hudLetter');
const confFillEl   = document.getElementById('confFill');
const confPctEl    = document.getElementById('confPct');
const holdRingEl   = document.getElementById('holdRing');
const historyList  = document.getElementById('historyList');
const fpsTagEl     = document.getElementById('fpsTag');
const camFooterEl  = document.getElementById('camFooter');
const camIndicator = document.getElementById('camIndicator');
const waveCanvas   = document.getElementById('waveCanvas');
const waveCtx      = waveCanvas.getContext('2d');
const btnBackspace = document.getElementById('btnBackspace');
const btnSpace     = document.getElementById('btnSpace');
const btnSpeak     = document.getElementById('btnSpeak');
const btnClear     = document.getElementById('btnClear');

// Ring circumference = 2 * PI * 52
const RING_CIRCUMFERENCE = 326.56;

// ==============================
// STATE
// ==============================
let formedWord      = "";
let lastPrediction  = "";
let stableStartTime = null;
let noHandTimer     = null;
let isSpeaking      = false;
let isHandPresent   = false;
let lastFrameTime   = Date.now();
let frameCount      = 0;

const LETTER_HOLD_TIME = 1000;
const RESET_DELAY      = 2500;

// ==============================
// CAMERA
// ==============================
navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 1280, height: 720 } })
  .then(stream => {
    video.srcObject = stream;
    camFooterEl.textContent = 'Camera ready — show a sign!';
    statusTextEl.textContent = 'Live & ready!';
    camIndicator.classList.add('online');
  })
  .catch(err => {
    console.error(err);
    camFooterEl.textContent = 'Camera error: ' + err.message;
    statusTextEl.textContent = 'Camera problem';
    camIndicator.classList.add('error');
  });

// ==============================
// SPEECH
// ==============================
function speak(text) {
  if (!text.trim() || isSpeaking) return;
  isSpeaking = true;
  statusTextEl.textContent = '🔊 Speaking…';
  speechSynthesis.cancel();

  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = 0.92; utt.pitch = 1; utt.volume = 1;
  utt.onend = () => {
    isSpeaking = false;
    statusTextEl.textContent = 'Live & ready!';
  };
  speechSynthesis.speak(utt);
  addHistory(text);
}

// ==============================
// HISTORY
// ==============================
function addHistory(text) {
  const empty = historyList.querySelector('.history-empty');
  if (empty) empty.remove();

  const now = new Date();
  const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const li = document.createElement('li');
  li.innerHTML = `<span>${escHtml(text)}</span><span class="h-time">${time}</span>`;
  historyList.prepend(li);
  while (historyList.children.length > 20) historyList.removeChild(historyList.lastChild);
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ==============================
// WORD DISPLAY
// ==============================
function renderWord() {
  if (!formedWord) {
    wordEl.innerHTML = '<span class="word-hint">Hold a sign to begin!</span>';
    return;
  }
  wordEl.innerHTML = formedWord
    .split('')
    .map((ch, i) => `<span class="word-char" style="animation-delay:${i * 0.01}s">${ch === ' ' ? '&nbsp;' : escHtml(ch)}</span>`)
    .join('');
}

// ==============================
// RESET
// ==============================
function resetDetection() {
  formedWord = ""; lastPrediction = ""; stableStartTime = null;
  setHoldProgress(0);
  predictionEl.textContent = '–';
  hudLetterEl.textContent  = '?';
  confFillEl.style.width   = '0%';
  confPctEl.textContent    = '–';
  renderWord();
}

// ==============================
// HOLD RING PROGRESS
// ==============================
function setHoldProgress(pct) {
  const offset = RING_CIRCUMFERENCE * (1 - Math.min(1, Math.max(0, pct / 100)));
  holdRingEl.style.strokeDashoffset = offset;
}

// ==============================
// FPS
// ==============================
let fpsLastTime = Date.now();
let fpsTick = 0;

function updateFps() {
  fpsTick++;
  const now = Date.now();
  if (now - fpsLastTime >= 1000) {
    fpsTagEl.textContent = fpsTick + ' fps';
    fpsTick = 0;
    fpsLastTime = now;
  }
}

// ==============================
// WAVE
// ==============================
const waveData = new Array(80).fill(0);

function drawWave() {
  const W = waveCanvas.offsetWidth;
  const H = waveCanvas.offsetHeight;
  if (waveCanvas.width !== W) waveCanvas.width = W;
  if (waveCanvas.height !== H) waveCanvas.height = H;

  waveCtx.clearRect(0, 0, W, H);

  const amp = isHandPresent
    ? (0.45 + Math.random() * 0.55)
    : (0.04 + Math.random() * 0.07);

  waveData.push(amp);
  if (waveData.length > 80) waveData.shift();

  const step = W / (waveData.length - 1);
  const mid  = H / 2;

  waveCtx.beginPath();
  for (let i = 0; i < waveData.length; i++) {
    const x = i * step;
    const y = mid - waveData[i] * (H / 2 - 4);
    if (i === 0) waveCtx.moveTo(x, y);
    else {
      const px = (i - 1) * step;
      const py = mid - waveData[i - 1] * (H / 2 - 4);
      waveCtx.bezierCurveTo((px + x) / 2, py, (px + x) / 2, y, x, y);
    }
  }

  waveCtx.strokeStyle = isHandPresent ? '#ff7c3e' : '#e0c8b0';
  waveCtx.lineWidth = 2.5;
  waveCtx.lineJoin = 'round';
  waveCtx.stroke();
}

setInterval(drawWave, 50);

// ==============================
// SEND FRAME
// ==============================
async function sendFrame() {
  updateFps();
  if (video.videoWidth === 0 || video.videoHeight === 0 || isSpeaking) return;

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');

    try {
      const res = await fetch('http://127.0.0.1:8000/predict', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();

      if (data.success) {
        clearTimeout(noHandTimer);
        isHandPresent = true;
        camFooterEl.textContent = '✋ Hand detected!';
        statusTextEl.textContent = 'Detecting…';

        const letter     = data.prediction;
        const confidence = data.confidence ?? null;

        predictionEl.textContent = letter;
        hudLetterEl.textContent  = letter;

        if (confidence !== null) {
          const pct = Math.round(confidence * 100);
          confFillEl.style.width = pct + '%';
          confPctEl.textContent  = pct + '%';
        }

        const now = Date.now();

        if (letter !== lastPrediction) {
          lastPrediction  = letter;
          stableStartTime = now;
          setHoldProgress(0);
        } else if (stableStartTime) {
          const elapsed = now - stableStartTime;
          setHoldProgress((elapsed / LETTER_HOLD_TIME) * 100);

          if (elapsed >= LETTER_HOLD_TIME) {
            formedWord     += letter;
            stableStartTime = null;
            lastPrediction  = "";
            setHoldProgress(0);
            renderWord();
          }
        }

        noHandTimer = setTimeout(() => {
          isHandPresent = false;
          if (formedWord.trim().length > 0 && !isSpeaking) {
            statusTextEl.textContent = 'Almost done…';
            speak(formedWord.trim());
            setTimeout(resetDetection, 1500);
          }
        }, RESET_DELAY);

      } else {
        isHandPresent = false;
        predictionEl.textContent = '–';
        hudLetterEl.textContent  = '?';
        camFooterEl.textContent  = 'Waiting for a hand…';
        statusTextEl.textContent = 'Live & ready!';
        confFillEl.style.width   = '0%';
        confPctEl.textContent    = '–';
        lastPrediction  = "";
        stableStartTime = null;
        setHoldProgress(0);
      }

    } catch (err) {
      console.error(err);
      camFooterEl.textContent  = 'Server offline';
      statusTextEl.textContent = 'Server issue';
      camIndicator.classList.remove('online');
      camIndicator.classList.add('error');
    }
  }, 'image/jpeg', 0.85);
}

setInterval(sendFrame, 80);

// ==============================
// MANUAL CONTROLS
// ==============================
btnBackspace.addEventListener('click', () => {
  if (!formedWord.length) return;
  formedWord = formedWord.slice(0, -1);
  renderWord();
});

btnSpace.addEventListener('click', () => {
  formedWord += ' ';
  renderWord();
});

btnSpeak.addEventListener('click', () => {
  if (formedWord.trim()) { speak(formedWord.trim()); setTimeout(resetDetection, 1500); }
});

btnClear.addEventListener('click', resetDetection);

document.addEventListener('keydown', e => {
  if (document.activeElement.tagName === 'INPUT') return;
  if (e.key === 'Backspace') { e.preventDefault(); btnBackspace.click(); }
  if (e.key === ' ')         { e.preventDefault(); btnSpace.click(); }
  if (e.key === 'Enter')     { btnSpeak.click(); }
  if (e.key === 'Escape')    { btnClear.click(); }
});

// ==============================
// INIT
// ==============================
renderWord();
